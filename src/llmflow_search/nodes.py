"""Graph nodes, routers, and graph assembly."""

import asyncio
import json

from mcp import ClientSession

from . import llm, mcp_client, policy, trace
from . import memory as _memmod
from .answering import answer_node as answer_node
from .answering import verify_node as verify_node
from .config import (
    DISCOVERED_URL_CATALOG_TOP_K,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    LISTING_DRILLDOWN_TOP_K,
    MAX_EVIDENCE_ROUNDS,
    MAX_PARALLEL_FETCHES,
    MAX_PLAN_STEPS,
    MAX_SEARCH_CALLS_PER_QUESTION,
    READ_RANKING,
    SEARCH_GROUP_DELAY_SECONDS,
    SNIPPET_CATALOG_MAX_CHARS,
    SNIPPET_MEMORY_MAX_CHARS,
    group_batch_cap,
)
from .console import print
from .constraints import build_registry, format_for_prompt, open_constraints
from .evidence import (
    _evidence_challenge_schema,
    _evidence_ledger_schema,
    _normalize_evidence_challenge_result,
    _normalize_evidence_ledger_result,
    _numbered_requirements_block,
)
from .llm import _json_loads_best_effort
from .observations import (
    _diagnose_observation_with_model,
    _enrich_sources_with_observation,
    _record_observation,
)
from .passages import PassageStore, SearchRankScorer
from .policy import (
    DISCOVERY_REQUIRED_PREFIXES,
    URL_STEP_PREFIXES,
    RoundView,
)
from .profiles import Profile
from .reports import _assimilate_research
from .requirements import (
    _normalize_requirements,
    _proof_requirements,
)
from .requirements import (
    requirements_node as requirements_node,
)
from .routing import (
    route_after_evidence_challenge as route_after_evidence_challenge,
)
from .routing import (
    route_after_evidence_ledger as route_after_evidence_ledger,
)
from .routing import (
    route_after_evidence_reextract as route_after_evidence_reextract,
)
from .routing import (
    route_after_execute as route_after_execute,
)
from .routing import (
    route_after_plan as route_after_plan,
)
from .routing import (
    route_after_verify as route_after_verify,
)
from .search_memory import (
    _DISCOVERY_SEARCH_TOOLS,
    _append_unique,
    _format_search_memory_for_prompt,
    _merge_search_memory,
    _strategy_plan_from_memory,
    _update_search_memory,
)
from .sources import (
    _effective_question,
    _format_sources_for_llm,
    _merge_sources,
    _normalize_source_url,
    _urls_in_sources,
    _urls_in_text,
)
from .state import AgentState
from .tool_policy import ToolAuthorizer, authorize_tool_call
from .tool_steps import (
    _step_identity,
    _strip_spurious_head,
    _tool_call_from_schema_step,
)

# Every JSON-producing node decodes against its schema, not against a prompt asking
# nicely for JSON: a malformed field becomes impossible rather than merely unlikely,
# and the best-effort parser stops silently substituting defaults. Verdict fields come
# last in each schema because property order is generation order.

POST_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "decision": {"type": "string", "enum": ["DONE", "CONTINUE", "NEXT"]},
    },
    "required": ["reason", "decision"],
}


STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "search_hypothesis": {"type": "string"},
        "failure_diagnosis": {"type": "string"},
        "mutation_dimension": {"type": "string"},
        "exhausted_direction": {"type": "string"},
        "next_queries": {"type": "array", "items": {"type": "string"}},
        "strategy_note": {"type": "string"},
    },
    "required": [
        "search_hypothesis",
        "failure_diagnosis",
        "mutation_dimension",
        "next_queries",
    ],
}



PLAN_STEPS_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "arg": {"type": "string"},
                },
                "required": ["tool", "arg"],
            },
        },
    },
    "required": ["steps"],
}



def _insufficient_evidence_update(
    reason: str, gaps: list[str], audit: dict, state: AgentState
) -> dict:
    notes = [reason] if reason else []
    return {
        "verification_result": {
            "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "claim_checks": [],
            "task_complete": False,
            "coverage": {"overall_status": "missing"},
            "gaps": list(dict.fromkeys(gaps)),
            "notes": notes,
            "insufficient_evidence": True,
            "audit": audit,
        },
        "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
        "evidence_round": state.get("evidence_round", 0) + 1,
    }


def _steps_from_plan_objects(content: str) -> list[str]:
    """Convert the planner output into canonical 'tool: arg' step strings.

    Schema-constrained decoding (GGUF/llama.cpp) returns {"steps":[{tool,arg}]},
    but MLX models ignore the schema, so we also accept a bare list of step
    objects, a single step object, or the legacy list of 'tool: arg' strings.
    When we build the string from a {tool,arg} object the format is always clean.
    """
    parsed = _json_loads_best_effort(content, None)

    # Normalize to a list of items (objects or strings).
    if isinstance(parsed, dict):
        if isinstance(parsed.get("steps"), list):
            items = parsed["steps"]
        elif parsed.get("tool"):
            items = [parsed]  # single bare step object
        else:
            items = []
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    steps: list[str] = []
    for item in items:
        if isinstance(item, dict):
            tool = str(item.get("tool", "")).strip()
            raw_arg = item.get("arguments", item.get("arg", ""))
            arg = (
                json.dumps(raw_arg, ensure_ascii=False, separators=(",", ":"))
                if isinstance(raw_arg, (dict, list))
                else str(raw_arg).strip()
            )
            if tool and arg:
                steps.append(f"{tool}: {arg}")
        elif isinstance(item, str) and ":" in item:
            steps.append(item.strip())  # legacy "tool: arg" string
    return steps


async def plan_node(
    state: AgentState, model: str, tools: list[dict], profile: Profile
) -> dict:
    """Break task into steps on first call, or use replanned steps."""
    task = state["task"]
    plan = state["plan"]
    requirements = _normalize_requirements(
        state.get("requirements_result"), _effective_question(task)
    )
    evidence_round = state.get("evidence_round", 0)
    search_memory = _merge_search_memory(state.get("search_memory"))
    iteration = state["iteration"] + 1

    if profile.uses_search_memory and search_memory.get("search_exhausted"):
        return {"iteration": iteration}

    if plan:
        # already have a plan, use it
        return {"iteration": iteration}

    if (
        profile.uses_search_memory
        and evidence_round
        and search_memory.get("next_queries")
    ):
        question = _effective_question(task)
        live_names = {tool.get("function", {}).get("name") for tool in tools or []}
        steps = _strategy_plan_from_memory(question, search_memory, live_names)
        print(
            f"\n  [PLAN] Using evolved search strategy (retry {evidence_round}/{MAX_EVIDENCE_ROUNDS})"
        )
        print(f"  [PLAN] {len(steps)} steps:")
        for i, s in enumerate(steps, 1):
            print(f"    {i}. {s}")
        search_memory["next_queries"] = []
        return {"plan": steps, "search_memory": search_memory, "iteration": iteration}

    if evidence_round and profile.uses_search_memory:
        question = _effective_question(task)
        planning_input = f"""Task: {question}

Task requirements:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

The previous answer attempt did not have enough supported evidence.
Additional evidence retry: {evidence_round} of {MAX_EVIDENCE_ROUNDS}.

Search memory:
{_format_search_memory_for_prompt(search_memory)}

Create a new plan of discovery steps only — do NOT add steps that fetch a specific URL
(web_read and the other url-argument tools), because no new URL is known yet.
Change the access path, not only the wording: when plain web_search already failed, reach
for the discovery tool in the catalog that matches the subject (scholarly, code, reference,
recency-filtered, archived) instead of rephrasing the same query again."""
        print(
            f"\n  [PLAN] Searching for more evidence (retry {evidence_round}/{MAX_EVIDENCE_ROUNDS})"
        )
    elif evidence_round:
        # generic: replan from scratch, nudged by the reflection on why the last try fell short
        question = _effective_question(task)
        reflections = search_memory.get("reflections", [])
        reflection_block = (
            "\n\nWhy the last attempt fell short:\n"
            + "\n".join(f"- {r}" for r in reflections[-3:])
            if reflections
            else ""
        )
        planning_input = f"""Task: {question}

Task requirements:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

A previous attempt did not gather enough evidence (retry {evidence_round} of {MAX_EVIDENCE_ROUNDS}).{reflection_block}

Create a fresh plan of steps to gather the needed information using the available tools."""
        print(f"\n  [PLAN] Re-planning (retry {evidence_round}/{MAX_EVIDENCE_ROUNDS})")
    else:
        conv_ctx = state.get("conversation_context", "")
        context_block = (
            f"\nPrior conversation (for context only — do NOT copy into steps):\n{conv_ctx}\n"
            if conv_ctx
            else ""
        )
        planning_input = f"""Task: {task}
{context_block}
Task requirements:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

Create a plan using the available MCP tools."""
        print(f"\n  [PLAN] Breaking down: {task[:80]}")

    planning_input += f"""

LIVE MCP TOOL CATALOG (generated from list_tools; * means required):
{mcp_client._format_tool_catalog(tools)}

Use only tool names from this catalog. For a tool with several required parameters,
put a JSON object encoded as the step's arg string. For a tool with one required
parameter, arg may be that single value."""

    with trace.model_call("plan", planning_input, model):
        response = llm._ollama_chat(
            model,
            [{"role": "user", "content": planning_input}],
            tools=None,
            system=profile.plan,
            format_schema=PLAN_STEPS_SCHEMA,
        )
    content = response.get("content", "{}")

    steps = _steps_from_plan_objects(content)
    if not steps:
        live_names = {tool.get("function", {}).get("name") for tool in tools}
        steps = [profile.fallback_step(task)] if "web_search" in live_names else [task]

    print(f"  [PLAN] {len(steps)} steps:")
    for i, s in enumerate(steps, 1):
        print(f"    {i}. {s}")

    return {"plan": steps, "iteration": iteration}


def _drop_undiscovered_url_steps(
    steps: list[str],
    known_urls: set[str],
    prefixes: tuple[str, ...] = URL_STEP_PREFIXES,
) -> tuple[list[str], list[str]]:
    """Keep only fetch steps whose URL the run has actually seen.

    The planner composes plausible government, reference and vendor addresses; they
    resolve to 404 stubs and spend a fetch on nothing. The same rule applies to steps the
    post-batch controller proposes, but there it is one of ``policy.admissible_steps``'s
    filters — this is the planner's copy, applied before any search has run.
    """
    kept: list[str] = []
    invented: list[str] = []
    for step in steps:
        prefix = next((p for p in prefixes if step.startswith(p)), "")
        if not prefix:
            kept.append(step)
            continue
        url = _normalize_source_url(step[len(prefix) :].strip())
        if url and url in known_urls:
            kept.append(step)
        else:
            invented.append(step)
    return kept, invented


def _publish_registry(
    completion_criteria: list[str], ledger_result: dict, candidate_sources: list[dict]
) -> list[dict]:
    """Rebuild the condition registry from the ledger and put it in the trace.

    Written on every ledger pass, so at any step of a run it is answerable from the trace
    alone which conditions are closed and by which page — a question that previously had
    no answer at all, at any step.
    """
    registry = build_registry(completion_criteria, ledger_result, candidate_sources)
    if registry:
        still_open = open_constraints(registry)
        print(
            f"  [CONDITIONS] {len(registry) - len(still_open)}/{len(registry)} settled"
        )
        for item in still_open[:3]:
            print(f"    open: {str(item.get('text'))[:100]}")
        trace.emit("constraint_registry", constraints=registry)
    return registry


def _rank_read_candidates(
    question: str, catalog: list[str], search_memory: dict
) -> list[str]:
    """Order the addresses a round could read next, best first.

    Everything the run has learned about these pages — the snippet, the title, the rank —
    is already in the passage store, so the choice is made there rather than from a raw
    ordering here. ``READ_RANKING`` picks which comparison the store applies.
    """
    scorer = SearchRankScorer() if READ_RANKING == "search_rank" else None
    store = PassageStore(scorer=scorer).add_search_snippets(search_memory)
    # Reversed, so that URLs the store cannot separate fall back to most-recent-first:
    # equal rank means top hits of different queries, and the most recent query is the
    # one whose wording the run refined toward.
    return store.rank_urls(question, list(reversed(catalog)), origin="search_snippet")


def _emit_evidence_admitted(
    candidate_sources: list[dict], admitted: list[dict], ledger_result: dict
) -> None:
    """Record which fetched pages actually ended up supporting a claim.

    This is what turns "pages read" into a precision figure: a read that no ledger row
    ever cites was a page the run paid for and could not use.
    """
    if trace.active() is None:
        return
    admitted_urls = {
        _normalize_source_url(str(source.get("url") or ""))
        for source in admitted
        if isinstance(source, dict)
    }
    trace.emit(
        "evidence_admitted",
        admitted=sorted(url for url in admitted_urls if url),
        candidates=sorted(
            {
                _normalize_source_url(str(source.get("url") or ""))
                for source in candidate_sources
                if isinstance(source, dict) and source.get("url")
            }
        ),
        supported_claims=ledger_result.get("supported_claim_count", 0),
        answer_ready=bool(ledger_result.get("answer_ready")),
    )


def _emit_search_results(tool_name: str, args: dict, tool_result: str) -> None:
    """Record what a discovery call returned, in rank order, with its snippets.

    This is the ground truth for "did retrieval find the answer at all" — the question
    that separates a retrieval failure from a selection failure, and the one the run's
    final score cannot distinguish.
    """
    if tool_name not in _DISCOVERY_SEARCH_TOOLS or trace.active() is None:
        return
    payload = _json_loads_best_effort(tool_result, {})
    if not isinstance(payload, dict):
        return
    records = list(payload.get("results") or []) + list(payload.get("sources") or [])
    trace.emit(
        "search_results",
        tool=tool_name,
        query=str(args.get("query", "") or args.get("url", "")),
        results=[
            {
                "rank": rank,
                "url": _normalize_source_url(str(record.get("url") or "")),
                "title": str(record.get("title") or "")[:200],
                "snippet": " ".join(
                    str(
                        record.get("snippet")
                        or record.get("description")
                        or record.get("text")
                        or ""
                    ).split()
                )[:SNIPPET_MEMORY_MAX_CHARS],
            }
            for rank, record in enumerate(records)
            if isinstance(record, dict)
        ],
    )


def _emit_read_decision(
    by: str,
    selected: list[str],
    catalog: list[str],
    ranks: dict,
) -> None:
    """Record which pages a round chose to open, on whose authority, and what it passed over.

    "The agent read 3 of 140 documents" is the finding that matters most, and it cannot be
    recovered afterwards from a score or from prose in a log: the pages *not* read leave no
    trace at all unless the decision itself is written down.
    """
    chosen = set(selected)
    trace.emit(
        "read_decision",
        by=by,
        selected=[
            {"url": url, "rank": ranks.get(url)} for url in selected
        ],
        passed_over=[
            {"url": url, "rank": ranks.get(url)}
            for url in catalog
            if url not in chosen
        ][:40],
        catalog_size=len(catalog),
    )


def _names_a_live_tool(step: str, tools: list[dict]) -> bool:
    head = step.split(":", 1)[0].strip()
    return head in {tool.get("function", {}).get("name") for tool in tools or []}


async def _execute_single_step(
    step: str,
    model: str,
    tools: list[dict],
    mcp_session: ClientSession | None,
    question: str,
    requirements: dict,
    profile: Profile,
    source_context: dict | None = None,
    tool_authorizer: ToolAuthorizer | None = None,
) -> tuple[str, str, list, dict]:
    """Execute one resolved step. Returns (step, result_text, sources, search_memory_updates)."""
    # A planner occasionally wraps a step as "tool: web_read: <url>"; normalize so the
    # executed/recorded form matches every other step of the same call.
    step = _strip_spurious_head(step)
    # Prefer the live schema registry. This supports every tool returned by list_tools,
    # including tools added after this client was released. Profile-specific parsing is
    # retained only for legacy/Python-like step syntax.
    deterministic_tool_call = _tool_call_from_schema_step(step, tools)
    if not deterministic_tool_call:
        deterministic_tool_call = profile.tool_call_from_step(step)
        if deterministic_tool_call and tools:
            live_names = {tool.get("function", {}).get("name") for tool in tools}
            if (
                deterministic_tool_call.get("function", {}).get("name")
                not in live_names
            ):
                deterministic_tool_call = None
    if deterministic_tool_call:
        tool_calls = [deterministic_tool_call]
        response: dict = {}
    elif tools and profile.uses_search_memory and not _names_a_live_tool(step, tools):
        # A step whose head is not one of the live tools is malformed, not merely oddly
        # phrased — steps here are always "<tool>: <argument>". Handing it to the model
        # to improvise is how "tool: web_navigate" became a real call with no arguments.
        print("✗ unknown tool in step")
        return (
            step,
            f"\n[step skipped: '{step[:60]}' does not name a live tool]\n",
            [],
            {},
        )
    else:
        with trace.model_call("execute", step, model):
            # Same reason as the observation call below: a synchronous client called
            # directly from a coroutine stalls every other step in the batch.
            response = await asyncio.to_thread(
                llm._ollama_chat,
                model,
                [
                    {
                        "role": "user",
                        "content": f"Execute this step using ONE tool call: {step}",
                    }
                ],
                tools=tools,
                system=profile.execute,
            )
        tool_calls = response.get("tool_calls", [])

    if len(tool_calls) > 1:
        # The execution contract is exactly one action per plan step. Aggregating a model's
        # surprise batch here would bypass per-call budgets, authorization, and provenance.
        return (
            step,
            f"\n[step rejected: model returned {len(tool_calls)} tool calls; exactly one is allowed]\n",
            [],
            {},
        )

    result_text = ""
    step_sources: list = []
    sm_updates: dict = {}

    if not tool_calls:
        content = response.get("content", "")
        if content:
            result_text = content
    else:
        for tc in tool_calls:
            name = tc.get("function", tc).get("name", "?")
            args = tc.get("function", tc).get("arguments", tc.get("args", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            print(f"    [{name}]", end=" ", flush=True)
            try:
                await authorize_tool_call(name, args, tools, tool_authorizer)
                # Timed here rather than at the transport: this is the boundary the agent
                # pays for, and it is the one that stays comparable when the transport
                # changes. The rate-limit wait inside is traced separately, so the two can
                # be told apart in the breakdown.
                with trace.span("tool_call", tool=name, args=args) as call_event:
                    tool_result = await mcp_client._call_mcp_tool(
                        name, args, session=mcp_session
                    )
                    call_event["result_chars"] = len(tool_result)
                result_text += f"\n--- {name} ---\n{tool_result}\n"
                sm_updates = {"name": name, "args": args, "result": tool_result}
                tool_payload = _json_loads_best_effort(tool_result, {})
                # Handed to a thread because the Ollama client is synchronous: called
                # directly it blocks the event loop, so a batch of five parallel reads
                # fetched in parallel and then diagnosed one after another. The fetches
                # were concurrent and the model calls behind them never were.
                observation = await asyncio.to_thread(
                    _diagnose_observation_with_model,
                    model,
                    name,
                    args,
                    tool_payload if isinstance(tool_payload, dict) else {},
                    question=question,
                    requirements=requirements,
                    current_step=step,
                )
                sm_updates["observation"] = observation
                step_sources = profile.sources_from_tool_result(
                    name, tool_result, source_context
                )
                step_sources = _enrich_sources_with_observation(
                    step_sources, observation
                )
                print(f"→ {len(tool_result)} chars")
            except TimeoutError:
                # A timed-out long-lived stdio session must escape to its owner. Exiting
                # stdio_client then closes stdin and escalates SIGTERM -> SIGKILL if the
                # child ignores shutdown. Continuing with that session risks reusing a
                # wedged subprocess for every remaining step.
                raise
            except Exception as e:
                result_text += f"\n[{name} error: {e}]\n"
                trace.emit(
                    "tool_call", tool=name, args=args, error=f"{type(e).__name__}: {e}"
                )
                print(f"✗ {str(e)[:60]}")

    return step, result_text, step_sources, sm_updates


async def execute_node(
    state: AgentState,
    model: str,
    tools: list[dict],
    profile: Profile,
    mcp_session: ClientSession | None = None,
    tool_authorizer: ToolAuthorizer | None = None,
) -> dict:
    """Execute a batch of same-tool plan steps in parallel."""
    plan = state["plan"]
    completed = list(state["completed_steps"])
    scratchpad = state["scratchpad"]
    candidate_sources = list(state.get("candidate_sources") or state.get("sources", []))
    sources = list(state.get("sources", []))
    search_memory = _merge_search_memory(state.get("search_memory"))
    question = _effective_question(state["task"])
    requirements = _normalize_requirements(state.get("requirements_result"), question)
    iteration = state["iteration"] + 1
    trace.set_round(state.get("evidence_round", 0))

    if iteration > MAX_PLAN_STEPS * 3:
        return {"iteration": iteration}

    if not plan:
        return {"iteration": iteration}

    # The gate used to run only on steps the post-batch controller proposed, by which
    # point discovery had produced candidates. The planner writes its steps before any
    # search has run, so its addresses were never checked at all — and a plausible
    # invented one spends a fetch and returns an empty page.
    # "Seen" is deliberately broad: anything a search returned, anything already read,
    # anything the user wrote in the task, and anything written inside a page the agent
    # fetched. Narrowing it to search results would block a citation the agent read with
    # its own eyes. What stays blocked is only an address that appears nowhere.
    known_urls = (
        set(search_memory.get("discovered_urls", []))
        | set(search_memory.get("read_urls", []))
        | _urls_in_text(state["task"])
        | _urls_in_sources(candidate_sources)
    )
    plan, invented_in_plan = _drop_undiscovered_url_steps(
        plan, known_urls, DISCOVERY_REQUIRED_PREFIXES
    )
    if invented_in_plan:
        print(
            f"  [EXEC] dropped {len(invented_in_plan)} planned step(s) naming an unseen URL: "
            f"{invented_in_plan[0][:70]}"
        )
    if not plan:
        return {"plan": [], "iteration": iteration}

    # Collect batch: all consecutive steps sharing the same tool prefix
    current_tool = plan[0].split(":", 1)[0].strip()
    batch: list[str] = []
    for step in plan:
        if step.split(":", 1)[0].strip() == current_tool:
            batch.append(step)
        else:
            break
    remaining = plan[len(batch) :]

    # Batch steps are concrete tool calls. Dedup exact repeated steps before execution:
    # repeated searches waste rate-limit budget and repeated reads waste source slots.
    final_steps = list(dict.fromkeys(batch))

    # A rate-limited backend gets one request at a time, and only a few per round.
    # Every tool drawing on a throttled family counts, not just web_search: one call
    # already fans out to several engines inside the server, so a parallel batch of
    # them is a burst of a dozen upstream requests and is what gets us cut off.
    throttle_group = mcp_client._throttle_group(current_tool)
    if throttle_group:
        # Keyed providers bill per call. A question that keeps re-searching without
        # finding anything must hit a wall instead of draining the month's quota.
        spent = int(search_memory.get("search_calls", 0))
        budget_left = max(0, MAX_SEARCH_CALLS_PER_QUESTION - spent)
        if budget_left <= 0:
            search_memory["search_exhausted"] = True
            print(
                f"\n  [EXEC] Search budget spent ({spent}/{MAX_SEARCH_CALLS_PER_QUESTION}) — "
                f"answering from what was already fetched"
            )
            return {
                "plan": [
                    s
                    for s in remaining
                    if mcp_client._throttle_group(s.split(":", 1)[0].strip()) is None
                ],
                "search_memory": search_memory,
                "iteration": iteration,
            }
        if len(final_steps) > budget_left:
            final_steps = final_steps[:budget_left]
            remaining = []
        batch_cap = group_batch_cap(throttle_group)
        if len(final_steps) > batch_cap:
            deferred = final_steps[batch_cap:]
            final_steps = final_steps[:batch_cap]
            remaining = deferred + remaining
            print(
                f"\n  [EXEC] Deferring {len(deferred)} {current_tool} step(s) to the next round"
            )
        search_memory["search_calls"] = spent + len(final_steps)

    # Serializing a batch is what makes the pause between its calls observable; a family
    # that does not pause gains nothing from being run one at a time.
    group_delay = (
        SEARCH_GROUP_DELAY_SECONDS.get(throttle_group, 0.0) if throttle_group else 0.0
    )
    concurrency = (
        1 if group_delay > 0 else max(1, min(len(final_steps), MAX_PARALLEL_FETCHES))
    )
    pacing = f", {group_delay:g}s apart ({throttle_group})" if throttle_group else ""
    print(
        f"\n  [EXEC] Batch ({len(final_steps)} steps, concurrency={concurrency}{pacing}):"
    )
    for s in final_steps:
        print(f"    • {s[:100]}")

    sem = asyncio.Semaphore(concurrency)
    # Browser-session output (web_extract) carries no address of its own; the page the
    # session is on was recorded when web_navigate ran in an earlier batch.
    source_context = {"browser_url": search_memory.get("browser_url", "")}

    async def run_with_sem(_index: int, step: str):
        async with sem:
            return await _execute_single_step(
                step,
                model,
                tools,
                mcp_session,
                question,
                requirements,
                profile,
                source_context,
                tool_authorizer,
            )

    batch_results = await asyncio.gather(
        *[run_with_sem(i, s) for i, s in enumerate(final_steps)]
    )

    # Merge results sequentially into state
    for step, result_text, step_sources, sm_updates in batch_results:
        scratchpad += f"\n## Step: {step}\n{result_text}\n"
        candidate_sources = _merge_sources(candidate_sources, step_sources)
        sources = candidate_sources
        if sm_updates:
            name = sm_updates.get("name", "")
            args = sm_updates.get("args", {})
            tool_result = sm_updates.get("result", "")
            search_memory = _update_search_memory(
                search_memory, name, args, tool_result
            )
            _emit_search_results(name, args, tool_result)
            observation = sm_updates.get("observation")
            if observation:
                search_memory = _record_observation(
                    search_memory, observation, requirements
                )
            if name == "generate_search_queries":
                generated = _json_loads_best_effort(tool_result, {})
                queries = (
                    generated.get("queries", []) if isinstance(generated, dict) else []
                )
                attempted = {
                    q.lower() for q in search_memory.get("attempted_queries", [])
                }
                for query in queries:
                    query = str(query).strip()
                    if query and query.lower() not in attempted:
                        _append_unique(
                            search_memory.setdefault("next_queries", []), query
                        )
        completed.append(
            {
                "step": step,
                "result": result_text,
                "tools_used": [],
            }
        )

    # Post-batch decision: LLM sees what was found and decides next action
    completion_criteria = requirements.get("completion_criteria", [])
    read_urls = [
        c["step"].split(": ", 1)[1]
        for c in completed
        if c["step"].startswith("web_read: ")
    ]
    read_urls_set = set(read_urls)
    drilldown_urls: list[str] = []
    for source in candidate_sources:
        if not isinstance(source, dict) or not source.get("is_listing_page"):
            continue
        for link in source.get("candidate_links", []):
            link_url = str(link.get("url", "")) if isinstance(link, dict) else ""
            if not link_url or link_url in read_urls_set:
                continue
            if link_url not in drilldown_urls:
                drilldown_urls.append(link_url)
    drilldown_urls = drilldown_urls[:LISTING_DRILLDOWN_TOP_K]
    if drilldown_urls:
        print(
            f"  [DRILLDOWN] listing page detected — {len(drilldown_urls)} article link(s)"
        )
    unhelpful_urls = search_memory.get("avoid_urls", [])[-15:]
    unhelpful_domains = search_memory.get("bad_domains", [])[-15:]
    # The post-batch prompt only ever saw the tail of the scratchpad, so the model
    # was asked to name URLs to read without being shown any. It answered with
    # plausible addresses from its own memory, which 404.
    discovered_titles = search_memory.get("discovered_titles", {})
    discovered_snippets = search_memory.get("discovered_snippets", {})
    discovered_ranks_snapshot = search_memory.get("discovered_ranks", {})
    unread_urls = [
        url
        for url in search_memory.get("discovered_urls", [])
        if url not in read_urls_set and url not in unhelpful_urls
    ]
    readable_catalog = unread_urls[-DISCOVERED_URL_CATALOG_TOP_K:]
    # The registry goes in whole, ahead of the scratchpad tail. The tail says what the
    # last few steps printed; it cannot say which conditions the run has already closed,
    # so a round would go looking again for something an earlier round had established.
    registry = list(state.get("constraint_registry") or [])
    still_open = open_constraints(registry)

    def build_post_input() -> str:
        return (
            f"Task: {question}\n\n"
            f"Completion criteria (ALL must be met before DONE):\n"
            + "\n".join(f"  - {c}" for c in completion_criteria)
            + (
                "\n\nCONDITION REGISTRY — what is already settled and by which source.\n"
                "Do not spend another step on a condition marked OK; spend it on one marked ?? or PART.\n"
                + format_for_prompt(registry)
                if registry
                else ""
            )
            + f"\n\nLast batch executed: {len(final_steps)} {current_tool} steps\n"
            f"Steps completed so far ({len(completed)} total):\n"
            + "\n".join(f"  - {c['step']}" for c in completed[-8:])
            + "\n\nURLs already read (do NOT revisit these):\n"
            + ("\n".join(f"  - {u}" for u in read_urls) if read_urls else "  (none yet)")
            + (
                "\n\nURLs/domains that proved unhelpful this session (do NOT read these again):\n"
                + "\n".join(f"  - {u}" for u in unhelpful_urls + unhelpful_domains)
                if (unhelpful_urls or unhelpful_domains)
                else ""
            )
            + "\n\nURLs found by search and not yet read — a web_read step MUST name one of these,\n"
            "and any other address will be discarded:\n"
            + (
                "\n".join(
                    f"  - {url}"
                    + (
                        f"  ({discovered_titles[url][:80]})"
                        if discovered_titles.get(url)
                        else ""
                    )
                    + (
                        f"\n      {discovered_snippets[url][:SNIPPET_CATALOG_MAX_CHARS]}"
                        if discovered_snippets.get(url)
                        else ""
                    )
                    for url in readable_catalog
                )
                if readable_catalog
                else "  (none — search again instead of reading)"
            )
            + f"\n\nFetched candidate sources (pages actually read): {len(candidate_sources)}\n"
            f"Remaining plan: {[s[:60] for s in remaining[:4]] if remaining else '(empty — no more steps planned)'}\n\n"
            f"AVAILABLE TOOLS (* means required; a next_step must name one of these):\n"
            f"{mcp_client._format_tool_catalog(tools)}\n\n"
            f"Recent findings:\n{scratchpad[-2500:]}"
        )
    # One decision, made once. The state may leave no choice — a listing page's own
    # links, a search round that queued no read, every condition already settled — and in
    # that case the controller is not asked at all: the old guard chain asked, paid for
    # the answer, and then overrode it.
    view = RoundView(
        remaining=tuple(remaining),
        ranked_unread=tuple(_rank_read_candidates(question, readable_catalog, search_memory)),
        drilldown=tuple(drilldown_urls),
        known_urls=frozenset(known_urls | set(unread_urls) | read_urls_set),
        completed_identities=frozenset(
            _step_identity(str(c.get("step", "")), tools) for c in completed
        ),
        tool_just_run=current_tool,
        has_sources=bool(sources),
        has_read_anything=bool(read_urls_set or candidate_sources),
        open_conditions=len(still_open),
        total_conditions=len(registry),
        tools=tuple(tools or []),
    )

    forced, forced_by = policy.forced_reads(view)
    model_decision = ""
    reason = ""
    post_input = ""
    proposed: list = []
    if policy.should_ask_controller(view, forced):
        post_input = build_post_input()
        with trace.model_call("post_batch", post_input, model):
            post_content = llm._ollama_chat_schema(
                model,
                [{"role": "user", "content": post_input}],
                system=profile.post_batch,
                format_schema=POST_BATCH_SCHEMA,
            )
        post = _json_loads_best_effort(post_content, {"decision": "CONTINUE"})
        model_decision = str(post.get("decision", "CONTINUE")).upper()
        reason = str(post.get("reason", ""))[:500]
        raw_steps = post.get("next_steps")
        proposed = [str(step) for step in raw_steps] if isinstance(raw_steps, list) else []

    action = policy.decide(view, model_decision, proposed, forced, forced_by)

    print(
        f"  [POST-BATCH] {'FINISH' if action.finishes else action.source.upper()}"
        f" — {reason or action.reason}"[:110]
    )
    for step, why in action.rejected:
        print(f"    dropped ({why}): {step[:70]}")
    trace.emit(
        "post_batch_decision",
        input_chars=len(post_input),
        input_excerpt=trace.excerpt(post_input),
        asked_model=bool(post_input),
        model_decision=model_decision,
        decision="FINISH" if action.finishes else "RUN",
        chosen_by=action.source,
        reason=reason or action.reason,
        rejected=[{"step": step[:200], "why": why} for step, why in action.rejected][:8],
        catalog_size=len(readable_catalog),
        remaining_steps=len(action.steps),
        conditions_open=len(still_open),
        conditions_total=len(registry),
    )
    if action.reads:
        _emit_read_decision(
            action.source,
            list(action.reads),
            readable_catalog,
            discovered_ranks_snapshot,
        )
    remaining = list(action.steps)

    return {
        "plan": remaining,
        "completed_steps": completed,
        "scratchpad": scratchpad,
        "candidate_sources": candidate_sources,
        "sources": sources,
        "search_memory": search_memory,
        "iteration": iteration,
    }


async def evidence_ledger_node(
    state: AgentState, model: str, tools: list[dict], profile: Profile
) -> dict:
    """Build the evidence ledger and choose the next tool step when proof is missing."""
    question = _effective_question(state["task"])
    requirements = _normalize_requirements(state.get("requirements_result"), question)
    proof_requirements = _proof_requirements(requirements)
    completion_criteria = list(proof_requirements.get("completion_criteria", []))
    answer_mode = requirements.get("answer_mode", "strict")
    quality_preferences = list(requirements.get("quality_preferences", []))
    candidate_sources = list(state.get("candidate_sources") or state.get("sources", []))
    sources_text, valid_source_ids = _format_sources_for_llm(candidate_sources, question)
    completed = state.get("completed_steps", [])
    previous_ledger = state.get("evidence_ledger_result", {})
    iteration = state["iteration"] + 1

    print(
        f"\n  [LEDGER] Reviewing proof from {len(valid_source_ids)} candidate sources (answer_mode={answer_mode})..."
    )
    if not valid_source_ids:
        ledger_result = _normalize_evidence_ledger_result(
            {
                "answer_ready": False,
                "ledger": [],
                "global_missing": [
                    {
                        "requirement_index": 0,
                        "text": "No fetched tool result can support final-answer claims.",
                    }
                ]
                if completion_criteria
                else [],
                "next_steps": [],
                "reason": "No candidate sources are available.",
            },
            [],
            completion_criteria,
            answer_mode,
        )
        return {
            "evidence_ledger_result": ledger_result,
            "admissible_sources": [],
            "sources": [],
            "iteration": iteration,
        }

    prompt = f"""QUESTION:
{question}

ANSWER_MODE:
{answer_mode}

PROOF_REQUIREMENTS (numbered — use these indices as requirement_index):
{_numbered_requirements_block(completion_criteria)}

PROOF_REQUIREMENTS (full):
{json.dumps(proof_requirements, ensure_ascii=False, indent=2)}

NON_BLOCKING_QUALITY_PREFERENCES:
{json.dumps(quality_preferences, ensure_ascii=False, indent=2)}

AVAILABLE_TOOLS:
{mcp_client._format_tool_catalog(tools)}

COMPLETED_STEPS:
{json.dumps([c.get("step", "") for c in completed[-12:]], ensure_ascii=False, indent=2)}

PREVIOUS_EVIDENCE_LEDGER:
{json.dumps(previous_ledger, ensure_ascii=False, indent=2)[:5000]}

CANDIDATE SOURCES:
{sources_text}

Build the evidence ledger and choose next tool steps if proof is still missing."""

    with trace.model_call("evidence_ledger", prompt, model):
        raw = llm._ollama_chat_schema(
            model,
            [{"role": "user", "content": prompt}],
            system=profile.evidence_ledger,
            format_schema=_evidence_ledger_schema(len(completion_criteria)),
        )
    ledger_result = _normalize_evidence_ledger_result(
        _json_loads_best_effort(raw, {}),
        candidate_sources,
        completion_criteria,
        answer_mode,
        valid_source_ids,
    )
    admitted = ledger_result["admissible_sources"]
    _emit_evidence_admitted(candidate_sources, admitted, ledger_result)
    registry = _publish_registry(completion_criteria, ledger_result, candidate_sources)
    next_steps = ledger_result.get("next_steps", [])
    gaps = [
        item["missing"]
        for item in ledger_result.get("ledger", [])
        if item.get("missing")
    ]
    gaps.extend(ledger_result.get("global_missing", []))
    print(
        "  [LEDGER] "
        f"supported claims {ledger_result['supported_claim_count']}; "
        f"supported sources {len(admitted)}/{len(candidate_sources)}; "
        f"answer_ready={ledger_result['answer_ready']}"
    )
    if gaps:
        print(f"    gaps: {', '.join(gaps[:4])}")
    if ledger_result.get("dropped_gaps"):
        print(
            f"    dropped (no matching requirement): {', '.join(ledger_result['dropped_gaps'][:4])}"
        )
    if next_steps:
        print(f"    next: {next_steps[0][:100]}")

    current_supported = ledger_result["supported_claim_count"]
    last_supported = state.get("last_supported_claim_count", 0)
    stagnant_rounds = (
        0 if current_supported > last_supported else state.get("stagnant_rounds", 0) + 1
    )

    audit = {
        "passed": ledger_result["answer_ready"],
        "gaps": list(dict.fromkeys(gaps)),
        "strategy_hints": next_steps,
        "source_count": len(admitted),
        "candidate_source_count": len(candidate_sources),
        "structured_source_count": sum(
            1
            for source in admitted
            if isinstance(source, dict)
            and source.get("kind")
            in {"table", "file", "json", "recipe_rows", "browser_table"}
        ),
        "ledger": ledger_result,
    }

    update = {
        "evidence_ledger_result": ledger_result,
        "admissible_sources": admitted,
        "sources": admitted,
        "evidence_audit": audit,
        "constraint_registry": registry,
        "iteration": iteration,
        "stagnant_rounds": stagnant_rounds,
        "last_supported_claim_count": current_supported,
    }
    if ledger_result["answer_ready"]:
        update["plan"] = []
    elif next_steps:
        # Identity, not text: the ledger re-proposes the same search worded differently
        # ("web_search: X" vs "web_search: query: 'X' num=10") and a string compare
        # lets it through as if it were a fresh attempt.
        already_done = {
            _step_identity(str(c.get("step", "")), tools) for c in completed
        }
        fresh_steps = [
            step
            for step in next_steps
            if _step_identity(step, tools) not in already_done
        ]
        if fresh_steps and _merge_search_memory(state.get("search_memory")).get(
            "search_exhausted"
        ):
            # Search budget spent: throttled searches can never execute again
            # (execute_node drops them), so re-proposing them just loops until the
            # recursion limit. Keep only non-throttled steps (reads of URLs already
            # discovered) and, if none remain, take the terminal branch.
            fresh_steps = [
                step
                for step in fresh_steps
                if mcp_client._throttle_group(step.split(":", 1)[0].strip()) is None
            ]
            if not fresh_steps:
                update |= _insufficient_evidence_update(
                    ledger_result.get("reason", ""), audit["gaps"], audit, state
                )
            else:
                update["plan"] = fresh_steps + list(state.get("plan", []))
        elif fresh_steps:
            update["plan"] = fresh_steps + list(state.get("plan", []))
        elif (
            len(completed) < MAX_PLAN_STEPS
            and state.get("evidence_round", 0) < MAX_EVIDENCE_ROUNDS
        ):
            # Every proposed next_step duplicates a step already executed — the ledger likely
            # failed to extract facts already present in fetched text, not that more fetching
            # is needed. Route to a re-extraction pass instead of giving up immediately.
            audit["dedup_stalled"] = True
            update["evidence_audit"] = audit
        else:
            update |= _insufficient_evidence_update(
                ledger_result.get("reason", ""), audit["gaps"], audit, state
            )
    else:
        update |= _insufficient_evidence_update(
            ledger_result.get("reason", ""), audit["gaps"], audit, state
        )
    return update


async def evidence_challenge_node(
    state: AgentState, model: str, tools: list[dict], profile: Profile
) -> dict:
    """Challenge the ledger before allowing answer drafting."""
    question = _effective_question(state["task"])
    requirements = _normalize_requirements(state.get("requirements_result"), question)
    proof_requirements = _proof_requirements(requirements)
    completion_criteria = list(proof_requirements.get("completion_criteria", []))
    answer_mode = requirements.get("answer_mode", "strict")
    quality_preferences = list(requirements.get("quality_preferences", []))
    candidate_sources = list(state.get("candidate_sources") or state.get("sources", []))
    sources_text, _valid_source_ids = _format_sources_for_llm(candidate_sources, question)
    completed = state.get("completed_steps", [])
    ledger_result = state.get("evidence_ledger_result", {})
    iteration = state["iteration"] + 1

    print("  [CHALLENGE] Stress-testing evidence ledger...")
    prompt = f"""QUESTION:
{question}

ANSWER_MODE:
{answer_mode}

PROOF_REQUIREMENTS (numbered — use these indices as requirement_index):
{_numbered_requirements_block(completion_criteria)}

PROOF_REQUIREMENTS (full):
{json.dumps(proof_requirements, ensure_ascii=False, indent=2)}

NON_BLOCKING_QUALITY_PREFERENCES:
{json.dumps(quality_preferences, ensure_ascii=False, indent=2)}

AVAILABLE_TOOLS:
{mcp_client._format_tool_catalog(tools)}

COMPLETED_STEPS:
{json.dumps([c.get("step", "") for c in completed[-12:]], ensure_ascii=False, indent=2)}

EVIDENCE_LEDGER:
{json.dumps(ledger_result, ensure_ascii=False, indent=2)[:7000]}

CANDIDATE SOURCES:
{sources_text}

Challenge the ledger and decide whether a final answer is permitted."""

    with trace.model_call("evidence_challenge", prompt, model):
        raw = llm._ollama_chat_schema(
            model,
            [{"role": "user", "content": prompt}],
            system=profile.evidence_challenge,
            format_schema=_evidence_challenge_schema(len(completion_criteria)),
        )
    challenge = _normalize_evidence_challenge_result(
        _json_loads_best_effort(raw, {}), ledger_result, len(completion_criteria)
    )
    next_steps = challenge.get("next_steps", []) or list(
        ledger_result.get("next_steps", []) or []
    )
    gaps = list(challenge.get("blocking_gaps", []))
    if (
        not gaps
        and not challenge["answer_permitted"]
        and ledger_result.get("global_missing")
    ):
        gaps = [str(gap) for gap in ledger_result.get("global_missing", []) if str(gap)]

    print(f"  [CHALLENGE] answer_permitted={challenge['answer_permitted']}")
    if gaps:
        print(f"    blocks: {', '.join(gaps[:4])}")
    if challenge.get("dropped_gaps"):
        print(
            f"    dropped (no matching requirement): {', '.join(challenge['dropped_gaps'][:4])}"
        )
    if next_steps:
        print(f"    next: {next_steps[0][:100]}")

    audit = dict(state.get("evidence_audit", {}) or {})
    audit["passed"] = challenge["answer_permitted"]
    audit["gaps"] = list(dict.fromkeys(gaps))
    audit["strategy_hints"] = next_steps[:4]
    audit["challenge"] = challenge
    audit.pop("dedup_stalled", None)

    update = {
        "evidence_challenge_result": challenge,
        "evidence_audit": audit,
        "iteration": iteration,
    }
    if challenge["answer_permitted"]:
        update["plan"] = []
    elif next_steps:
        # Identity, not text: the ledger re-proposes the same search worded differently
        # ("web_search: X" vs "web_search: query: 'X' num=10") and a string compare
        # lets it through as if it were a fresh attempt.
        already_done = {
            _step_identity(str(c.get("step", "")), tools) for c in completed
        }
        fresh_steps = [
            step
            for step in next_steps
            if _step_identity(step, tools) not in already_done
        ]
        if fresh_steps and _merge_search_memory(state.get("search_memory")).get(
            "search_exhausted"
        ):
            # Search budget spent: throttled searches can never execute again
            # (execute_node drops them), so re-proposing them just loops until the
            # recursion limit. Keep only non-throttled steps (reads of URLs already
            # discovered) and, if none remain, take the terminal branch, which
            # bumps evidence_round and lets routing exit to answer/assimilate.
            fresh_steps = [
                step
                for step in fresh_steps
                if mcp_client._throttle_group(step.split(":", 1)[0].strip()) is None
            ]
            if not fresh_steps:
                update |= _insufficient_evidence_update(
                    challenge.get("reason", ""), audit["gaps"], audit, state
                )
            else:
                update["plan"] = fresh_steps + list(state.get("plan", []))
        elif fresh_steps:
            update["plan"] = fresh_steps + list(state.get("plan", []))
        elif (
            len(completed) < MAX_PLAN_STEPS
            and state.get("evidence_round", 0) < MAX_EVIDENCE_ROUNDS
        ):
            audit["dedup_stalled"] = True
            update["evidence_audit"] = audit
        else:
            update |= _insufficient_evidence_update(
                challenge.get("reason", ""), audit["gaps"], audit, state
            )
    else:
        update |= _insufficient_evidence_update(
            challenge.get("reason", ""), audit["gaps"], audit, state
        )
    return update


async def evidence_reextract_node(
    state: AgentState, model: str, tools: list[dict], profile: Profile
) -> dict:
    """Re-mine already-fetched candidate_sources without issuing new tool calls.

    Reached only when the ledger/challenge's proposed next_steps exactly duplicated steps
    already executed — that means the ledger/challenge failed to extract facts already
    present in fetched text, not that more fetching is needed (see evidence_ledger_node /
    evidence_challenge_node's dedup_stalled branch).
    """
    question = _effective_question(state["task"])
    requirements = _normalize_requirements(state.get("requirements_result"), question)
    proof_requirements = _proof_requirements(requirements)
    completion_criteria = list(proof_requirements.get("completion_criteria", []))
    answer_mode = requirements.get("answer_mode", "strict")
    quality_preferences = list(requirements.get("quality_preferences", []))
    candidate_sources = list(state.get("candidate_sources") or state.get("sources", []))
    sources_text, valid_source_ids = _format_sources_for_llm(candidate_sources, question)
    iteration = state["iteration"] + 1

    print("  [REEXTRACT] Re-mining already-fetched sources (no new tool calls)...")
    prompt = f"""QUESTION:
{question}

ANSWER_MODE:
{answer_mode}

PROOF_REQUIREMENTS (numbered — use these indices as requirement_index):
{_numbered_requirements_block(completion_criteria)}

NON_BLOCKING_QUALITY_PREFERENCES:
{json.dumps(quality_preferences, ensure_ascii=False, indent=2)}

CANDIDATE SOURCES (already fetched — do not propose fetching any of these again):
{sources_text}

The previous evidence ledger could not fully support the requirements from these sources, and its own
proposed next tool steps duplicated steps already executed. Re-read the CANDIDATE SOURCES content
carefully and mine it harder for claims you may have missed the first time. Propose NO next_steps —
build the evidence ledger from what is already fetched only."""

    with trace.model_call("evidence_reextract", prompt, model):
        raw = llm._ollama_chat_schema(
            model,
            [{"role": "user", "content": prompt}],
            system=profile.evidence_ledger,
            format_schema=_evidence_ledger_schema(len(completion_criteria)),
        )
    ledger_result = _normalize_evidence_ledger_result(
        _json_loads_best_effort(raw, {}),
        candidate_sources,
        completion_criteria,
        answer_mode,
        valid_source_ids,
    )
    admitted = ledger_result["admissible_sources"]
    _emit_evidence_admitted(candidate_sources, admitted, ledger_result)
    registry = _publish_registry(completion_criteria, ledger_result, candidate_sources)
    gaps = [
        item["missing"]
        for item in ledger_result.get("ledger", [])
        if item.get("missing")
    ]
    gaps.extend(ledger_result.get("global_missing", []))
    print(
        "  [REEXTRACT] "
        f"supported claims {ledger_result['supported_claim_count']}; "
        f"answer_ready={ledger_result['answer_ready']}"
    )

    audit = dict(state.get("evidence_audit", {}) or {})
    audit["passed"] = ledger_result["answer_ready"]
    audit["gaps"] = list(dict.fromkeys(gaps))
    audit["ledger"] = ledger_result
    audit.pop("dedup_stalled", None)

    update = {
        "evidence_ledger_result": ledger_result,
        "admissible_sources": admitted,
        "sources": admitted,
        "evidence_audit": audit,
        "constraint_registry": registry,
        "iteration": iteration,
    }
    if ledger_result["answer_ready"]:
        update["plan"] = []
        print(
            "  [REEXTRACT] recovered supporting evidence from already-fetched sources"
        )
    else:
        reason = (
            ledger_result.get("reason", "")
            or "Re-mining already-fetched sources did not surface enough support."
        )
        update |= _insufficient_evidence_update(reason, audit["gaps"], audit, state)
    return update



async def assimilate_node(state: AgentState, _model: str, _tools: list[dict]) -> dict:
    """Persist research experience — always, regardless of outcome."""
    iteration = state["iteration"] + 1
    print("  [ASSIMILATE] Recording research experience...")
    _assimilate_research(state)  # type: ignore[arg-type]
    return {"iteration": iteration}


async def strategy_node(
    state: AgentState, model: str, tools: list[dict], profile: Profile
) -> dict:
    """Evolve search queries from live search memory and verifier feedback."""
    iteration = state["iteration"] + 1
    if not profile.uses_search_memory:
        # Generic mode has no web-search query space to evolve; plan_node replans from
        # scratch using the evidence audit already recorded by the ledger/challenge.
        print("  [STRATEGY] Re-planning from scratch (generic mode)...")
        return {"iteration": iteration}

    question = _effective_question(state["task"])
    requirements = _normalize_requirements(state.get("requirements_result"), question)
    memory = _merge_search_memory(state.get("search_memory"))
    verification = state.get("verification_result", {})
    evidence_audit = state.get("evidence_audit", {})
    source_count = len(state.get("sources", []))
    store = _memmod._get_research_store()
    champion = store.best_strategy()

    reflections = memory.get("reflections", [])
    reflection_block = ""
    if reflections:
        reflection_block = f"\nREFLECTION (why previous search failed):\n{chr(10).join(f'- {r}' for r in reflections[-3:])}\n"

    prompt = f"""QUESTION:
{question}

SOURCE_COUNT:
{source_count}

CONDITION REGISTRY (the next query exists to close a ?? or PART line, nothing else):
{format_for_prompt(state.get("constraint_registry"))}

TASK_REQUIREMENTS:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

VERIFIER_FEEDBACK:
{json.dumps(verification, ensure_ascii=False, indent=2)[:3000]}

EVIDENCE_AUDIT:
{json.dumps(evidence_audit, ensure_ascii=False, indent=2)[:3000]}
{reflection_block}
SEARCH_MEMORY:
{_format_search_memory_for_prompt(memory)}

AVAILABLE_DISCOVERY_TOOLS (* means required; a "tool: argument" entry must name one of these):
{mcp_client._format_tool_catalog(tools)}

Propose the next search strategy."""

    print("  [STRATEGY] Evolving search queries...")
    with trace.model_call("strategy", prompt, model):
        response = {
            "content": llm._ollama_chat_schema(
                model,
                [{"role": "user", "content": prompt}],
                system=profile.strategy,
                format_schema=STRATEGY_SCHEMA,
                temperature=0.4,
            )
        }
    result = _json_loads_best_effort(response.get("content", ""), {})
    next_queries = result.get("next_queries") if isinstance(result, dict) else None
    if not isinstance(next_queries, list):
        next_queries = []

    attempted = {query.lower() for query in memory["attempted_queries"]}
    candidates = []
    if champion:
        candidates.append(
            {
                "origin": "exploit",
                "desc": champion.get("desc", ""),
                "success_rate": champion.get("success_rate", 0.0),
            }
        )
    note = result.get("strategy_note") if isinstance(result, dict) else ""
    hypothesis = (
        str(result.get("search_hypothesis") or "").strip()
        if isinstance(result, dict)
        else ""
    )
    failure_diagnosis = (
        str(result.get("failure_diagnosis") or "").strip()
        if isinstance(result, dict)
        else ""
    )
    mutation_dimension = (
        str(result.get("mutation_dimension") or "").strip()
        if isinstance(result, dict)
        else ""
    )
    exhausted_direction = (
        str(result.get("exhausted_direction") or "").strip()
        if isinstance(result, dict)
        else ""
    )
    if hypothesis:
        _append_unique(memory["search_hypotheses"], hypothesis[:500])
    if failure_diagnosis:
        _append_unique(memory["failure_diagnoses"], failure_diagnosis[:500])
    if mutation_dimension:
        _append_unique(memory["mutation_history"], mutation_dimension[:200])
    if exhausted_direction:
        _append_unique(memory["exhausted_directions"], exhausted_direction[:300])
    if note:
        candidates.append({"origin": "explore", "desc": str(note)[:300]})
    candidates.append(
        {
            "origin": "fallback",
            "desc": "Use a different source-discovery angle and fetch evidence before answering.",
        }
    )

    fresh_queries = []
    for query in next_queries:
        query = str(query).strip()
        if query and query.lower() not in attempted and query not in fresh_queries:
            fresh_queries.append(query)

    if not note:
        note = "No fresh search strategy was proposed."
    memory["strategy_notes"].append(str(note)[:500])
    memory["next_queries"] = fresh_queries[:4]
    memory["strategy_candidates"] = candidates
    memory["current_strategy"] = candidates[0]["desc"] if candidates else str(note)

    print(f"  [STRATEGY] Next queries: {len(memory['next_queries'])}")
    for i, query in enumerate(memory["next_queries"], 1):
        print(f"    {i}. {query[:100]}")

    if not memory["next_queries"]:
        memory["search_exhausted"] = True
        audit = dict(evidence_audit or {})
        gaps = list(audit.get("gaps", []) or [])
        if failure_diagnosis:
            gaps.append(failure_diagnosis)
        audit["passed"] = False
        audit["gaps"] = list(dict.fromkeys(gaps))
        update = _insufficient_evidence_update(str(note), audit["gaps"], audit, state)
        update["search_memory"] = memory
        update["iteration"] = iteration
        print("  [STRATEGY] No fresh search direction remains.")
        return update

    if memory["next_queries"] and int(
        memory.get("search_calls", 0)
    ) < MAX_SEARCH_CALLS_PER_QUESTION:
        memory["search_exhausted"] = False
    return {"search_memory": memory, "iteration": iteration}
