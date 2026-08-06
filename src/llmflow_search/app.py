"""Interactive entry point (REPL) that wires the graph to a live MCP session."""

import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import SERVER_CMD
from .console import print
from .llm import pick_model, pop_pending_initial_task
from .mcp_client import _tool_schema_list
from .nodes import build_graph
from .profiles import select_profile
from .reports import _write_debug_report, _write_pdf_report
from .search_memory import _default_search_memory
from .state import AgentState


async def main():
    model = pick_model()

    print(f"Connecting to MCP server ({SERVER_CMD[0]})...", end=" ", flush=True)
    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = _tool_schema_list(await session.list_tools())
                print(f"✓ ({len(tools)} tools)")

                if not tools:
                    print("[!] No tools")
                    sys.exit(1)

                profile = select_profile(t["function"]["name"] for t in tools)
                print(f"  Profile: {profile.name}")
                graph = build_graph(model, tools, mcp_session=session, profile=profile)
                history: list[dict] = []  # conversation memory

                print(f"\n{'='*50}")
                print("  Interactive mode. Type 'exit' to quit.")
                print(f"{'='*50}\n")

                pending_task = pop_pending_initial_task()
                while True:
                    if pending_task:
                        task = pending_task
                        pending_task = ""
                        print(f">>> {task}")
                    else:
                        try:
                            task = input(">>> ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print("\nBye!")
                            break

                    if not task or task.lower() in ("exit", "quit"):
                        print("Bye!")
                        break

                    # build context from history — kept separate from task
                    history_context = ""
                    if history:
                        recent = history[-3:]  # last 3 exchanges
                        history_context = "\n".join(
                            f"Q: {h['q']}\nA: {h['a'][:200]}" for h in recent
                        )

                    print(f"\n{'─'*50}")
                    state: AgentState = {
                        "task": task,  # always clean — no history embedded
                        "conversation_context": history_context,
                        "requirements_result": {},
                        "plan": [],
                        "completed_steps": [],
                        "scratchpad": "",
                        "candidate_sources": [],
                        "admissible_sources": [],
                        "sources": [],
                        "evidence_ledger_result": {},
                        "evidence_challenge_result": {},
                        "draft_result": {},
                        "verification_result": {},
                        "evidence_audit": {},
                        "final_answer": "",
                        "iteration": 0,
                        "replan_count": 0,
                        "evidence_round": 0,
                        "search_memory": _default_search_memory(),
                        "answer_mode": "strict",
                        "stagnant_rounds": 0,
                        "last_supported_claim_count": 0,
                    }

                    try:
                        final = await graph.ainvoke(state, {"recursion_limit": 200})
                    except Exception as e:
                        print(f"\n[!] Error: {e}")
                        continue

                    answer = final.get("final_answer", "")
                    if hasattr(answer, "content"):
                        answer = answer.content

                    history.append({"q": task, "a": answer})
                    debug_report_path = _write_debug_report(final)
                    pdf_report_path = None
                    try:
                        pdf_report_path = _write_pdf_report(final)
                    except Exception as exc:
                        print(f"[!] PDF export failed: {exc}")

                    print(f"\n{answer}\n")
                    sources = final.get("sources", [])
                    if sources:
                        print("Sources:")
                        for i, src in enumerate(sources, 1):
                            title = src.get("title", "").strip()
                            url = src.get("url", "").strip()
                            print(f"  [{i}] {title} — {url}" if title else f"  [{i}] {url}")
                    verdict = final.get("verification_result", {}) or {}
                    gaps = verdict.get("gaps", []) or []
                    notes = verdict.get("notes", []) or []
                    if gaps or notes:
                        print("\nCoverage note (not fully covered by sources):")
                        for g in gaps[:6]:
                            print(f"  - missing: {g}")
                        for n in notes[:4]:
                            print(f"  - note: {n}")
                    if debug_report_path:
                        print(f"Debug report: {debug_report_path}")
                    if pdf_report_path:
                        print(f"PDF report: {pdf_report_path}")
                    print(f"{'─'*50}")
                    print(f"  Steps: {len(final['completed_steps'])} | Type next question or 'exit'")
                    print(f"{'─'*50}")
    except Exception as e:
        print(f"\n[!] Failed: {e}")
        sys.exit(1)
