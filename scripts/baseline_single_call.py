"""The baseline the architecture has to beat: one model, the same tools, no graph.

Every claim this project makes rests on a comparison it never ran. Planning, an evidence
ledger, a challenge pass and a fail-closed verifier are expensive — tens of model calls
and minutes of wall clock per question — and the only honest way to know whether they buy
anything is to give the same model the same tools over the same corpus, let it search and
read in a plain tool-calling loop, and score both with the same scorer.

What is held constant, so that the difference measured is the architecture and not the
wording: the model, the corpus, the MCP server, the tool budget's upper bound, the answer
system prompt, the source rendering (so ``[n]`` means the same thing on both sides) and
the scoring. What is removed: requirement extraction, planning, per-result observation
diagnosis, the evidence ledger, the challenge pass, further evidence rounds, and
verification. The baseline answers from whatever it retrieved, which is what a single
tool-calling agent does.

    uv run --no-sync python scripts/baseline_single_call.py --sample-size 10
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_browsecomp import (  # noqa: E402  (path shim must run first)
    evaluate_answer,
    load_browsecomp_questions,
)

from llmflow_search import eval_records, llm, mcp_client, trace  # noqa: E402
from llmflow_search.config import EVAL_MODEL  # noqa: E402
from llmflow_search.mcp_client import _tool_schema_list  # noqa: E402
from llmflow_search.nodes import _emit_search_results  # noqa: E402
from llmflow_search.profiles import select_profile  # noqa: E402
from llmflow_search.sources import (  # noqa: E402
    _format_sources_for_llm,
    _merge_sources,
    _normalize_source_url,
)

# Upper bound on tool calls the baseline may make. Not a handicap: it is above what the
# loop actually spends (the model stops on its own), and it exists so one confused run
# cannot spin forever against a local corpus that always answers.
DEFAULT_MAX_TOOL_CALLS = 10

# Turns of the tool-calling loop. Each turn may request several tools at once.
DEFAULT_MAX_TURNS = 6

BASELINE_SYSTEM = """You are a web research assistant with search tools.

Research the question using the tools, then answer.
Call web_search to find candidate pages and web_read to read the ones that matter.
Read pages before answering — a search snippet is not evidence.
When you have enough, stop calling tools and reply with the word DONE."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS
    )
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument(
        "--output", type=str, default="reports/baseline_results.json"
    )
    parser.add_argument("--trace", type=str, default=None)
    return parser.parse_args()


async def _run_tool_loop(
    question: str,
    model: str,
    tools: list[dict],
    profile,
    session: ClientSession,
    max_tool_calls: int,
    max_turns: int,
) -> tuple[list[dict], int]:
    """Let the model search and read until it stops, and keep what it retrieved."""
    messages: list[dict] = [{"role": "user", "content": question}]
    sources: list[dict] = []
    tool_calls_made = 0

    for _turn in range(max_turns):
        with trace.model_call("baseline_tool_loop", question, model):
            response = await asyncio.to_thread(
                llm._ollama_chat,
                model,
                messages,
                tools,
                BASELINE_SYSTEM,
            )
        calls = response.get("tool_calls") or []
        if not calls:
            break
        messages.append(response)

        for call in calls:
            if tool_calls_made >= max_tool_calls:
                break
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            args = function.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls_made += 1
            try:
                with trace.span("tool_call", tool=name, args=args) as call_event:
                    result = await mcp_client._call_mcp_tool(
                        name, args, session=session
                    )
                    call_event["result_chars"] = len(result)
                _emit_search_results(name, args, result)
                sources = _merge_sources(
                    sources, profile.sources_from_tool_result(name, result, None)
                )
            except Exception as exc:
                result = f"error: {type(exc).__name__}: {exc}"
                trace.emit("tool_call", tool=name, args=args, error=result)
            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "tool_call_id": call.get("id", ""),
                    "content": result[:20000],
                }
            )
        if tool_calls_made >= max_tool_calls:
            break

    return sources, tool_calls_made


def _answer(question: str, sources: list[dict], model: str, profile) -> str:
    """One drafting call, with the same prompt and source rendering the agent uses.

    Sharing the answer prompt is what makes the two systems comparable: both are asked
    for prose with inline ``[n]`` markers over an identically numbered source list, so a
    difference in the attribution score is a difference in the evidence each system
    brought back, not in how it was told to write.
    """
    sources_text, valid_ids = _format_sources_for_llm(sources, question)
    if not valid_ids:
        return "The found sources do not provide enough information for a reliable answer."
    prompt = f"""QUESTION:
{question}

SOURCES:
{sources_text}

Answer only from SOURCES."""
    with trace.model_call("baseline_answer", prompt, model):
        return (
            llm._ollama_chat(
                model,
                [{"role": "user", "content": prompt}],
                tools=None,
                system=profile.answer_prose,
            )
            .get("content", "")
            .strip()
        )


async def run_single_query(
    record: dict,
    model: str,
    python_bin: str,
    server_script: Path,
    temp_dir: Path,
    answers_path: Path,
    max_tool_calls: int,
    max_turns: int,
) -> dict[str, Any]:
    query_id = str(record.get("query_id", "unknown"))
    question = record.get("query", "")
    ground_truth = record.get("answer", "")
    gold_docs = record.get("gold_docs", []) or []
    corpus = list(gold_docs) + list(record.get("negative_docs", []) or [])
    gold_urls = {doc.get("url") for doc in gold_docs if doc.get("url")}

    corpus_file = temp_dir / f"corpus_{query_id}.json"
    corpus_file.write_text(
        json.dumps(corpus, ensure_ascii=False), encoding="utf-8"
    )
    env = os.environ.copy()
    env["BROWSECOMP_DOCS_JSON"] = str(corpus_file)

    print(f"\n[{query_id}] {question[:90]}...")
    trace.begin_query(
        query_id,
        query=question,
        ground_truth=ground_truth,
        gold_urls=sorted(_normalize_source_url(url) for url in gold_urls),
        corpus_size=len(corpus),
        system="baseline",
    )
    started = time.time()
    try:
        async with stdio_client(
            StdioServerParameters(
                command=python_bin, args=[str(server_script)], env=env
            )
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = _tool_schema_list(await session.list_tools())
                profile = select_profile(t["function"]["name"] for t in tools)

                sources, tool_calls_made = await _run_tool_loop(
                    question,
                    model,
                    tools,
                    profile,
                    session,
                    max_tool_calls,
                    max_turns,
                )
                answer = _answer(question, sources, model, profile)
                elapsed = time.time() - started

                metrics = evaluate_answer(answer, ground_truth, gold_urls, sources)
                eval_records.write_record(
                    answers_path,
                    {
                        "system": "baseline",
                        "model": model,
                        "query_id": query_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "answer": answer,
                        "gold_urls": sorted(
                            _normalize_source_url(url) for url in gold_urls
                        ),
                        "elapsed_seconds": round(elapsed, 2),
                        "sources": eval_records.source_records(sources, question),
                    },
                )
                trace.emit(
                    "query_end",
                    elapsed_seconds=round(elapsed, 2),
                    tool_calls=tool_calls_made,
                    cited_urls=[
                        _normalize_source_url(str(s.get("url") or ""))
                        for s in sources
                    ],
                    answer=answer,
                    **metrics,
                )
                print(
                    f"[{query_id}] {elapsed:.1f}s | tools: {tool_calls_made} | "
                    f"sources: {len(sources)} | match: {metrics['is_exact_match']}"
                )
                return {
                    "query_id": query_id,
                    "query": question,
                    "ground_truth": ground_truth,
                    "agent_answer": answer,
                    "tool_calls": tool_calls_made,
                    "elapsed_seconds": round(elapsed, 2),
                    "eval_metrics": metrics,
                }
    except Exception as exc:
        elapsed = time.time() - started
        print(f"[!] {query_id} failed: {exc}")
        trace.emit("query_end", error=str(exc), elapsed_seconds=round(elapsed, 2))
        return {
            "query_id": query_id,
            "query": question,
            "error": str(exc),
            "elapsed_seconds": round(elapsed, 2),
            "eval_metrics": {"score": 0.0, "is_exact_match": False},
        }


async def main_async() -> None:
    args = parse_args()
    model = args.model or os.environ.get("BROWSECOMP_MODEL") or EVAL_MODEL
    print("=== Single-call baseline (BrowseComp-Plus corpus) ===")
    print(f"Model: {model} | sample: {args.sample_size} | offset: {args.offset}")

    records = load_browsecomp_questions(args.sample_size, args.offset)
    temp_dir = Path("tmp/browsecomp_runs")
    temp_dir.mkdir(parents=True, exist_ok=True)
    server_script = Path(__file__).resolve().parent / "browsecomp_mcp_server.py"

    trace_path = Path(
        args.trace or Path(args.output).with_suffix("").as_posix() + ".trace.jsonl"
    )
    answers_path = eval_records.sidecar_path(trace_path)
    trace.start_run(trace_path)
    print(f"Run trace: {trace_path}\nAnswers:   {answers_path}")

    results = []
    try:
        for record in records:
            results.append(
                await run_single_query(
                    record,
                    model,
                    sys.executable,
                    server_script,
                    temp_dir,
                    answers_path,
                    args.max_tool_calls,
                    args.max_turns,
                )
            )
    finally:
        trace.close_run()

    answered = [r for r in results if "error" not in r]
    summary = {
        "system": "baseline_single_call",
        "model": model,
        "total_queries": len(results),
        "successful_queries": len(answered),
        "exact_match_rate": round(
            sum(1 for r in answered if r["eval_metrics"]["is_exact_match"])
            / len(results),
            3,
        )
        if results
        else 0.0,
        "average_duration_seconds": round(
            sum(r["elapsed_seconds"] for r in answered) / len(answered), 2
        )
        if answered
        else 0.0,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nExact match: {summary['exact_match_rate']:.0%}")
    print(f"Results: {output}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
