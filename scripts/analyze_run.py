"""Offline metrics from a run trace — no model call, no network, no log scraping.

A run's final score says whether it answered, not why. This reads the JSONL trace written
by ``llmflow_search.trace`` and separates the three things that score conflates:

* **Retrieval** — did search put a gold document in front of the agent at all, and at
  what rank?
* **Selection** — of the documents it was shown, which did it choose to open, and were
  those the ones that mattered?
* **Synthesis** — given what it actually read, did it produce the right answer?

A run can fail at any one of them, and the fix is different in each case.

    uv run --no-sync python scripts/analyze_run.py reports/browsecomp_results.trace.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

# Gold URLs, search results and read arguments all have to compare equal, and they reach
# the trace from three different places. The agent's own normalizer is the one authority
# on when two addresses are the same page.
from llmflow_search.sources import _normalize_source_url

# Ranks at which retrieval is scored. 5 is the size of one auto-read batch, 10 the size
# of one search result page: "the answer was on screen" and "the answer was one click in".
RECALL_AT = (1, 5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("trace", type=str, help="Path to the JSONL run trace.")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Also write the computed metrics to this JSON file.",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="Print one row per query in addition to the totals.",
    )
    # No price table ships with this repo. Token counts are measured; a dollar figure is
    # a local fact about one account's contract, and inventing one would put a number in
    # a report that nobody can reproduce.
    parser.add_argument(
        "--price-in",
        type=float,
        default=None,
        help="USD per million prompt tokens, for a cost-per-question figure.",
    )
    parser.add_argument(
        "--price-out",
        type=float,
        default=None,
        help="USD per million completion tokens.",
    )
    return parser.parse_args()


def _percentiles(values: list[float]) -> dict[str, float | None]:
    """p50/p90/p95/max of a sample, by nearest rank.

    The mean is what a benchmark reports and the tail is what a user feels: these runs
    have a long right tail (a question that re-searches four times costs minutes, not
    seconds), and averaging it away hides the only latency anyone complains about.
    """
    if not values:
        return {name: None for name in ("p50", "p90", "p95", "max", "mean")}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        rank = max(1, min(len(ordered), round(fraction * len(ordered) + 0.5)))
        return round(ordered[rank - 1], 2)

    return {
        "mean": round(fmean(ordered), 2),
        "p50": at(0.50),
        "p90": at(0.90),
        "p95": at(0.95),
        "max": round(ordered[-1], 2),
    }


def load_events(path: Path) -> list[dict]:
    events = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                events.append(record)
    return events


def group_by_query(events: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        grouped[str(event.get("query_id"))].append(event)
    return grouped


def _read_urls(events: list[dict]) -> list[str]:
    """Every page the run actually opened, in the order it opened them."""
    seen: list[str] = []
    for event in events:
        if event.get("event") != "tool_call" or event.get("error"):
            continue
        if event.get("tool") != "web_read":
            continue
        url = _normalize_source_url(str((event.get("args") or {}).get("url") or ""))
        if url and url not in seen:
            seen.append(url)
    return seen


def _best_ranks(events: list[dict]) -> dict[str, int]:
    """The best rank each URL ever held across every search this query ran."""
    best: dict[str, int] = {}
    for event in events:
        if event.get("event") != "search_results":
            continue
        for hit in event.get("results") or []:
            url = str(hit.get("url") or "")
            rank = hit.get("rank")
            if not url or rank is None:
                continue
            if url not in best or rank < best[url]:
                best[url] = int(rank)
    return best


def analyze_query(query_id: str, events: list[dict]) -> dict[str, Any]:
    start = next((e for e in events if e.get("event") == "query_start"), {})
    end = next((e for e in events if e.get("event") == "query_end"), {})
    gold = {str(url) for url in (start.get("gold_urls") or []) if url}

    ranks = _best_ranks(events)
    retrieved = set(ranks)
    gold_ranks = sorted(ranks[url] for url in gold & retrieved)
    best_gold_rank = gold_ranks[0] if gold_ranks else None

    read = _read_urls(events)
    read_set = set(read)
    admitted: set[str] = set()
    for event in events:
        if event.get("event") == "evidence_admitted":
            admitted |= {str(url) for url in (event.get("admitted") or []) if url}

    llm_calls = [e for e in events if e.get("event") == "llm_call"]
    roles = Counter(str(e.get("role")) for e in llm_calls)
    prompt_tokens = sum(int(e.get("prompt_tokens") or 0) for e in llm_calls)
    completion_tokens = sum(int(e.get("completion_tokens") or 0) for e in llm_calls)
    tokens_by_role: Counter = Counter()
    for event in llm_calls:
        tokens_by_role[str(event.get("role"))] += int(
            event.get("prompt_tokens") or 0
        ) + int(event.get("completion_tokens") or 0)
    # Calls that asked for a JSON schema. On a backend that does not enforce one the
    # request is silently ignored, so this is the only place the difference shows up.
    schema_calls = [e for e in llm_calls if e.get("schema_conforms") is not None]
    schema_conforming = sum(1 for e in schema_calls if e.get("schema_conforms"))
    round_trips = sum(int(e.get("round_trips") or 0) for e in llm_calls)
    duration_by_event: dict[str, float] = defaultdict(float)
    for event in events:
        if event.get("duration_ms") is not None:
            duration_by_event[str(event.get("event"))] += float(event["duration_ms"])

    tool_calls = [e for e in events if e.get("event") == "tool_call"]
    tools_used = Counter(str(e.get("tool")) for e in tool_calls)
    decisions = [e for e in events if e.get("event") == "post_batch_decision"]
    asked = [e for e in decisions if e.get("asked_model")]
    # The controller said one thing and the policy did another. A rule nobody ever
    # overrules is dead weight; one that fires constantly is a prompt problem.
    overruled = sum(
        1
        for e in asked
        if (e.get("model_decision") == "DONE") != (e.get("decision") == "FINISH")
    )
    rejected = sum(len(e.get("rejected") or []) for e in decisions)
    reads_by = Counter(
        str(e.get("by"))
        for e in events
        if e.get("event") == "read_decision"
        for _ in (e.get("selected") or [])
    )

    return {
        "query_id": query_id,
        "corpus_size": start.get("corpus_size"),
        # Retrieval
        "gold_count": len(gold),
        "retrieved_count": len(retrieved),
        "gold_retrieved": len(gold & retrieved),
        "best_gold_rank": best_gold_rank,
        "gold_in_top": {
            f"top{k}": bool(best_gold_rank is not None and best_gold_rank < k)
            for k in RECALL_AT
        },
        # Selection
        "read_count": len(read),
        "gold_read": len(gold & read_set),
        "gold_read_rate": round(len(gold & read_set) / len(gold), 3) if gold else None,
        # Of the pages opened, how many the ledger could actually use, and how many were gold.
        "read_precision_admitted": round(len(admitted & read_set) / len(read), 3)
        if read
        else None,
        "read_precision_gold": round(len(gold & read_set) / len(read), 3)
        if read
        else None,
        "reads_by": dict(reads_by),
        # Synthesis
        "is_exact_match": bool(end.get("is_exact_match")),
        "hit_gold_sources": bool(end.get("hit_gold_sources")),
        "score": end.get("score"),
        "error": end.get("error"),
        # Cost
        "llm_calls": len(llm_calls),
        "llm_calls_by_role": dict(roles),
        "tool_calls": len(tool_calls),
        "tool_calls_by_tool": dict(tools_used),
        "tool_call_errors": sum(1 for e in tool_calls if e.get("error")),
        "round_trips": round_trips,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "tokens_by_role": dict(tokens_by_role),
        "schema_calls": len(schema_calls),
        "schema_conforming": schema_conforming,
        "post_batch_decisions": len(decisions),
        "post_batch_asked": len(asked),
        "post_batch_overruled": overruled,
        "proposed_steps_rejected": rejected,
        "elapsed_seconds": end.get("elapsed_seconds"),
        "seconds_by_event": {
            name: round(ms / 1000, 2) for name, ms in sorted(duration_by_event.items())
        },
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def summarize(
    per_query: list[dict], price_in: float | None = None, price_out: float | None = None
) -> dict[str, Any]:
    answered = [q for q in per_query if not q.get("error")]
    seconds: dict[str, float] = defaultdict(float)
    roles: Counter = Counter()
    tokens_by_role: Counter = Counter()
    for query in per_query:
        for name, value in (query.get("seconds_by_event") or {}).items():
            seconds[name] += value
        roles.update(query.get("llm_calls_by_role") or {})
        tokens_by_role.update(query.get("tokens_by_role") or {})

    latencies = [
        float(q["elapsed_seconds"])
        for q in per_query
        if q.get("elapsed_seconds") is not None
    ]
    prompt_tokens = sum(q.get("prompt_tokens") or 0 for q in per_query)
    completion_tokens = sum(q.get("completion_tokens") or 0 for q in per_query)
    schema_calls = sum(q.get("schema_calls") or 0 for q in per_query)
    schema_conforming = sum(q.get("schema_conforming") or 0 for q in per_query)
    queries = len(per_query) or 1
    cost: dict[str, Any] = {}
    if price_in is not None and price_out is not None:
        usd = (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
        cost = {
            "usd_total": round(usd, 4),
            "usd_per_query": round(usd / queries, 4),
            "price_in_per_mtok": price_in,
            "price_out_per_mtok": price_out,
        }

    return {
        "queries": len(per_query),
        "answered": len(answered),
        "retrieval": {
            f"gold_in_{k}": round(
                sum(1 for q in per_query if q["gold_in_top"][k]) / len(per_query), 3
            )
            for k in (f"top{n}" for n in RECALL_AT)
        }
        if per_query
        else {},
        "selection": {
            "gold_read_rate": _mean(
                [q["gold_read_rate"] for q in per_query if q["gold_read_rate"] is not None]
            ),
            "reads_per_query": _mean([float(q["read_count"]) for q in per_query]),
            "read_precision_admitted": _mean(
                [
                    q["read_precision_admitted"]
                    for q in per_query
                    if q["read_precision_admitted"] is not None
                ]
            ),
            "read_precision_gold": _mean(
                [
                    q["read_precision_gold"]
                    for q in per_query
                    if q["read_precision_gold"] is not None
                ]
            ),
        },
        "synthesis": {
            "exact_match_rate": _mean(
                [1.0 if q["is_exact_match"] else 0.0 for q in per_query]
            ),
            "gold_citation_rate": _mean(
                [1.0 if q["hit_gold_sources"] else 0.0 for q in per_query]
            ),
            "average_score": _mean(
                [float(q["score"]) for q in per_query if q.get("score") is not None]
            ),
        },
        "latency_seconds": _percentiles(latencies),
        "tokens": {
            "prompt_per_query": round(prompt_tokens / queries, 1),
            "completion_per_query": round(completion_tokens / queries, 1),
            "total_per_query": round((prompt_tokens + completion_tokens) / queries, 1),
            "prompt_total": prompt_tokens,
            "completion_total": completion_tokens,
            "by_role_per_query": {
                role: round(total / queries, 1)
                for role, total in tokens_by_role.most_common()
            },
            **({"cost": cost} if cost else {}),
        },
        "schema": {
            "calls_requesting_a_schema": schema_calls,
            # Below 100% means the backend ignored the schema: the decoder was not
            # constraining anything and the output's shape was a matter of luck.
            "conforming": _rate(schema_conforming, schema_calls),
        },
        "cost": {
            "llm_calls_per_query": _mean([float(q["llm_calls"]) for q in per_query]),
            "tool_calls_per_query": _mean([float(q["tool_calls"]) for q in per_query]),
            "tool_call_errors": sum(q["tool_call_errors"] for q in per_query),
            "round_trips_per_query": _mean(
                [float(q.get("round_trips") or 0) for q in per_query]
            ),
            "llm_calls_by_role": dict(roles.most_common()),
            "post_batch_decisions": sum(q["post_batch_decisions"] for q in per_query),
            "post_batch_asked": sum(q["post_batch_asked"] for q in per_query),
            "post_batch_overruled": sum(q["post_batch_overruled"] for q in per_query),
            "proposed_steps_rejected": sum(
                q["proposed_steps_rejected"] for q in per_query
            ),
            "seconds_per_query": _mean(
                [
                    float(q["elapsed_seconds"])
                    for q in per_query
                    if q.get("elapsed_seconds") is not None
                ]
            ),
            "total_seconds_by_event": {
                name: round(value, 1) for name, value in sorted(seconds.items())
            },
        },
    }


def _format(summary: dict[str, Any], per_query: list[dict], show_rows: bool) -> str:
    latency = summary["latency_seconds"]
    tokens = summary["tokens"]
    lines = [
        f"Queries: {summary['answered']}/{summary['queries']} answered",
        "",
        "RETRIEVER (did search surface a gold document at all)",
    ]
    for key, value in summary["retrieval"].items():
        label = f"gold in {key.removeprefix('gold_in_')}"
        lines.append(
            f"  {label:<24} {value:>4.0%}" if value is not None else f"  {label:<24} n/a"
        )

    selection = summary["selection"]
    lines += [
        "",
        "PAGE SELECTION (of what it was shown, what it opened)",
        f"  reads per query          {selection['reads_per_query']}",
        f"  gold documents read      {selection['gold_read_rate']:.0%}"
        if selection["gold_read_rate"] is not None
        else "  gold documents read      n/a",
        f"  reads the ledger used    {selection['read_precision_admitted']:.0%}"
        if selection["read_precision_admitted"] is not None
        else "  reads the ledger used    n/a",
        f"  reads that were gold     {selection['read_precision_gold']:.0%}"
        if selection["read_precision_gold"] is not None
        else "  reads that were gold     n/a",
        "",
        "SYNTHESIS (given what it read, did it answer)",
        f"  exact match              {summary['synthesis']['exact_match_rate']:.0%}"
        if summary["synthesis"]["exact_match_rate"] is not None
        else "  exact match              n/a",
        f"  gold citation            {summary['synthesis']['gold_citation_rate']:.0%}"
        if summary["synthesis"]["gold_citation_rate"] is not None
        else "  gold citation            n/a",
        f"  average score            {summary['synthesis']['average_score']}",
        "",
        "LATENCY (seconds per question)",
        f"  mean {latency['mean']}   p50 {latency['p50']}   p90 {latency['p90']}"
        f"   p95 {latency['p95']}   max {latency['max']}",
        "",
        "COST",
        f"  tokens per query         {tokens['total_per_query']}"
        f"  ({tokens['prompt_per_query']} in / {tokens['completion_per_query']} out)",
        f"  model calls per query    {summary['cost']['llm_calls_per_query']}"
        f"  ({summary['cost']['round_trips_per_query']} HTTP round trips)",
        f"  tool calls per query     {summary['cost']['tool_calls_per_query']}"
        f"  ({summary['cost']['tool_call_errors']} errored)",
        f"  seconds per query        {summary['cost']['seconds_per_query']}",
        f"  post-batch rounds        {summary['cost']['post_batch_decisions']}"
        f" ({summary['cost']['post_batch_asked']} asked the model)",
        f"  controller overruled     {summary['cost']['post_batch_overruled']}",
        f"  proposed steps refused   {summary['cost']['proposed_steps_rejected']}",
    ]
    if tokens.get("cost"):
        lines.append(
            f"  USD per query            {tokens['cost']['usd_per_query']}"
            f"  (at {tokens['cost']['price_in_per_mtok']}/"
            f"{tokens['cost']['price_out_per_mtok']} per Mtok)"
        )
    schema = summary["schema"]
    if schema["calls_requesting_a_schema"]:
        conforming = schema["conforming"]
        lines.append(
            f"  schema honored           {conforming:.0%} of"
            f" {schema['calls_requesting_a_schema']} constrained calls"
            if conforming is not None
            else "  schema honored           n/a"
        )
    lines.append("  tokens by role (per query)")
    for role, value in list(tokens["by_role_per_query"].items())[:12]:
        lines.append(f"    {role:<20} {value}")
    for role, count in summary["cost"]["llm_calls_by_role"].items():
        lines.append(f"    {role:<20} {count} calls")
    lines.append("  seconds by event type")
    for name, value in summary["cost"]["total_seconds_by_event"].items():
        lines.append(f"    {name:<20} {value}")

    if show_rows:
        lines += ["", "PER QUERY", f"  {'query':<24} {'rank':>5} {'read':>5} {'gold':>5} {'llm':>4} {'sec':>7}  match"]
        for query in per_query:
            rank = query["best_gold_rank"]
            lines.append(
                f"  {query['query_id'][:24]:<24} "
                f"{('-' if rank is None else rank):>5} "
                f"{query['read_count']:>5} "
                f"{query['gold_read']:>5} "
                f"{query['llm_calls']:>4} "
                f"{(query['elapsed_seconds'] or 0):>7.1f}  "
                f"{'yes' if query['is_exact_match'] else 'no'}"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    events = load_events(Path(args.trace))
    grouped = group_by_query(events)
    per_query = [
        analyze_query(query_id, query_events)
        for query_id, query_events in grouped.items()
        if query_id != "None"
    ]
    per_query.sort(key=lambda q: q["query_id"])
    summary = summarize(per_query, args.price_in, args.price_out)

    print(_format(summary, per_query, args.per_query))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"summary": summary, "queries": per_query}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        print(f"\nMetrics written to {out}")


if __name__ == "__main__":
    main()
