"""Put two or more measured runs side by side, as a markdown table.

A number on its own says nothing. "62% of claims are supported" is only interpretable
next to what the same scorer said about the same questions before a change, or about the
baseline the architecture is supposed to beat. This reads the artifacts each run already
leaves behind and prints the comparison a report can paste.

Each run is named by the prefix its files share:

    <prefix>.trace.jsonl        the run trace          (analyze_run.py)
    <prefix>.answers.jsonl      the stored answers     (score_attribution.py)
    <prefix>.attribution.json   scored attribution, if it has been scored

    uv run --no-sync python scripts/compare_runs.py \\
        before=reports/eval/before_agent after=reports/eval/after_agent
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_run  # noqa: E402

from llmflow_search import eval_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "runs",
        nargs="+",
        help="label=prefix pairs, in the order the columns should appear.",
    )
    parser.add_argument("--out", type=str, default=None, help="Write markdown here.")
    return parser.parse_args()


def load_run(prefix: str) -> dict[str, Any]:
    root = Path(prefix)
    trace_path = Path(str(root) + ".trace.jsonl")
    summary: dict[str, Any] = {}
    if trace_path.exists():
        events = analyze_run.load_events(trace_path)
        per_query = [
            analyze_run.analyze_query(query_id, query_events)
            for query_id, query_events in analyze_run.group_by_query(events).items()
            if query_id != "None"
        ]
        summary["trace"] = analyze_run.summarize(per_query)
    attribution_path = Path(str(root) + ".attribution.json")
    if attribution_path.exists():
        summary["attribution"] = json.loads(
            attribution_path.read_text(encoding="utf-8")
        )["summary"]
    answers_path = Path(str(root) + ".answers.jsonl")
    if answers_path.exists():
        records = eval_records.read_records(answers_path)
        summary["answers"] = {
            "answered_with_sources": sum(1 for r in records if r.get("sources")),
            "records": len(records),
        }
    return summary


def _get(run: dict, path: str) -> Any:
    node: Any = run
    for key in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value):.0%}"


def _num(value: Any) -> str:
    return "—" if value is None else f"{float(value):g}"


def _round1(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1f}"


def _thousands(value: Any) -> str:
    return "—" if value is None else f"{round(float(value)):,}"


# (section, label, path into the loaded run, formatter)
ROWS: list[tuple[str, str, str, Callable[[Any], str]]] = [
    ("Answer quality", "claims supported by a cited source", "attribution.attribution_rate", _pct),
    ("Answer quality", "claims carrying any citation", "attribution.citation_coverage", _pct),
    ("Answer quality", "claims judged", "attribution.claims_total", _num),
    ("Answer quality", "questions refused for lack of evidence", "attribution.questions_refused", _num),
    ("Answer quality", "support rate among cited claims", "attribution.supported_of_cited", _pct),
    ("Answer quality", "exact match vs. ground truth", "trace.synthesis.exact_match_rate", _pct),
    ("Answer quality", "cited at least one gold document", "trace.synthesis.gold_citation_rate", _pct),
    ("Evidence", "questions answered without crashing", "trace.answered", _num),
    ("Evidence", "questions that produced any source", "answers.answered_with_sources", _num),
    ("Evidence", "gold document in the top 5 results", "trace.retrieval.gold_in_top5", _pct),
    ("Evidence", "pages read per question", "trace.selection.reads_per_query", _round1),
    ("Evidence", "gold documents actually read", "trace.selection.gold_read_rate", _pct),
    ("Evidence", "pages read that were gold", "trace.selection.read_precision_gold", _pct),
    ("Latency", "seconds per question (p50)", "trace.latency_seconds.p50", _round1),
    ("Latency", "seconds per question (p95)", "trace.latency_seconds.p95", _round1),
    ("Latency", "seconds per question (mean)", "trace.latency_seconds.mean", _round1),
    ("Cost", "tokens per question", "trace.tokens.total_per_query", _thousands),
    ("Cost", "prompt tokens per question", "trace.tokens.prompt_per_query", _thousands),
    ("Cost", "model calls per question", "trace.cost.llm_calls_per_query", _round1),
    ("Cost", "tool calls per question", "trace.cost.tool_calls_per_query", _round1),
]


def render(runs: list[tuple[str, dict]]) -> str:
    labels = [label for label, _ in runs]
    lines = [
        "| Metric | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---:" for _ in labels) + " |",
    ]
    section = ""
    for group, label, path, formatter in ROWS:
        values = [formatter(_get(run, path)) for _, run in runs]
        if all(value == "—" for value in values):
            continue
        if group != section:
            section = group
            lines.append(f"| **{group}** | " + " | ".join("" for _ in labels) + " |")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"expected label=prefix, got {spec!r}")
        label, prefix = spec.split("=", 1)
        runs.append((label, load_run(prefix)))

    table = render(runs)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")
        print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
