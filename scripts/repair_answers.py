"""Re-run the citation repair over answers a finished run already produced.

A change to how answers are written can be measured two ways. Running the whole benchmark
again is one of them, and on ten questions it mostly measures noise: the searches, the
pages opened and the evidence admitted all differ between runs, so the answers being
compared were written from different evidence and the metric moves for reasons that have
nothing to do with the change.

The citation repair is a pass over finished prose. Applied offline to stored answers it
becomes a controlled experiment: the same questions, the same retrieved sources, the same
verified draft, one step different. The output is another ``.answers.jsonl``, scored by
the same judge as the original, and the difference between the two is the change.

    uv run --no-sync python scripts/repair_answers.py \\
        reports/eval/after_agent.answers.jsonl \\
        --out reports/eval/after_cite_offline.answers.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llmflow_search import attribution, eval_records  # noqa: E402
from llmflow_search.answering import _citation_coverage, _repair_citations  # noqa: E402
from llmflow_search.config import EVAL_MODEL  # noqa: E402
from llmflow_search.profiles import FOOTNOTE_PROFILE  # noqa: E402
from llmflow_search.sources import _format_sources_for_llm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("answers", type=str)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model", type=str, default=EVAL_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = eval_records.read_records(Path(args.answers))
    out = Path(args.out)
    if out.exists():
        out.unlink()

    before_total = after_total = 0.0
    counted = 0
    for record in records:
        answer = record.get("answer") or ""
        sources = record.get("sources") or []
        if attribution.is_refusal(answer) or not sources:
            eval_records.write_record(out, record)
            continue
        # Rendered from the stored source texts by the pipeline's own formatter, so the
        # repair sees the same numbered blob the verifier did.
        sources_text, _valid = _format_sources_for_llm(
            [
                {
                    "url": source.get("url"),
                    "title": source.get("title"),
                    "content": source.get("content"),
                }
                for source in sources
            ],
            record.get("question", ""),
        )
        before, _ = _citation_coverage(answer)
        repaired = _repair_citations(answer, sources_text, args.model, FOOTNOTE_PROFILE)
        after, _ = _citation_coverage(repaired)
        before_total, after_total, counted = (
            before_total + before,
            after_total + after,
            counted + 1,
        )
        print(f"  {record.get('query_id')}: {before:.0%} → {after:.0%} cited")
        eval_records.write_record(out, {**record, "answer": repaired})

    if counted:
        print(
            f"\nCitation coverage over {counted} answers: "
            f"{before_total / counted:.0%} → {after_total / counted:.0%}"
        )
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
