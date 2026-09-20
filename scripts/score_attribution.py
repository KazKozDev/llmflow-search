"""Score how much of an answer its own citations actually support.

The run-level metrics already in ``analyze_run.py`` answer "did it cite a gold page".
This answers the question a reader of the answer actually has: of the sentences in front
of me, how many are things the page attached to them really says?

Method, per stored answer (see ``llmflow_search.eval_records``):

1. Cut the answer into claim units and read their ``[n]`` markers (deterministic —
   ``llmflow_search.attribution``).
2. For each (claim, cited source) pair, show a judge the claim and the source text and
   require two things back: a verdict, and a **verbatim quote** from that source which
   carries the claim.
3. Check the quote against the source text offline. A pair counts as supported only when
   the judge says supported *and* its quote is really in the page.

Step 3 is the point. A judge asked only for a verdict is a second model with the same
failure mode as the first, and nothing in the output distinguishes a correct "supported"
from a confident one. Requiring a span the checker can find moves the most common failure
— agreeable hallucination — into something that shows up as a number
(``quote_absent_rate``) instead of silently inflating the headline.

An uncited claim counts against the headline rate. A sentence with no marker is a
sentence no source was offered for, and the metric is about the answer, not about the
subset of it that happens to be annotated.

    uv run --no-sync python scripts/score_attribution.py reports/agent.answers.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llmflow_search import attribution, eval_records, schema_guard  # noqa: E402
from llmflow_search.config import JUDGE_MODEL  # noqa: E402
from llmflow_search.llm import _json_loads_best_effort, _ollama_chat  # noqa: E402
from llmflow_search.passages import PassageStore, split_into_passages  # noqa: E402

# A source short enough to show whole is shown whole: passage selection can only lose the
# sentence that settles the claim, and a false "unsupported" is the expensive error here.
FULL_SOURCE_MAX_CHARS = 8000

# When a page is longer than that, the judge sees its opening plus the windows that bear
# on the claim. The opening is always included because datelines, bylines and "as of"
# qualifiers live there and half the claims in a research answer are dated.
SOURCE_HEAD_CHARS = 1200
SOURCE_PASSAGE_BUDGET = 6000

# Sources consulted per claim. Answers cite two or three; a sentence that lists eight is
# citing a section, not a fact, and the first few settle it either way.
MAX_SOURCES_PER_CLAIM = 3

JUDGE_SYSTEM = """You check whether a source supports a claim. You are strict.

Read the SOURCE text and the CLAIM. Decide what the SOURCE alone establishes:
- "supported": the SOURCE states the claim, or states something that entails it outright.
- "partial": the SOURCE states part of the claim, or states it with a qualification the claim drops.
- "unsupported": the SOURCE does not state it, contradicts it, or merely discusses the topic.

Plausibility is not support. Your own knowledge is not support. If the SOURCE does not
contain it, the verdict is "unsupported" however true the claim may be.

Return JSON only, with exactly these keys:
- "verdict": "supported" | "partial" | "unsupported"
- "quote": a span copied character-for-character from the SOURCE that carries the claim,
  or "" when the verdict is "unsupported". Never paraphrase, never repair, never
  translate. Copy it exactly as it appears, at most 300 characters."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["supported", "partial", "unsupported"],
        },
        "quote": {"type": "string"},
    },
    "required": ["verdict", "quote"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("answers", type=str, help="Path to a .answers.jsonl sidecar.")
    parser.add_argument(
        "--judge-model",
        type=str,
        default=JUDGE_MODEL,
        help=f"Model that judges support (default: {JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--json", type=str, default=None, help="Write full per-claim results here."
    )
    parser.add_argument(
        "--workers", type=int, default=6, help="Concurrent judge calls (default: 6)."
    )
    parser.add_argument(
        "--per-query", action="store_true", help="Print one row per question."
    )
    return parser.parse_args()


def source_view(claim_text: str, source_text: str) -> str:
    """The slice of a source the judge is shown for this claim."""
    if len(source_text) <= FULL_SOURCE_MAX_CHARS:
        return source_text
    head = source_text[:SOURCE_HEAD_CHARS]
    store = PassageStore()
    for passage in split_into_passages(source_text, url="", title=""):
        store.add(passage)
    selected, used = [], 0
    for _score, passage in store.ranked(claim_text):
        if used + len(passage.text) > SOURCE_PASSAGE_BUDGET:
            break
        selected.append(passage.text)
        used += len(passage.text)
    return head + "\n[...]\n" + "\n[...]\n".join(selected)


def _contract_violations(verdict) -> list[str]:
    """Every way a judge reply fails the contract, including the one a schema cannot state.

    "supported" with an empty quote is well-formed JSON and useless: the whole point of
    asking for a span is that the verdict can be checked, and a verdict with nothing to
    check is an opinion. It goes back for one repair round like any other malformed reply.
    """
    if not isinstance(verdict, dict):
        return ["not a JSON object"]
    violations = schema_guard.schema_violations(verdict, JUDGE_SCHEMA)
    if verdict.get("verdict") == "supported" and not str(verdict.get("quote") or "").strip():
        violations.append("$.quote: empty for a 'supported' verdict")
    return violations


def judge(claim_text: str, source_text: str, model: str) -> dict[str, Any]:
    """One support verdict, repaired once if the model ignores the output contract."""
    prompt = f"""SOURCE:
{source_view(claim_text, source_text)}

CLAIM:
{claim_text}

Return the JSON verdict."""
    messages = [{"role": "user", "content": prompt}]
    content = _ollama_chat(
        model, messages, tools=None, system=JUDGE_SYSTEM, temperature=0, json_mode=True
    ).get("content", "")
    verdict = _json_loads_best_effort(content, None)
    violations = _contract_violations(verdict)
    if violations:
        repair = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": "That did not match the contract: "
                + "; ".join(violations)
                + "\nReturn only JSON with keys verdict and quote.",
            },
        ]
        content = _ollama_chat(
            model,
            repair,
            tools=None,
            system=JUDGE_SYSTEM,
            temperature=0,
            json_mode=True,
        ).get("content", "")
        verdict = _json_loads_best_effort(content, None)
        violations = _contract_violations(verdict)
    if violations or not isinstance(verdict, dict):
        # A judge that will not answer in the contract cannot be read as agreeing.
        return {"verdict": "unsupported", "quote": "", "judge_malformed": True}
    return {
        "verdict": verdict["verdict"],
        "quote": str(verdict["quote"])[:300],
        "judge_malformed": False,
    }


def score_record(record: dict, model: str, pool: ThreadPoolExecutor) -> dict[str, Any]:
    sources = {
        int(source["id"]): source for source in record.get("sources") or []
    }
    claims = attribution.split_claims(record.get("answer") or "")

    pairs: list[tuple[attribution.Claim, dict]] = []
    for claim in claims:
        for source_id in claim.citations[:MAX_SOURCES_PER_CLAIM]:
            source = sources.get(source_id)
            if source is not None:
                pairs.append((claim, source))

    verdicts = list(
        pool.map(
            lambda pair: judge(pair[0].bare_text, pair[1]["content"], model), pairs
        )
    )

    checked: list[dict] = []
    for (claim, source), verdict in zip(pairs, verdicts, strict=True):
        quote_check = attribution.find_quote(verdict["quote"], source["content"])
        in_shown = bool(
            quote_check.found
            and quote_check.window
            and attribution.normalize(quote_check.window)
            in attribution.normalize(source.get("shown") or "")
        )
        checked.append(
            {
                "claim_index": claim.index,
                "claim": claim.bare_text,
                "source_id": source["id"],
                "source_url": source.get("url", ""),
                "verdict": verdict["verdict"],
                "judge_malformed": verdict["judge_malformed"],
                "quote": verdict["quote"],
                "quote_status": quote_check.verdict,
                "quote_ratio": quote_check.ratio,
                "quote_inside_shown_excerpt": in_shown,
                "lexical_overlap": attribution.lexical_overlap(
                    claim.bare_text, source["content"]
                ),
                # The pair counts only when the judge agreed *and* the span it offered is
                # really on the page. Either half alone is an opinion.
                "supported": verdict["verdict"] == "supported" and quote_check.found,
            }
        )

    by_claim: dict[int, list[dict]] = {}
    for row in checked:
        by_claim.setdefault(row["claim_index"], []).append(row)

    cited = [claim for claim in claims if claim.is_cited]
    supported_claims = [
        claim
        for claim in claims
        if any(row["supported"] for row in by_claim.get(claim.index, []))
    ]
    dangling = [
        source_id
        for claim in claims
        for source_id in claim.citations
        if source_id not in sources
    ]

    return {
        "query_id": record.get("query_id"),
        "system": record.get("system"),
        "refused": attribution.is_refusal(record.get("answer") or ""),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "claims_total": len(claims),
        "claims_cited": len(cited),
        "claims_supported": len(supported_claims),
        "dangling_citations": len(dangling),
        "source_count": len(sources),
        "claim_rows": checked,
        "claims": [
            {
                "index": claim.index,
                "text": claim.bare_text,
                "citations": claim.citations,
                "supported": any(
                    row["supported"] for row in by_claim.get(claim.index, [])
                ),
            }
            for claim in claims
        ],
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def summarize(per_query: list[dict]) -> dict[str, Any]:
    rows = [row for query in per_query for row in query["claim_rows"]]
    claims_total = sum(q["claims_total"] for q in per_query)
    claims_cited = sum(q["claims_cited"] for q in per_query)
    claims_supported = sum(q["claims_supported"] for q in per_query)
    agreed = [row for row in rows if row["verdict"] == "supported"]
    quote_absent = [row for row in agreed if row["quote_status"] == "absent"]

    refused = [q for q in per_query if q["refused"]]
    answered = [q for q in per_query if q["claims_total"] > 0]
    per_answer_rates = [
        q["claims_supported"] / q["claims_total"] for q in answered
    ]

    return {
        "questions": len(per_query),
        "questions_with_claims": len(answered),
        # A refusal asserts nothing and is scored as nothing. Reported on its own, because
        # a system that refuses everything would otherwise show a perfect attribution rate.
        "questions_refused": len(refused),
        "claims_total": claims_total,
        # The headline: of everything the answer asserts, how much a cited page really says.
        "attribution_rate": _rate(claims_supported, claims_total),
        # The same question asked only of the sentences that carry a marker. The gap
        # between the two is how much of the answer went out unattributed.
        "supported_of_cited": _rate(claims_supported, claims_cited),
        "citation_coverage": _rate(claims_cited, claims_total),
        "uncited_claims": claims_total - claims_cited,
        "dangling_citations": sum(q["dangling_citations"] for q in per_query),
        # Averaged per answer rather than per claim, so one long answer cannot decide the run.
        "mean_attribution_per_answer": round(
            statistics.fmean(per_answer_rates), 3
        )
        if per_answer_rates
        else None,
        "judge": {
            "pairs": len(rows),
            "verdicts": {
                name: sum(1 for row in rows if row["verdict"] == name)
                for name in ("supported", "partial", "unsupported")
            },
            # A "supported" whose quote is nowhere on the page: the judge's own error rate.
            "quote_absent_rate": _rate(len(quote_absent), len(agreed)),
            "quote_exact_rate": _rate(
                sum(1 for row in agreed if row["quote_status"] == "exact"), len(agreed)
            ),
            "malformed_rate": _rate(
                sum(1 for row in rows if row["judge_malformed"]), len(rows)
            ),
            # Support found only outside the excerpt the answer model was shown: the
            # claim is true of the page, but the run could not have known it.
            "supported_outside_shown_excerpt": sum(
                1
                for row in rows
                if row["supported"] and not row["quote_inside_shown_excerpt"]
            ),
            "mean_lexical_overlap_supported": round(
                statistics.fmean(
                    [row["lexical_overlap"] for row in rows if row["supported"]]
                ),
                3,
            )
            if any(row["supported"] for row in rows)
            else None,
            "mean_lexical_overlap_unsupported": round(
                statistics.fmean(
                    [row["lexical_overlap"] for row in rows if not row["supported"]]
                ),
                3,
            )
            if any(not row["supported"] for row in rows)
            else None,
        },
    }


def _format(summary: dict[str, Any], per_query: list[dict], show_rows: bool) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0%}"

    judge = summary["judge"]
    lines = [
        f"Answers scored: {summary['questions_with_claims']}/{summary['questions']}"
        f"  ({summary['claims_total']} claims,"
        f" {summary['questions_refused']} refused for lack of evidence)",
        "",
        "ATTRIBUTION",
        f"  claims supported by a cited source   {pct(summary['attribution_rate'])}",
        f"  of the claims that carry a citation  {pct(summary['supported_of_cited'])}",
        f"  claims carrying any citation         {pct(summary['citation_coverage'])}",
        f"  uncited claims                       {summary['uncited_claims']}",
        f"  citations to a source that does not exist  {summary['dangling_citations']}",
        f"  per-answer mean                      {pct(summary['mean_attribution_per_answer'])}",
        "",
        "JUDGE (how much to trust the number above)",
        f"  claim/source pairs judged            {judge['pairs']}",
        f"  verdicts                             {judge['verdicts']}",
        f"  'supported' with no such quote       {pct(judge['quote_absent_rate'])}",
        f"  quotes copied verbatim               {pct(judge['quote_exact_rate'])}",
        f"  judge returned malformed output      {pct(judge['malformed_rate'])}",
        f"  supported only outside shown excerpt {judge['supported_outside_shown_excerpt']}",
        f"  lexical overlap  supported/not       "
        f"{judge['mean_lexical_overlap_supported']} / {judge['mean_lexical_overlap_unsupported']}",
    ]
    if show_rows:
        lines += ["", "PER QUESTION", f"  {'query':<12} {'claims':>6} {'cited':>6} {'ok':>4}  rate"]
        for query in per_query:
            total = query["claims_total"]
            rate = (
                f"{query['claims_supported'] / total:.0%}"
                if total
                else ("refused" if query["refused"] else "—")
            )
            lines.append(
                f"  {str(query['query_id'])[:12]:<12} {total:>6} "
                f"{query['claims_cited']:>6} {query['claims_supported']:>4}  {rate}"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = eval_records.read_records(Path(args.answers))
    print(f"Scoring {len(records)} answers with judge {args.judge_model}...")
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        per_query = [score_record(record, args.judge_model, pool) for record in records]
    summary = summarize(per_query)
    summary["judge"]["model"] = args.judge_model
    summary["judge"]["scoring_seconds"] = round(time.time() - started, 1)

    print()
    print(_format(summary, per_query, args.per_query))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"summary": summary, "queries": per_query},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nPer-claim detail written to {out}")


if __name__ == "__main__":
    main()
