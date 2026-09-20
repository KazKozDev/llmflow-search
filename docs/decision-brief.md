# Does the agent earn its cost? — decision brief

**Period covered:** one working session, 2026-09-20. All runs and all scoring were
executed on that date.

**Question answered:** whether LLMFlow-Search's graph — planning, an evidence ledger, a
challenge pass, a fail-closed verifier — produces better answers than the same model
given the same tools and no graph, and what to change next.

**In scope:** measured behaviour on 30 BrowseComp-Plus questions over the benchmark's own
document corpus, served locally over MCP; one model under test
(`deepseek-v4.1-flash:cloud`); one judge model (`gemma4:31b-cloud`).

**Deliberately out of scope:** live-web retrieval, any second model, any second question
set, statistical significance testing, and cost in currency. Method and full tables are in
[evaluation.md](evaluation.md); this document is the decision layer over it.

---

## The answer in one paragraph

The graph now beats the no-graph baseline on every quality measure, at 2.6× the tokens and
3.7× the median latency. It did not at the start of the session: it lost on answer quality
while costing 6× more. Four defects closed the gap, and the single largest improvement
came from **deleting** a component rather than adding one. Two further ideas were measured
and rejected. The remaining inefficiency is page selection — 14 pages opened per question,
12% of them useful — and the two obvious fixes for it have now both been tried and do not
work on this corpus.

---

## What changed

One consistent scoring pass, all runs re-scored on 2026-09-20 with the current scorer and
the same judge. Each row differs from the one above it by one change.

| | attribution | exact match | cited gold doc | tokens/q | p50 sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-call baseline | 45% | 43% | 57% | 33,941 | 5.8 |
| agent, session start | 39% | 43% | 50% | 195,387 | 49.3 |
| + long-payload fix | 42% | 53% | 60% | 236,319 | 72.9 |
| + criterion kinds | 49% | 53% | 60% | 168,377 | 53.1 |
| + diagnosis removed | **52%** | **57%** | **63%** | **89,640** | **21.6** |

*Attribution* = share of the answer's claim sentences that a cited source demonstrably
states, verified by locating the judge's quote in the page. Refusals count as zero claims
and are reported separately (10 of 30 at the end, 11 for the baseline).

### The four defects, by size of effect

**1. A tool result that arrived unreadable.** Every FastMCP tool declared `-> str` answers
twice — once as text, once as `structuredContent` wrapping the same string. Both copies
were kept and merged into a JSON array. Every consumer reads a tool result as an object,
found nothing, and reported nothing. Runs searched twelve times, discovered zero URLs and
declared the evidence insufficient, with no error anywhere. Sources per question went from
0 to 6 in ten.

The first fix compared the two copies after each had been length-bounded; bounding
truncates them at different points, so it held only for pages under 12 kB. The real fix
was to deduplicate before bounding. Pages reaching the evidence ledger went from 2 of 12
to 12 of 12; exact match 43% → 53%.

**2. Half of every answer carried no citation.** 49% of claim sentences had no `[n]`,
capping attribution at about a half regardless of evidence quality. The answer prompt had
always demanded citations, so asking again was not a change. The verifier now re-reads its
own output with the same segmentation the scorer uses, names the failing sentences, and
spends one call requiring each to be cited or deleted. Measured offline on identical
stored answers: attribution 31% → 42%, and the support rate among cited claims rose rather
than fell, so coverage was not bought with careless markers.

**3. Proof demanded of the asker's own clues.** A puzzle question — born in 1886, mistaken
for a shaman on a 1915 trip, 35 years in one house, name the publication — decomposes into
eight criteria, and strict mode required every one to be independently sourced. Seven of
them are search constraints the asker already knows. Runs read the gold page, the ledger
proposed the right answer from it, and refused. Criteria now carry a kind; unconfirmed
clues are disclosed in the answer instead of blocking it. Attribution 42% → 49%.

The first attempt at this moved nothing at all: the gate had three checks — the readiness
rule, the gaps list, the challenge pass — and only one had been changed.

**4. A model call on every tool result.** 26 calls and 47% of all tokens per question went
to diagnosing each search and each page: is this useful, what kind of source, what next.
Its output was a page summary and a do-not-read list; the evidence ledger never reads any
of it. Removing it was expected to be much cheaper and slightly worse.

| | with diagnosis | without |
| --- | ---: | ---: |
| attribution | 49% | **52%** |
| exact match | 53% | **57%** |
| refusals | 12 | **10** |
| model calls/q | 38.0 | **12.0** |
| tokens/q | 168,377 | **89,640** |
| p50 seconds | 53.1 | **21.6** |

Nothing got worse. Two thirds of the model calls removed, every quality measure up.

---

## Why it matters

**The architecture's value was never measured, and for most of its life it was negative.**
Three of the four defects had shipped and were invisible: no test failed, no log line
appeared, and the project's existing metrics — did it cite something, did a citation point
at a gold document — could not see them. The one metric that surfaced all four was the one
that re-reads the cited page and asks whether the sentence is in it.

**Twice, a model's judgement lost to a mechanical rule.** Removing the per-result
diagnosis improved every measure; letting the model choose which pages to open (below)
made every measure worse. This is a finding about where to spend calls in this pipeline,
not a general claim about models.

**The remaining cost is page selection, and it is not obviously fixable.** The agent opens
14.4 pages per question; 12% are the ones that mattered.

---

## What follows

### Two options tried and rejected

| | now (top 5) | open top 2 | model chooses |
| --- | ---: | ---: | ---: |
| attribution | 52% | 52% | 46% |
| exact match | 57% | 57% | 50% |
| gold docs read | 72% | 54% | 51% |
| pages read/q | 14.4 | **5.5** | 5.9 |
| tokens/q | 89,640 | **75,827** | 113,429 |
| p50 seconds | **21.6** | 31.0 | 33.7 |

**Opening the top 2 instead of the top 5** removes two thirds of the reading at identical
quality — and makes the run slower. Reads inside a round are parallel, so five pages cost
what two cost; taking fewer per round means more rounds, each with its own search and
controller call. It is a 15% token saving for 43% more waiting, available as
`LLMFLOW_SEARCH_AUTO_READ_TOP_K=2` for anyone who prefers that trade. Default unchanged.

**Letting the controller choose pages from the catalog** — it is already shown each URL's
title and snippet, and is simply never asked — was worse on every axis and dearer.
Available as `LLMFLOW_SEARCH_READ_SELECTION=model`. Default unchanged.

Both results carry the same caveat: on this corpus the gold document is the top search hit
for 83% of questions, so "open the top of the list" is close to an oracle and judgement
has almost nothing to add. The measurement says the model does not beat a near-perfect
ranking. It does not say it would lose against a live-web ranking, which is the case that
matters and the one this benchmark does not test.

### What to do next, in order

1. **Nothing, to the read-selection logic.** Both candidate fixes are measured and neither
   wins. Further tuning against this corpus risks fitting the benchmark.
2. **Build a dirty-retrieval test case** — a question set where the right page is not the
   top hit. Every claim about page selection is currently untestable, and the two rejected
   options above may well win there. This is the highest-value next piece of work and it
   is measurement work, not agent work.
3. **Look at the 10 refusals.** After change 3 these are questions where nothing retrieved
   supports what was asked, not questions where the answer was held back. That is intended
   behaviour, but it is a third of the set.
4. **Re-measure on a second model before generalising anything here.** Every number in
   this document is from one model.

---

## What would change this conclusion

- **n is 30, each configuration run once.** On a 10-question slice, two systems that are
  identical at n=30 differed by 20 points of exact match. Differences under roughly 5
  points here should not be read as real.
- **The judge is not bit-reproducible.** Re-scoring the same stored answers with the same
  judge at temperature 0 moved results by 1–2 points. Differences under 3 points are
  inside the scorer's own noise.
- **One corpus, one model, no live web.** BrowseComp-Plus's documents are a curated set of
  a few hundred per question with unusually good BM25 ranking. The tool's actual use —
  live search, rate limits, adversarial SEO — is not represented.
- **The judge is a different model from the one under test**, and a verbatim-quote check
  bounds its fabrication (3–4% of `supported` verdicts had no locatable quote). Neither
  guards against a shared blind spot, and there is no human-labelled subset.
- **Exact match is generous**: substring containment of the ground truth in the answer. At
  least one baseline "match" was an answer that declined to answer and mentioned the right
  entity in passing.
