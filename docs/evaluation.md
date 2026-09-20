# How this agent is measured

Everything here is reproducible from the repository. The numbers below were produced by
the scripts named beside them, against a fixed corpus, with one pinned model.

## What is measured, and why these four

**Attribution — the share of an answer's claims that a cited source really states.**
This is the headline. Every other quality number the project had was about *citing*: did
the answer carry markers, did a marker point at a gold document. None of them read the
page back and asked whether the sentence the marker is attached to is in there. An answer
that cites five real URLs and states a sixth fact none of them contains scored perfectly
on all of them.

**Latency — seconds per question, with the tail.** The mean is what a benchmark reports
and p95 is what a user feels. These runs have a long right tail: a question that keeps
re-searching costs minutes, not seconds.

**Cost — tokens per question, broken down by which decision spent them.** Tokens rather
than dollars: a dollar figure is a fact about one account's contract, and no price table
that anyone could reproduce ships with this repo. `scripts/analyze_run.py --price-in
--price-out` converts to USD when you supply your own rates.

**A baseline with the same tools.** A graph with planning, an evidence ledger, a challenge
pass and a fail-closed verifier costs about twenty model calls per question. Whether that
buys anything is not answerable without running the cheap thing on the same questions.

## The setup

| | |
| --- | --- |
| Questions | BrowseComp-Plus test split, first 30 (`--offset 0 --sample-size 30`) |
| Corpus | The benchmark's own gold + negative documents per question, served over stdio MCP (`scripts/browsecomp_mcp_server.py`) — no live web, so runs are comparable |
| Model under test | `deepseek-v4.1-flash:cloud` via Ollama, thinking disabled, for both the agent and the baseline (`LLMFLOW_SEARCH_EVAL_MODEL`) |
| Judge | `gemma4:31b-cloud` at temperature 0 — a **different** model, required to return a verbatim supporting quote (`LLMFLOW_SEARCH_JUDGE_MODEL`) |

BrowseComp-Plus questions are deliberately hard: each is a conjunction of oblique
constraints ("someone born in 1886, mistaken for a shaman on a trip taken between April
and November…") whose answer is one entity. Absolute scores on it are low for everyone;
what it is good for is comparing two systems over the same evidence.

## How attribution is scored

`scripts/score_attribution.py`, over the `.answers.jsonl` sidecar that every run writes:

1. The answer is cut into claim units — sentences, list items, table rows — and their
   `[n]` markers are read. Deterministic, in `llmflow_search/attribution.py`, and unit
   tested. Headings, lead-ins and the trailing bibliography are not claims. A refusal
   ("the found sources do not provide enough information") asserts nothing and is counted
   as zero claims, with refusals reported separately.
2. For each (claim, cited source) pair the judge is shown the claim and the source text,
   and must return a verdict **and a span copied verbatim from that source**.
3. The span is then looked for in the source text offline. A pair counts as supported only
   if the judge said `supported` **and** its quote is really on the page.

Step 3 is the part that makes the number worth quoting. A judge asked only for a verdict
is a second model with the same failure mode as the first, and an agreeable hallucination
is indistinguishable from a correct call. Requiring a locatable span turns that failure
into a reported number instead of a silent inflation: `quote_absent_rate` in the output.
On the runs below it was 3–4%, with 96% of quotes matching the source
character-for-character.

The judge is a different model from the one under test, and is held fixed across every
run so that two scored runs are comparable. That the check is not decorative showed up
immediately when the judges were swapped: the first judge tried copied spans
character-for-character 98–100% of the time, the second stitches two real spans together
with an ellipsis and edits the seam. Those are both legitimate — the checker now locates
each half separately — but a "supported" verdict whose quote is nowhere on the page is
still discarded, and a "supported" with no quote at all is sent back once and then
discarded as malformed.

An uncited claim counts against the headline rate. A sentence with no marker is a sentence
for which no source was offered, and the metric is about the whole answer, not the part of
it that happens to be annotated. The report also prints `supported_of_cited` and
`citation_coverage`, whose product is the headline — which is how the second change below
was found.

## Reproducing

```bash
uv sync
uv run --no-sync python scripts/benchmark_browsecomp.py --sample-size 30 \
    --output reports/eval/agent.json --report reports/eval/agent.md
uv run --no-sync python scripts/baseline_single_call.py --sample-size 30 \
    --output reports/eval/baseline.json
uv run --no-sync python scripts/score_attribution.py reports/eval/agent.answers.jsonl \
    --json reports/eval/agent.attribution.json
uv run --no-sync python scripts/analyze_run.py reports/eval/agent.trace.jsonl
uv run --no-sync python scripts/compare_runs.py \
    baseline=reports/eval/baseline agent=reports/eval/agent
```

Each run leaves three artifacts behind: `.trace.jsonl` (one JSON object per event),
`.answers.jsonl` (the answer plus the full text of every citable source) and, once scored,
`.attribution.json` (one row per claim/source pair, with the judge's quote). The second is
what makes re-scoring possible without re-running: change the judge, the prompt or a
threshold, and the same stored answers can be scored again.

None of them are committed. `reports/` is gitignored, and these particular files must stay
that way: BrowseComp-Plus ships its questions and answers encrypted behind a canary string
that asks, in as many words, that the benchmark's contents never appear as plain text
online. The answers sidecar holds decrypted questions, ground truths and source documents.
Run the commands above to produce your own; do not publish them.

## Results

30 questions, `--offset 0`, `deepseek-v4.1-flash:cloud`, scored by the same judge. The
agent column is the current tree, with both changes from the next section already in it.

| Metric | single-call baseline | this agent |
| --- | ---: | ---: |
| **Answer quality** |  |  |
| claims supported by a cited source | 45% | **52%** |
| claims carrying any citation | 61% | **68%** |
| support rate among cited claims | 73% | **77%** |
| claims judged | 235 | 252 |
| questions refused for lack of evidence | 11 | 10 |
| exact match vs. ground truth | 43% | **57%** |
| cited at least one gold document | 57% | **63%** |
| **Evidence** |  |  |
| questions answered without crashing | 30 | 30 |
| questions that produced any source | 19 | 20 |
| gold document in the top 5 search results | 93% | 97% |
| pages read per question | 1.2 | 14.4 |
| gold documents actually read | 40% | **72%** |
| pages read that were gold | **88%** | 12% |
| **Latency** |  |  |
| seconds per question (p50) | 5.8 | 21.6 |
| seconds per question (p95) | 7.6 | 39.3 |
| **Cost** |  |  |
| tokens per question | 33,941 | 89,640 |
| model calls per question | 5.4 | 12.0 |

**The graph wins on every quality measure, at 2.6× the tokens and 3.7× the median
latency.** More right answers (57% against 43%), more answers whose claims a cited page
really states (52% against 45%), fewer refusals (10 against 11).

The breakdown says where the remaining cost goes. Retrieval is not where the two systems
differ — both put a gold document in the top five about nine questions in ten — but the
agent *opens* far more of them (72% against 40%) and reads 14.4 pages per question to do
it, at 12% precision. That is the next thing worth attacking.

## The five changes that moved the numbers

### 1. A duplicated MCP payload that silenced every structured consumer

`_mcp_result_text` collected the tool's text block *and* its `structuredContent`. For any
FastMCP tool declared `-> str` those are the same string, once raw and once wrapped as
`{"result": "..."}`; compared byte for byte they are never equal, so both were kept, and
two kept parts are merged into a JSON **array**. Every consumer downstream reads a tool
result as an object — the search memory, the source extractor, the trace's search-result
recorder all start by asking the payload for a key. Handed an array they found nothing
and said nothing.

The visible symptom was a run that searched twelve times, discovered zero URLs, opened
zero pages and reported that the evidence was insufficient. No error was raised anywhere.
It had been shipped in the previous commit and no metric in the project could see it.

10 questions, before and after:

| Metric | before | after |
| --- | ---: | ---: |
| questions that produced any source | 0 / 10 | 6 / 10 |
| pages read per question | 0.0 | 13.0 |
| gold documents actually read | 0% | 54% |
| exact match vs. ground truth | 0% | 30% |
| claims supported by a cited source | n/a — every answer was a refusal | 31% |
| seconds per question (p50) | 17.1 | 44.5 |
| tokens per question | 33,149 | 134,831 |

The latency and cost went up because the agent started doing the work it had been
skipping. Regression test: `tests/test_mcp_client.py::test_scalar_structured_wrapper_does_not_turn_a_payload_into_a_list`.

### 2. Half of every answer carried no citation at all

With evidence flowing again, the scorer showed the next ceiling: only 49% of claim
sentences carried an `[n]` marker. Those sentences cannot be attributed by anyone, so no
amount of better evidence could push the headline past about a half. The answer prompt had
always demanded a citation on every factual sentence — asking again was not a change.

Instead the verifier now re-reads its own output with the same segmentation the scorer
uses, names the sentences that failed, and spends one call asking for each to be either
cited or deleted (`_repair_citations`). The result is discarded unless it raises coverage
without gutting the answer, so "delete everything" cannot score well.

Measured offline over the *same* stored answers — same questions, same retrieved evidence,
same verified draft, one step different — because a second live run of 10 questions mostly
measures its own variance:

| Metric | before | after |
| --- | ---: | ---: |
| claims supported by a cited source | 31% | **42%** |
| claims carrying any citation | 49% | 61% |
| support rate among cited claims | 63% | 68% |
| claim sentences in the answers | 172 | 166 |
| claim sentences with no citation | 88 | 64 |
| `supported` verdicts with no such quote | 7% | 4% |

The support rate among cited claims went *up*, not down: the pass was not buying coverage
by attaching markers to whatever was nearby. It costs one model call, ~1.8s and ~6.8k
tokens per answered question — about 1.5% of the run. `LLMFLOW_SEARCH_CITATION_REPAIR=0`
restores the old behaviour.

### 3. The same duplicate again, on every page long enough to matter

The fix in change 1 compared the text block and the structured block *after* bounding
each. Bounding truncates a long page's body inside the payload, and truncates the whole
payload inside the `{"result": ...}` wrapper at a different point — so the two stopped
comparing equal exactly when a page exceeded the 12 kB bound, which is most real pages.
The array came back, and with it the silence.

What made it visible was a number, not a log line: pages read per question stood at 14.7
while candidate sources reaching the evidence ledger stood at 2 to 5. Replaying one
question's twelve reads through the source extractor gave two sources out of twelve. The
run was opening the right pages and throwing five in six of them away before anything
could judge them.

Deduplicating before bounding, over the same 30 questions:

| Metric | before | after |
| --- | ---: | ---: |
| exact match vs. ground truth | 43% | **53%** |
| cited at least one gold document | 50% | **60%** |
| claims supported by a cited source | 39% | 41% |
| questions that produced any source | 17 / 28 | 19 / 29 |
| questions refused for lack of evidence | 11 | 10 |
| seconds per question (p50) | 49.3 | 72.9 |
| tokens per question | 195,387 | 236,319 |

Slower and dearer, because the ledger and the challenge pass now actually receive the
pages and have to read them. Regression test:
`tests/test_mcp_client.py::test_a_long_payload_is_deduplicated_too`.

### 4. Demanding proof of the clues the asker supplied

BrowseComp question 776 asks for one publication, and `requirements_node` extracts eight
completion criteria from it — born in 1886, mistaken for a shaman on a 1915 trip, the
mistake caused by a misused word, 35 years in the same house, and, last, the title. Strict
mode admitted an answer only when every criterion had a row the fetched sources supported.
Replaying that question through the real ledger showed the run reading the gold document,
the ledger proposing the right title from it — and refusing, because the seven identifying
clues around it were never independently proven.

Those clues are search constraints, not reporting obligations. The asker already knows
them; they exist to single out one entity, and nobody wants them restated with citations.
Treating them as proof requirements makes strict mode unreachable by construction on
exactly the questions it was built for.

So a criterion now carries a kind. `requirements_node` marks the positions that only
identify the subject; strict mode requires the rest. The clues are still searched for and
still shown in the condition registry, and a clue that was never confirmed is handed to
the answer, which has to say so in a sentence rather than present the identification as
settled. `LLMFLOW_SEARCH_IDENTIFYING_CRITERIA=0` restores the old gate.

**The first attempt at this did nothing, and that is the useful part.** Relaxing the
ledger's readiness rule alone moved no number at all — 50% exact match against 53%, 38%
attribution against 41%, the same ten refusals. The gate had three doors and only one had
been opened: a `global_missing` entry indexed to a clue still blocked readiness, and the
challenge pass that runs next re-raised the same clue as a blocking gap. A relaxation
applied to one of three checks is not a relaxation.

With all three consistent, over 30 questions:

| Metric | before | after |
| --- | ---: | ---: |
| claims supported by a cited source | 42% | **49%** |
| claims carrying any citation | 60% | **68%** |
| support rate among cited claims | 70% | 71% |
| exact match vs. ground truth | 53% | 53% |
| questions refused for lack of evidence | 10 | 12 |
| claim sentences in the answers | 374 | 269 |
| seconds per question (p50) | 72.9 | **53.1** |
| tokens per question | 236,319 | **168,377** |

The answers got shorter, better attributed, faster and cheaper at the same accuracy: a run
that is not demanding proof of the asker's own clues stops spending evidence rounds
searching for it. Refusals went slightly *up* rather than down, which was not the
prediction — the gate is no longer blocked by clue gaps, but nothing about this change
helps a question whose answer was never found.

The guard that mattered held: the share of cited claims a source really supports did not
fall (70% → 71%). Had it dropped, the extra answers would have been bought by lowering the
bar rather than by aiming it at the right criteria, and this would have been reverted.

### 5. Asking a model what every tool result meant

The cost breakdown had one line far larger than the rest. Per question: 26 calls and
78,700 tokens — 47% of everything — on a single role, `observation`. Every search and every
page opened was handed straight back to a model to be diagnosed: is this useful, what kind
of source is it, what should be searched next. More tokens than the evidence ledger, the
challenge pass and the post-batch controller put together.

What it produced was a per-page summary, a source-quality guess, and a list of addresses
not to open again, which reached the post-batch controller as advice. The evidence ledger
never reads any of it — it judges a source by its text. And the objective parts of the
diagnosis (title, publication date, whether content came back) were already available
without a model, in the fallback path that existed for when the call failed.

So the experiment was a switch, not a rewrite. Expected: much cheaper, slightly worse.

| Metric | with diagnosis | without |
| --- | ---: | ---: |
| claims supported by a cited source | 49% | **52%** |
| support rate among cited claims | 71% | **77%** |
| exact match vs. ground truth | 53% | **57%** |
| cited at least one gold document | 60% | **63%** |
| questions refused for lack of evidence | 12 | **10** |
| model calls per question | 38.0 | **12.0** |
| tokens per question | 168,377 | **89,640** |
| seconds per question (p50) | 53.1 | **21.6** |

Nothing got worse. Removing two thirds of the model calls improved every quality measure
at once, halved the token bill and cut median latency by 59%. The diagnosis was not merely
expensive: its verdicts about which pages were worth reading were worse than no verdicts,
and the run did better deciding from the pages themselves.

It is now off by default (`LLMFLOW_SEARCH_DIAGNOSE_OBSERVATIONS=1` restores it). This is
the change that made the architecture worth its cost against the baseline, and it did so
by taking a piece of the architecture out.

## Two things that were tried and did not work

Both were plausible, both were measured, both are kept behind a flag rather than shipped.

**Opening fewer pages.** The agent opens 14.4 pages per question and 12% of them are the
ones that mattered; the gold document is the top search hit for 83% of questions, so
taking the top 2 instead of the top 5 ought to be nearly free. It is — for quality:

| Metric | top 5 (default) | top 2 |
| --- | ---: | ---: |
| claims supported by a cited source | 52% | 52% |
| exact match vs. ground truth | 57% | 57% |
| pages read per question | 14.4 | **5.5** |
| pages read that were gold | 12% | **22%** |
| tokens per question | 89,640 | **75,827** |
| seconds per question (p50) | **21.6** | 31.0 |

Two thirds of the reading disappears, quality does not move, and the run gets *slower*.
The reason is that reads inside a round run in parallel: opening five pages costs what
opening two costs, and taking fewer per round means more rounds, each with its own
search and its own controller call. The saving is real but it is a token saving, not a
time saving — 15% of the bill for 43% more waiting. `LLMFLOW_SEARCH_AUTO_READ_TOP_K=2`
makes that trade for anyone who prefers it.

**Letting the model choose which pages to open.** The post-batch controller is already
shown the candidate catalog with each URL's title and search snippet, and is already
allowed to propose reads — it just never gets asked, because the blind rule fires first
and its answer would be discarded. Unmuting it (`LLMFLOW_SEARCH_READ_SELECTION=model`)
looked like the change that should matter most, since a blind top-N can only work while
search ranking is good.

| Metric | blind top 5 | model chooses |
| --- | ---: | ---: |
| claims supported by a cited source | **52%** | 46% |
| exact match vs. ground truth | **57%** | 50% |
| cited at least one gold document | **63%** | 57% |
| gold documents actually read | **72%** | 51% |
| tokens per question | **89,640** | 113,429 |
| seconds per question (mean) | **22.5** | 43.1 |

Worse on every axis, and dearer. This is the second time in this project that replacing a
mechanical rule with a model's judgement about which pages are worth reading made things
worse — the first was the per-result diagnosis in change 5. A caveat with teeth, though:
on this corpus the gold document is the top hit 83% of the time, so "open the top of the
list" is close to an oracle and there is almost nothing for judgement to add. The result
says the model does not beat a near-perfect ranking; it does not say it would lose against
a live-web ranking, which is the case that matters and which this benchmark does not test.

## What the ledger is still refusing, and why

10 of 30 answers still end in a refusal, and after change 4 these are no longer runs that
hold the answer and will not say it. They are runs that did not find it: the remaining
refusals sit on questions where no fetched page supports the thing actually being asked
for. That is the behaviour the project promises, and improving it means better retrieval
and better page selection — 12% of the pages the agent opens are the ones that mattered —
not a further loosening of the gate.

## Scoring reproducibility

Re-scoring the same stored answers with the same judge at temperature 0 moves the
attribution rate by 1–2 points. Every figure in this document comes from one scoring pass
executed on 2026-09-20; differences under 3 points are inside the scorer's own noise and
should not be read as real.

## What these numbers do not show

- **n is 30.** On a 10-question slice the same two systems differed by 20 points of exact
  match; at 30 they are identical. Treat single-digit differences here as noise. Nothing
  in this document is a significance test.
- **One model, one corpus.** Everything was measured on `deepseek-v4.1-flash:cloud`
  against BrowseComp-Plus's own documents over a local MCP server. Live web retrieval —
  what the tool actually does in use — is slower, rate-limited, and has a far worse
  signal-to-noise ratio than a curated corpus of a few hundred documents.
- **The judge is one model, scored once.** It is a different model from the system under
  test (`gemma4:31b-cloud` against `deepseek-v4.1-flash:cloud`) and every run is scored by
  it, so runs are comparable. But it is still one model at temperature 0 with no second
  opinion and no human-labelled subset to calibrate against; the verbatim-quote check
  (3–4% of `supported` verdicts had no locatable quote) bounds fabrication, not
  misjudgement. Swapping the judge moved the absolute level by 1–5 points while leaving
  every comparison in this document pointing the same way.
- **Exact match is generous.** It is substring containment of the ground truth in the
  answer. An answer that says "I cannot answer this" and then lists the right entity among
  its notes scores as a match; at least one baseline "match" in the 10-question run was
  exactly that.
- **Claim segmentation is a heuristic.** Sentences, list items and table rows; headings,
  lead-ins and bibliographies excluded. It is deterministic and unit-tested, but a
  different cut of the same answer would give a different denominator.
- **The citation repair was measured offline.** That isolates it from run-to-run variance
  and is the right experiment for a post-processing pass; it is not a live end-to-end
  result.
