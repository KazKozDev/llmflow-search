"""The one place that decides what a round does next.

The post-batch controller used to emit a raw intent which was then repaired by a chain of
guards downstream of it: CONTINUE with an empty plan became DONE; DONE with no fetched
source became NEXT; invented addresses were discarded; repeated steps were discarded;
a page read was forced on top; a listing drill-down was forced on top of that. Each patch
was added after a specific failure, each was correct about that failure, and together they
meant the controller's answer was advisory at best — the run's real behaviour was the
composition of six rewrites, in an order nobody had chosen deliberately, spread across a
hundred lines of one function.

The rules did not need to move, they needed to become the decision. Here they are the
decision. Two entry points, and everything else is data:

* ``forced_action`` — the state leaves no choice. Nothing is asked, because whatever the
  answer was it would be discarded (which is what the old guards did to it, after paying
  for it).
* ``resolve_choice`` — the model's answer is turned into the round's action, with the
  admissibility rules applied as filters on the proposal rather than as rewrites of a
  decision already taken.

Nothing downstream of these functions changes what they returned.
"""

from dataclasses import dataclass, field

from .config import SKIP_CONTROLLER_WHEN_READS_FORCED
from .tool_steps import _step_identity

# Tools whose argument is, by definition, a page someone else pointed us at: there is no
# way to know such an address without having seen it. A step naming one the run never saw
# is an invention regardless of how plausible the address looks.
DISCOVERY_REQUIRED_PREFIXES = (
    "web_read: ",
    "web_extract_tables: ",
    "web_detect_downloads: ",
    "classify_source: ",
    "web_crawl: ",
)
# Endpoints a planner may legitimately construct from knowledge of a public interface —
# the plan prompt tells it to hit a known API directly rather than search for it — so
# these are only checked once discovery has actually produced candidates.
CONSTRUCTIBLE_URL_PREFIXES = ("web_parse_file: ", "web_fetch_json: ")

URL_STEP_PREFIXES = DISCOVERY_REQUIRED_PREFIXES + CONSTRUCTIBLE_URL_PREFIXES
# Archive lookups are deliberately absent from both: their whole purpose is reaching a URL
# that no longer resolves, so checking them against what a live search returned would block
# the one tool that can still recover the source.

# Discovery tools return candidate URLs but not source-grade evidence. A round that
# searched and queued no read spends its budget on snippets alone — that is how a run ends
# with three pages fetched out of a hundred discovered.
AUTO_READ_DISCOVERY_TOOLS = frozenset(
    {
        "web_search",
        "web_deep_search",
        "web_search_recent",
        "papers_search",
        "encyclopedia_search",
        "github_search",
        "archive_search",
    }
)
AUTO_READ_TOP_K = 5
# Later rounds already hold fetched sources, so they only need enough new pages to test
# the round's hypothesis, not another full sweep of the catalog.
AUTO_READ_FOLLOWUP_TOP_K = 3


@dataclass(frozen=True)
class RoundView:
    """Everything the policy is allowed to know. Plain data, so the rules are testable
    without a graph, a model, or an MCP server."""

    # Steps the plan still holds, in order.
    remaining: tuple[str, ...] = ()
    # Unread candidate addresses, already filtered and already ranked by the passage
    # layer. The policy never re-orders them; ordering is WP2's job.
    ranked_unread: tuple[str, ...] = ()
    # Article links found on a page that really is a listing — hyperlinks, not a guess.
    drilldown: tuple[str, ...] = ()
    # Addresses the run has seen anywhere: search results, fetched page bodies, the task.
    known_urls: frozenset[str] = frozenset()
    # Canonical identities of steps already executed.
    completed_identities: frozenset[str] = frozenset()
    tool_just_run: str = ""
    has_sources: bool = False
    has_read_anything: bool = False
    open_conditions: int = 0
    total_conditions: int = 0
    tools: tuple = ()

    @property
    def discovery_just_ran(self) -> bool:
        return self.tool_just_run in AUTO_READ_DISCOVERY_TOOLS

    @property
    def read_already_queued(self) -> bool:
        return any(
            step.strip().lower().startswith(DISCOVERY_REQUIRED_PREFIXES)
            for step in self.remaining
        )

    @property
    def conditions_all_settled(self) -> bool:
        return bool(self.total_conditions) and not self.open_conditions


@dataclass(frozen=True)
class Action:
    """What the round does next. ``steps`` empty means the round is over."""

    steps: tuple[str, ...] = ()
    # Who chose: "listing_drilldown", "auto_rank", "model", "plan", "exhausted".
    source: str = "plan"
    reason: str = ""
    # Steps the model proposed that the policy would not run, and why.
    rejected: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def finishes(self) -> bool:
        return not self.steps

    @property
    def reads(self) -> tuple[str, ...]:
        return tuple(
            step[len("web_read: ") :]
            for step in self.steps
            if step.startswith("web_read: ")
        )


def admissible_steps(
    steps: list[str], view: RoundView
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split proposed steps into the ones the run may execute and the ones it may not.

    Both rules used to be post-hoc rewrites of an answer already given. As filters they
    are the same two rules and produce the same outcome, but the model's proposal is now
    narrowed on the way in rather than corrected on the way out.
    """
    kept: list[str] = []
    rejected: list[tuple[str, str]] = []
    for step in steps:
        prefix = next((p for p in URL_STEP_PREFIXES if step.startswith(p)), "")
        if prefix:
            url = step[len(prefix) :].strip().split("#", 1)[0].rstrip("/")
            if not url or url not in view.known_urls:
                # A plausible-looking government or vendor address the run never saw
                # resolves to a 404 stub and spends a fetch on nothing.
                rejected.append((step, "undiscovered_url"))
                continue
        if _step_identity(step, list(view.tools)) in view.completed_identities:
            # Re-proposing an executed step is not progress; without this the same three
            # queries come back round after round until the iteration cap ends the run.
            rejected.append((step, "already_executed"))
            continue
        kept.append(step)
    return kept, rejected


def _auto_reads(view: RoundView) -> list[str]:
    top_k = AUTO_READ_TOP_K if not view.has_read_anything else AUTO_READ_FOLLOWUP_TOP_K
    queued = {_step_identity(step, list(view.tools)) for step in view.remaining}
    reads = []
    for url in view.ranked_unread[:top_k]:
        step = f"web_read: {url}"
        identity = _step_identity(step, list(view.tools))
        if identity in queued:
            continue
        queued.add(identity)
        reads.append(step)
    return reads


def forced_reads(view: RoundView) -> tuple[list[str], str]:
    """Pages the state says to open whatever anyone thinks, and who says so.

    Both cases are decided by facts on the page, not by judgement: a listing page's own
    article links are real hyperlinks, and a discovery round that has queued no read is
    about to end having spent its budget on snippets alone.
    """
    if view.drilldown:
        return (
            [f"web_read: {url}" for url in view.drilldown],
            "listing_drilldown",
        )
    if view.discovery_just_ran and view.ranked_unread and not view.read_already_queued:
        reads = _auto_reads(view)
        if reads:
            return reads, "auto_rank"
    return [], ""


def should_ask_controller(view: RoundView, forced: list[str]) -> bool:
    """Whether the controller has a decision left worth a model call.

    With ``SKIP_CONTROLLER_WHEN_READS_FORCED`` off this is always true, which is the
    behaviour measured so far. With it on, a round whose reads are already decided does
    not pay for an opinion whose only remaining effect is to queue further searching
    behind pages it has not read yet — searching the next round re-derives from a state
    that includes those pages.
    """
    if not SKIP_CONTROLLER_WHEN_READS_FORCED:
        return True
    return not forced


def decide(
    view: RoundView,
    decision: str = "",
    proposed_steps: list[str] | None = None,
    forced: list[str] | None = None,
    forced_by: str = "",
) -> Action:
    """The round's next action. The only function that decides one.

    The order below is the whole transition table. Each line was a separate rewrite
    downstream of a decision already taken; here they compose in one place, in an order
    that is written down rather than emergent.
    """
    kept, rejected = admissible_steps([str(s) for s in (proposed_steps or [])], view)
    forced = list(forced or [])

    asked = bool((decision or "").strip())
    effective = (decision or "CONTINUE").strip().upper()
    # CONTINUE means "run the rest of the plan", so with nothing left it is a no-op that
    # ends the round having fetched nothing — reached exactly after a search whose
    # snippets look like the answer, which is when a page still has to be read. Only a
    # controller that was actually asked can be read this way: a controller that was
    # skipped has expressed no view, and must not be given one.
    if asked and effective == "CONTINUE" and not view.remaining:
        effective = "DONE"
    # A finished run needs at least one fetched source. Search snippets are not evidence:
    # accepting them here is what let a run report DONE and leave the ledger with nothing.
    if effective == "DONE" and not view.has_sources:
        effective = "NEXT"
    # Every condition the question sets is settled by a fetched source. More searching
    # cannot improve on that, whatever the controller believes.
    if view.conditions_all_settled and view.has_sources:
        effective = "DONE"

    if effective == "DONE":
        steps: list[str] = []
        source = "finish"
        reason = "the fetched sources cover every stated condition"
    elif kept:
        steps = kept + list(view.remaining)
        source = "model"
        reason = "controller proposed admissible next steps"
    else:
        steps = list(view.remaining)
        source = "plan"
        reason = "the plan still has steps" if steps else "nothing remains to run"

    # Forced reads compose on top rather than replacing: a decision to keep searching and
    # a page that must be opened are not in conflict, and reading comes first.
    if forced and effective != "DONE":
        steps = forced + [step for step in steps if step not in forced]
        source = forced_by or source
        reason = "pages the state requires reading, ahead of anything else"
    elif forced and forced_by == "listing_drilldown":
        # A listing page detected on this very batch is content the run has not seen at
        # all; finishing without opening it is finishing on the index page.
        steps = forced
        source = forced_by
        reason = "a listing page was detected on this batch and not yet opened"

    return Action(
        steps=tuple(steps),
        source=source,
        reason=reason,
        rejected=tuple(rejected),
    )
