"""LangGraph assembly.

Node implementations stay focused on workflow operations; this module is the only place
that owns the production transition table.
"""

from langgraph.graph import END, StateGraph
from mcp import ClientSession

from . import nodes
from .profiles import FOOTNOTE_PROFILE, Profile
from .state import AgentState
from .tool_policy import ToolAuthorizer


def build_graph(
    model: str,
    tools: list[dict],
    mcp_session: ClientSession | None = None,
    profile: Profile | None = None,
    tool_authorizer: ToolAuthorizer | None = None,
):
    profile = profile or FOOTNOTE_PROFILE
    graph = StateGraph(AgentState)

    async def requirements(state):
        return await nodes.requirements_node(state, model, tools, profile)

    async def plan(state):
        return await nodes.plan_node(state, model, tools, profile)

    async def execute(state):
        return await nodes.execute_node(
            state,
            model,
            tools,
            profile,
            mcp_session=mcp_session,
            tool_authorizer=tool_authorizer,
        )

    async def evidence_ledger(state):
        return await nodes.evidence_ledger_node(state, model, tools, profile)

    async def evidence_challenge(state):
        return await nodes.evidence_challenge_node(state, model, tools, profile)

    async def evidence_reextract(state):
        return await nodes.evidence_reextract_node(state, model, tools, profile)

    async def answer(state):
        return await nodes.answer_node(state, model, tools, profile)

    async def verify(state):
        return await nodes.verify_node(state, model, tools, profile)

    async def strategy(state):
        return await nodes.strategy_node(state, model, tools, profile)

    async def assimilate(state):
        return await nodes.assimilate_node(state, model, tools)

    for name, node in (
        ("requirements", requirements),
        ("plan", plan),
        ("execute", execute),
        ("evidence_ledger", evidence_ledger),
        ("evidence_challenge", evidence_challenge),
        ("evidence_reextract", evidence_reextract),
        ("answer", answer),
        ("verify", verify),
        ("strategy", strategy),
        ("assimilate", assimilate),
    ):
        graph.add_node(name, node)

    graph.set_entry_point("requirements")
    graph.add_edge("requirements", "plan")
    graph.add_conditional_edges(
        "plan",
        nodes.route_after_plan,
        {"execute": "execute", "answer": "answer", "assimilate": "assimilate"},
    )
    graph.add_conditional_edges(
        "execute",
        nodes.route_after_execute,
        {"execute": "execute", "evidence_ledger": "evidence_ledger"},
    )
    graph.add_conditional_edges(
        "evidence_ledger",
        nodes.route_after_evidence_ledger,
        {
            "evidence_challenge": "evidence_challenge",
            "evidence_reextract": "evidence_reextract",
        },
    )
    graph.add_conditional_edges(
        "evidence_challenge",
        nodes.route_after_evidence_challenge,
        {
            "execute": "execute",
            "answer": "answer",
            "strategy": "strategy",
            "assimilate": "assimilate",
            "evidence_reextract": "evidence_reextract",
        },
    )
    graph.add_conditional_edges(
        "evidence_reextract",
        nodes.route_after_evidence_reextract,
        {
            "evidence_challenge": "evidence_challenge",
            "answer": "answer",
            "assimilate": "assimilate",
        },
    )
    graph.add_edge("answer", "verify")
    graph.add_conditional_edges(
        "verify",
        nodes.route_after_verify,
        {"strategy": "strategy", "assimilate": "assimilate"},
    )
    graph.add_edge("strategy", "plan")
    graph.add_edge("assimilate", END)
    return graph.compile()
