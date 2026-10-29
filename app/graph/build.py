"""LangGraph 编译。

并行 fan-out 使用 Send；四路都返回后通过 join 边进入 evidence_assembler。
人工确认使用 interrupt + checkpointer，恢复时用 Command(resume=...)。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import Nodes
from app.graph.state import DiagnosticState


def build_graph(nodes: Nodes, checkpointer=None):
    builder = StateGraph(DiagnosticState)

    builder.add_node("intent_router", nodes.intent_router)
    builder.add_node("clarify", nodes.clarify)
    builder.add_node("supervisor", nodes.supervisor)
    builder.add_node("log_agent", nodes.log_agent)
    builder.add_node("metric_agent", nodes.metric_agent)
    builder.add_node("change_agent", nodes.change_agent)
    builder.add_node("knowledge_agent", nodes.knowledge_agent)
    builder.add_node("evidence_assembler", nodes.evidence_assembler)
    builder.add_node("hypothesis_generator", nodes.hypothesis_generator)
    builder.add_node("critic", nodes.critic)
    builder.add_node("human_approval", nodes.human_approval)
    builder.add_node("action_executor", nodes.action_executor)
    builder.add_node("final_writer", nodes.final_writer)
    builder.add_node("memory_writer", nodes.memory_writer)

    builder.add_edge(START, "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        lambda state: state.get("route", "diagnose"),
        {"clarify": "clarify", "diagnose": "supervisor"},
    )
    builder.add_edge("clarify", END)

    builder.add_conditional_edges("supervisor", nodes.fan_out)
    builder.add_edge(
        ["log_agent", "metric_agent", "change_agent", "knowledge_agent"],
        "evidence_assembler",
    )
    builder.add_edge("evidence_assembler", "hypothesis_generator")
    builder.add_edge("hypothesis_generator", "critic")
    builder.add_conditional_edges(
        "critic",
        nodes.route_after_critic,
        {
            "supervisor": "supervisor",
            "human_approval": "human_approval",
            "final_writer": "final_writer",
        },
    )
    builder.add_conditional_edges(
        "human_approval",
        nodes.route_after_approval,
        {"action_executor": "action_executor"},
    )
    builder.add_edge("action_executor", "final_writer")
    builder.add_edge("final_writer", "memory_writer")
    builder.add_edge("memory_writer", END)

    return builder.compile(checkpointer=checkpointer)