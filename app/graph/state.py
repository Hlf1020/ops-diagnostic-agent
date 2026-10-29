"""LangGraph 状态定义。

并行分支用 reducer 追加 `raw_evidence / tool_calls / tool_errors / plans / llm_usage`；
`evidence` 是 assembler 产出的规范集合，不带 reducer，避免重试时无限累积。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class DiagnosticState(TypedDict, total=False):
    session_id: str
    thread_id: str
    trace_id: str
    user: dict
    query: str
    conversation: list[dict]
    status: str
    route: str

    intent: dict
    supervisor_plan: dict

    raw_evidence: Annotated[list[dict], operator.add]
    evidence: list[dict]
    timeline: list[dict]
    coverage: dict
    historical_cases: list[dict]

    tool_calls: Annotated[list[dict], operator.add]
    tool_errors: Annotated[list[dict], operator.add]
    plans: Annotated[list[dict], operator.add]
    llm_usage: Annotated[list[dict], operator.add]

    hypotheses: list[dict]
    hypothesis_notes: str
    critique: dict
    retry_count: int
    retry_focus: list[str]
    needs_approval: bool
    approval_request: dict
    approval: dict
    action_result: dict

    final_report: dict
    memory_result: dict
    clarification_question: str