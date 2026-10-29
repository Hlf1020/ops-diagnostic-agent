"""领域模型。

编排状态刻意使用普通 dict（见 app/graph/state.py）：LangGraph 的 checkpoint
需要 msgpack 序列化，dict 最稳。Pydantic 模型只用于 API 边界、工具入参和
LLM 结构化输出校验。
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Intent = Literal["latency", "error_rate", "resource", "change_related", "unknown"]
RootCause = Literal[
    "config_change",
    "bad_deploy",
    "traffic_surge",
    "dependency_failure",
    "resource_exhaustion",
    "network_issue",
    "unknown",
]
ActionKind = Literal["none", "rollback", "restart", "scale", "cleanup"]
EvidenceSource = Literal["log", "metric", "change", "knowledge", "memory"]


class Role(str, Enum):
    VIEWER = "viewer"
    SRE = "sre"
    ONCALL = "oncall"
    ADMIN = "admin"


class UserContext(BaseModel):
    user_id: str = Field(default="demo-user", min_length=1, max_length=64)
    team_id: str = Field(default="team-payments", min_length=1, max_length=64)
    role: Role = Role.ONCALL


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=80)
    user_id: str = Field(default="demo-user", max_length=64)
    team_id: str = Field(default="team-payments", max_length=64)
    role: Role = Role.ONCALL

    @field_validator("message")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class ApprovalRequest(BaseModel):
    approved: bool
    approver_id: str = Field(default="oncall-lead", min_length=1, max_length=64)
    approver_role: Role = Role.ONCALL
    comment: str = Field(default="", max_length=1000)
    confirm_token: str | None = Field(default=None, max_length=128)


class IntentOut(BaseModel):
    intent: Intent
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    service: str | None = None
    rationale: str = ""


class ClaimOut(BaseModel):
    hypothesis_id: str
    supported: bool
    reason: str = ""
    unsupported_claims: list[str] = Field(default_factory=list)


class CriticOut(BaseModel):
    verdicts: list[ClaimOut] = Field(default_factory=list)
    notes: str = ""


class HypothesisOut(BaseModel):
    title: str = Field(..., min_length=2, max_length=200)
    reasoning: str = Field(default="", max_length=3000)
    root_cause: RootCause
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    action_required: ActionKind = "none"
    caveats: list[str] = Field(default_factory=list)


class HypothesisListOut(BaseModel):
    hypotheses: list[HypothesisOut] = Field(default_factory=list, max_length=6)
    notes: str = ""


class EvidenceItem(BaseModel):
    id: str
    source: EvidenceSource
    title: str
    timestamp: str | None = None
    service: str | None = None
    raw_excerpt: str
    confidence: float = Field(ge=0.0, le=1.0)
    tool_trace_id: str
    tool_name: str
    metadata: dict = Field(default_factory=dict)
    untrusted: bool = True
    injection_flags: list[str] = Field(default_factory=list)


class ProposedAction(BaseModel):
    kind: ActionKind
    target: str
    command: str
    diff: str = ""
    rationale: str = ""
    risk: Literal["low", "medium", "high"] = "medium"
    dry_run: bool = True
    idempotency_key: str


class ChatResponse(BaseModel):
    session_id: str
    thread_id: str
    trace_id: str
    status: str
    message: str | None = None
    clarification_question: str | None = None
    intent: dict | None = None
    approval_request: dict | None = None
    hypotheses: list[dict] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    timeline: list[dict] = Field(default_factory=list)
    final_report: dict | None = None
    coverage: dict | None = None
    usage: dict = Field(default_factory=dict)
    latency_ms: float = 0.0