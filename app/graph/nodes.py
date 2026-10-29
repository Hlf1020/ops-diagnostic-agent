"""LangGraph 节点实现。所有副作用（工具、LLM、记忆写入、审计）都集中在这里。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from langgraph.types import interrupt

from app.agents.assembler import assemble_evidence
from app.agents.critic import apply_critic
from app.agents.hypothesis import generate_hypotheses
from app.agents.intent import classify_intent
from app.agents.report import build_final_report
from app.agents.subagents import AgentDeps, run_dimension
from app.config import Settings
from app.core.llm import LLMClient
from app.core.timeutil import CN_TZ, parse_iso
from app.memory.audit import AuditLog
from app.memory.long_term import LongTermMemory, make_case
from app.memory.session import SessionStore
from app.models import UserContext
from app.tools.gateway import ToolGateway, has_permission

logger = logging.getLogger(__name__)


@dataclass
class NodeDeps:
    settings: Settings
    llm: LLMClient
    gateway: ToolGateway
    memory: LongTermMemory
    sessions: SessionStore
    audit: AuditLog


class Nodes:
    def __init__(self, deps: NodeDeps):
        self.deps = deps
        self.agent_deps = AgentDeps(
            settings=deps.settings,
            llm=deps.llm,
            gateway=deps.gateway,
        )

    # ------------------------------------------------------------ routing
    def intent_router(self, state: dict) -> dict:
        result = classify_intent(
            state.get("query", ""),
            settings=self.deps.settings,
            llm=self.deps.llm,
            trace_id=state.get("trace_id"),
        )
        usage = result.pop("usage", None)
        missing = result.get("missing_slots") or []
        updates: dict = {
            "intent": result,
            "route": "clarify" if missing else "diagnose",
            "status": "needs_clarification" if missing else "diagnosing",
            "clarification_question": result.get("clarification_question"),
        }
        if usage:
            updates["llm_usage"] = [{**usage, "node": "intent_router"}]
        return updates

    def clarify(self, state: dict) -> dict:
        return {
            "status": "needs_clarification",
            "clarification_question": (state.get("intent") or {}).get("clarification_question")
            or "请提供更具体的服务名或时间窗口。",
        }

    def supervisor(self, state: dict) -> dict:
        focus = state.get("retry_focus") or []
        plan = {
            "branches": ["log", "metric", "change", "knowledge"],
            "focus": focus,
            "retry_count": int(state.get("retry_count", 0) or 0),
        }
        updates = {
            "supervisor_plan": plan,
            "status": "gathering_evidence",
            "needs_approval": False,
            "approval_request": None,
            "approval": None,
            "action_result": None,
        }
        if focus:
            updates["hypotheses"] = []
        return updates

    def fan_out(self, state: dict) -> list:
        from langgraph.types import Send

        payload = dict(state)
        return [
            Send("log_agent", payload),
            Send("metric_agent", payload),
            Send("change_agent", payload),
            Send("knowledge_agent", payload),
        ]

    # ------------------------------------------------------------ agents
    def log_agent(self, state: dict) -> dict:
        return run_dimension("log", state, self.agent_deps)

    def metric_agent(self, state: dict) -> dict:
        return run_dimension("metric", state, self.agent_deps)

    def change_agent(self, state: dict) -> dict:
        return run_dimension("change", state, self.agent_deps)

    def knowledge_agent(self, state: dict) -> dict:
        return run_dimension("knowledge", state, self.agent_deps)

    # ---------------------------------------------------------- reasoning
    def evidence_assembler(self, state: dict) -> dict:
        return assemble_evidence(
            state,
            settings=self.deps.settings,
            memory=self.deps.memory,
            trace_id=state.get("trace_id"),
        )

    def hypothesis_generator(self, state: dict) -> dict:
        return generate_hypotheses(
            state,
            llm=self.deps.llm,
            settings=self.deps.settings,
            trace_id=state.get("trace_id"),
        )

    def critic(self, state: dict) -> dict:
        return apply_critic(
            state,
            llm=self.deps.llm,
            settings=self.deps.settings,
            trace_id=state.get("trace_id"),
        )

    def route_after_critic(self, state: dict) -> str:
        critique = state.get("critique") or {}
        if critique.get("needs_retry"):
            return "supervisor"
        if state.get("needs_approval"):
            return "human_approval"
        return "final_writer"

    # ----------------------------------------------------------- approval
    def human_approval(self, state: dict) -> dict:
        request = state.get("approval_request") or {}
        decision = interrupt(request)
        if not isinstance(decision, dict):
            decision = {"approved": False, "comment": "invalid approval payload"}

        approved = bool(decision.get("approved"))
        approver_id = str(decision.get("approver_id") or "unknown")
        approver_role = str(decision.get("approver_role") or "viewer")
        comment = str(decision.get("comment") or "")
        notes = []

        if approved and not has_permission(approver_role, "remediation:write"):
            approved = False
            notes.append(f"approver role {approver_role} lacks remediation:write")
        expires_at = request.get("expires_at")
        if approved and expires_at:
            try:
                if parse_iso(expires_at) < datetime.now(timezone.utc):
                    approved = False
                    notes.append("approval request expired")
            except Exception:
                pass
        if not request:
            approved = False
            notes.append("no approval request in state")

        approval = {
            "approved": approved,
            "approver_id": approver_id,
            "approver_role": approver_role,
            "comment": comment,
            "notes": notes,
            "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hypothesis_id": request.get("hypothesis_id"),
        }
        self.deps.audit.record(
            {
                "event": "approval_decision",
                "session_id": state.get("session_id"),
                "trace_id": state.get("trace_id"),
                "hypothesis_id": request.get("hypothesis_id"),
                "root_cause": request.get("root_cause"),
                "action": (request.get("action") or {}).get("kind"),
                "target": (request.get("action") or {}).get("target"),
                "approved": approved,
                "approver_id": approver_id,
                "approver_role": approver_role,
                "comment": comment,
                "notes": notes,
            }
        )
        return {
            "approval": approval,
            "status": "approved" if approved else "rejected",
        }

    def route_after_approval(self, state: dict) -> str:
        # 拒绝也要经过 executor 生成可审计的 action_result（status=rejected）
        return "action_executor"

    def action_executor(self, state: dict) -> dict:
        request = state.get("approval_request") or {}
        approval = state.get("approval") or {}
        action = request.get("action") or {}
        if not approval.get("approved") or not action:
            result = {
                "status": "rejected",
                "approved": False,
                "approver": approval.get("approver_id"),
                "comment": approval.get("comment", ""),
                "error": "action was not approved",
            }
            return {"action_result": result, "status": "action_rejected"}

        session = self.deps.sessions.get(state.get("session_id", "")) if state.get("session_id") else None
        key = action.get("idempotency_key", "")
        if session is not None and key in session.executed_actions:
            return {
                "action_result": {
                    "status": "replayed",
                    "approved": True,
                    "approver": approval.get("approver_id"),
                    "result": session.executed_actions[key],
                    "idempotency_key": key,
                },
                "status": "action_executed",
            }

        user = state.get("user") or {}
        approver = {
            "user_id": approval.get("approver_id") or "unknown",
            "team_id": user.get("team_id", "team-default"),
            "role": approval.get("approver_role") or "viewer",
        }
        args = {
            "action": action.get("kind"),
            "service": (state.get("intent") or {}).get("service") or action.get("target"),
            "target": action.get("target") or "",
            "command": action.get("command") or "",
            "diff": action.get("diff") or "",
            "idempotency_key": key,
            "approved_by": approver["user_id"],
            "approval_note": approval.get("comment", ""),
        }
        tool_result = self.deps.gateway.call(
            "execute_remediation",
            args,
            UserContext.model_validate(approver),
            trace_id=state.get("trace_id"),
            node="action_executor",
        )
        if tool_result.ok:
            result = {
                "status": "simulated_success",
                "approved": True,
                "approver": approver["user_id"],
                "result": tool_result.output,
                "idempotency_key": key,
            }
            status = "action_executed"
        else:
            result = {
                "status": "failed",
                "approved": True,
                "approver": approver["user_id"],
                "error": (tool_result.error or {}).get("message"),
                "idempotency_key": key,
            }
            status = "action_failed"

        if session is not None and tool_result.ok:
            self.deps.sessions.record_executed(session, key, result)
        self.deps.audit.record(
            {
                "event": "action_executed" if tool_result.ok else "action_failed",
                "session_id": state.get("session_id"),
                "trace_id": state.get("trace_id"),
                "idempotency_key": key,
                "action": action.get("kind"),
                "target": action.get("target"),
                "command": action.get("command"),
                "approved_by": approver["user_id"],
                "ok": tool_result.ok,
                "error": (tool_result.error or {}).get("message"),
            }
        )
        return {"action_result": result, "status": status}

    # ------------------------------------------------------------ output
    def final_writer(self, state: dict) -> dict:
        report = build_final_report(state)
        return {
            "final_report": report,
            "status": report.get("status", "completed"),
        }

    def memory_writer(self, state: dict) -> dict:
        report = state.get("final_report")
        if not report:
            return {}
        top = (report.get("root_causes") or [{}])[0]
        approval = state.get("approval") or {}
        user = state.get("user") or {}
        case = make_case(
            team_id=user.get("team_id", "team-default"),
            user_id=user.get("user_id", "unknown"),
            service=(state.get("intent") or {}).get("service"),
            symptom=state.get("query", ""),
            intent=(state.get("intent") or {}).get("intent", "unknown"),
            root_cause=top.get("root_cause", "unknown"),
            conclusion=report.get("summary", ""),
            evidence_sources=[e.get("source") for e in state.get("evidence") or []],
            tools_called=[c.get("tool") for c in state.get("tool_calls") or []],
            human_feedback=approval.get("comment", "") if approval else "",
            action=top.get("action_required", "none"),
            approved=approval.get("approved") if approval else None,
            ttl_days=self.deps.settings.memory_ttl_days,
        )
        try:
            stored = self.deps.memory.add_case(case)
        except Exception as exc:
            logger.exception("long-term memory write failed")
            return {"memory_result": {"error": str(exc)[:300]}}

        session = self.deps.sessions.get(state.get("session_id", "")) if state.get("session_id") else None
        if session is not None:
            summary_line = (
                f"[{datetime.now(CN_TZ).strftime('%m-%d %H:%M')}] "
                f"{state.get('query', '')[:120]} -> {top.get('title')} "
                f"({top.get('root_cause')}, conf={top.get('confidence')}, {report.get('status')})"
            )
            session.summary = ((session.summary + "\n") if session.summary else "") + summary_line
            session.summary = session.summary[-1600:]
            session.append_turn(
                "assistant",
                report.get("summary", "")[:1500],
                {
                    "trace_id": state.get("trace_id"),
                    "root_cause": top.get("root_cause"),
                    "confidence": top.get("confidence"),
                    "status": report.get("status"),
                },
            )
            self.deps.sessions.save(session)

        return {
            "memory_result": {
                "case_id": stored.get("case_id"),
                "deduped": stored.get("deduped"),
                "backend": self.deps.memory.backend_name(),
            }
        }