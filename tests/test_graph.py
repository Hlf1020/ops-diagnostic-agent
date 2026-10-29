from __future__ import annotations

import uuid

from langgraph.types import Command

from app.agents.critic import apply_critic
from app.memory.session import session_context
from app.models import UserContext
from app.runtime import build_runtime
from tests.conftest import make_settings

QUERY = "支付服务 payment-service 从 14:00 开始 P99 延迟从 200ms 涨到 1.8s，帮我看看可能原因。"


def _run(runtime, query: str = QUERY, *, role: str = "oncall", session_id: str = "s1", team: str = "team-payments"):
    user = UserContext(user_id="u1", team_id=team, role=role)
    session = runtime.sessions.get_or_create(session_id, user)
    state = {
        "session_id": session.session_id,
        "thread_id": session.thread_id,
        "trace_id": uuid.uuid4().hex,
        "user": user.model_dump(mode="json"),
        "query": query,
        "conversation": session_context(session, 4),
        "status": "new",
        "retry_count": 0,
    }
    config = {"configurable": {"thread_id": session.thread_id}, "recursion_limit": 30}
    return session, runtime.graph.invoke(state, config), config


def test_full_diagnosis_and_approval(runtime):
    session, result, config = _run(runtime)
    assert "__interrupt__" in result
    hypotheses = result["hypotheses"]
    assert hypotheses[0]["root_cause"] == "config_change"
    assert hypotheses[0]["action_required"] == "rollback"
    assert "CHG-20261020-014" in hypotheses[0]["proposed_action"]["command"]
    assert result["coverage"]["missing_sources"] == []

    resumed = runtime.graph.invoke(
        Command(
            resume={
                "approved": True,
                "approver_id": "oncall-lead",
                "approver_role": "oncall",
                "comment": "diff 正确，批准回滚",
            }
        ),
        config,
    )
    assert resumed["status"] == "action_executed"
    report = resumed["final_report"]
    assert report["status"] == "action_executed"
    assert report["severity"] == "P1"
    assert "CHG-20261020-014" in report["markdown"]
    assert resumed["memory_result"]["backend"] == "memory"
    assert runtime.memory.stats()["cases"] == 1
    assert runtime.audit.approval_stats()["approved"] == 1
    assert runtime.sessions.get(session.session_id).summary


def test_rejection_is_audited(runtime):
    _, result, config = _run(runtime, session_id="s2")
    assert "__interrupt__" in result
    resumed = runtime.graph.invoke(
        Command(
            resume={
                "approved": False,
                "approver_id": "lead",
                "approver_role": "oncall",
                "comment": "证据不足，先观察",
            }
        ),
        config,
    )
    assert resumed["status"] == "action_rejected"
    assert resumed["action_result"]["status"] == "rejected"
    assert resumed["final_report"]["status"] == "action_rejected"
    stats = runtime.audit.approval_stats()
    assert stats["rejected"] == 1
    assert stats["rejection_rate"] == 1.0


def test_clarification_when_service_missing(runtime):
    _, result, _ = _run(runtime, "P99 延迟很高，帮我看一下", session_id="s3")
    assert result["status"] == "needs_clarification"
    assert result["intent"]["missing_slots"] == ["service"]
    assert not result.get("tool_calls")
    assert result.get("clarification_question")


def test_degraded_mode_when_metric_tool_fails(tmp_path):
    settings = make_settings(tmp_path, mock_fail_tools="query_metrics")
    runtime = build_runtime(settings)
    try:
        _, result, config = _run(runtime, session_id="s4")
        assert "metric" in result["coverage"]["missing_sources"]
        assert any(e["tool"] == "query_metrics" for e in result["tool_errors"])
        assert result["hypotheses"][0]["root_cause"] == "config_change"
        resumed = runtime.graph.invoke(
            Command(
                resume={
                    "approved": False,
                    "approver_id": "lead",
                    "approver_role": "oncall",
                    "comment": "指标缺失，先不执行",
                }
            ),
            config,
        )
        assert any("metric" in item for item in resumed["final_report"]["missing_information"])
    finally:
        runtime.close()


def test_viewer_role_is_denied_execution(runtime):
    _, result, config = _run(runtime, role="viewer", session_id="s5")
    assert "log" in result["coverage"]["missing_sources"]
    assert "change" in result["coverage"]["missing_sources"]
    assert "__interrupt__" in result
    resumed = runtime.graph.invoke(
        Command(
            resume={
                "approved": True,
                "approver_id": "viewer-lead",
                "approver_role": "viewer",
                "comment": "try to approve",
            }
        ),
        config,
    )
    assert resumed["status"] == "action_rejected"
    assert "lacks remediation:write" in " ".join(resumed["approval"]["notes"])


def test_critic_retry_is_capped(runtime):
    evidence = [
        {
            "id": "ev-metric-x",
            "source": "metric",
            "title": "CPU 异常",
            "raw_excerpt": "cpu 99%",
            "metadata": {"metric": "cpu_utilization_pct"},
            "confidence": 0.8,
        }
    ]
    hypothesis = {
        "id": "hyp-1",
        "title": "CPU 饱和",
        "reasoning": "cpu 高",
        "root_cause": "resource_exhaustion",
        "confidence": 0.8,
        "evidence_refs": ["ev-missing"],
        "action_required": "none",
        "proposed_action": None,
        "caveats": [],
    }
    base = {
        "evidence": evidence,
        "hypotheses": [hypothesis],
        "coverage": {
            "missing_sources": ["metric"],
            "missing_details": [{"source": "metric", "reason": "timeout", "retryable": True}],
        },
        "retry_count": 0,
    }
    first = apply_critic(base, llm=runtime.llm, settings=runtime.settings, trace_id="t")
    assert first["critique"]["needs_retry"] is True
    assert first["retry_count"] == 1
    assert first["retry_focus"] == ["metric"]
    assert first["hypotheses"][0]["confidence"] <= 0.55

    capped = dict(base)
    capped["retry_count"] = runtime.settings.max_retries
    second = apply_critic(capped, llm=runtime.llm, settings=runtime.settings, trace_id="t")
    assert second["critique"]["needs_retry"] is False
    assert second["retry_count"] == runtime.settings.max_retries