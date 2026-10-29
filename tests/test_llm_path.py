from __future__ import annotations

from types import SimpleNamespace

from app.agents.hypothesis import generate_hypotheses
from app.core.llm import Usage
from app.models import HypothesisListOut
from app.runtime import build_runtime
from tests.conftest import make_settings


class FakeLLM:
    enabled = True

    def __init__(self, payload: dict):
        self.payload = payload

    def complete_json(self, messages, model_cls, **kwargs):
        parsed = model_cls.model_validate(self.payload)
        return SimpleNamespace(
            parsed=parsed,
            text="{}",
            usage=Usage(input_tokens=100, output_tokens=50, model="fake-model"),
            attempts=1,
        )


def _state():
    evidence = [
        {
            "id": "ev-change-1",
            "source": "change",
            "title": "配置变更 CHG-1",
            "timestamp": "2026-10-20T14:01:00+08:00",
            "raw_excerpt": "db.pool.max_size 50 -> 10",
            "confidence": 0.78,
            "metadata": {
                "change_id": "CHG-1",
                "change_type": "config",
                "risk": "high",
                "diff": "- db.pool.max_size: 50\n+ db.pool.max_size: 10",
            },
        },
        {
            "id": "ev-metric-1",
            "source": "metric",
            "title": "db_connection_wait_ms 异常",
            "timestamp": "2026-10-20T14:03:00+08:00",
            "raw_excerpt": "3ms -> 2810ms",
            "confidence": 0.9,
            "metadata": {"metric": "db_connection_wait_ms", "baseline": 3, "peak": 2810, "unit": "ms", "change_pct": 93566},
        },
    ]
    return {
        "session_id": "s-llm",
        "query": "payment-service 14:00 延迟",
        "user": {"user_id": "u", "team_id": "t", "role": "oncall"},
        "intent": {
            "service": "payment-service",
            "intent": "latency",
            "intent_label": "延迟/性能类",
            "time_window": {"start": "2026-10-20T13:00:00+08:00", "end": "2026-10-20T15:00:00+08:00", "anchor": "2026-10-20T14:00:00+08:00"},
        },
        "evidence": evidence,
        "coverage": {"missing_sources": [], "missing_details": []},
        "historical_cases": [],
    }


def test_llm_hypotheses_are_validated_and_action_is_deterministic(tmp_path):
    settings = make_settings(tmp_path)
    runtime = build_runtime(settings)
    try:
        fake = FakeLLM(
            {
                "hypotheses": [
                    {
                        "title": "连接池配置变更导致延迟",
                        "reasoning": "配置在异常前 1 分钟生效，连接等待指标同步上升。",
                        "root_cause": "config_change",
                        "confidence": 0.92,
                        "evidence_refs": ["ev-change-1", "ev-metric-1", "ev-not-exist"],
                        "action_required": "rollback",
                        "caveats": [],
                    }
                ],
                "notes": "single strong hypothesis",
            }
        )
        result = generate_hypotheses(_state(), llm=fake, settings=settings, trace_id="t")
        assert result["llm_usage"][0]["input_tokens"] == 100
        hyp = result["hypotheses"][0]
        assert hyp["generator"] == "llm"
        assert hyp["evidence_refs"] == ["ev-change-1", "ev-metric-1"]
        assert hyp["action_required"] == "rollback"
        assert "CHG-1" in hyp["proposed_action"]["command"]
        assert "+ db.pool.max_size: 50" in hyp["proposed_action"]["diff"]
    finally:
        runtime.close()


def test_llm_hypotheses_without_valid_refs_fall_back_without_crashing(tmp_path):
    settings = make_settings(tmp_path)
    runtime = build_runtime(settings)
    try:
        fake = FakeLLM(
            {
                "hypotheses": [
                    {
                        "title": "编造结论",
                        "reasoning": "引用不存在的证据",
                        "root_cause": "bad_deploy",
                        "confidence": 0.9,
                        "evidence_refs": ["ev-fake"],
                        "action_required": "rollback",
                        "caveats": [],
                    }
                ]
            }
        )
        result = generate_hypotheses(_state(), llm=fake, settings=settings, trace_id="t")
        # 没有有效引用 -> 回退规则引擎，仍产出可审计的假设
        assert result["hypotheses"]
        assert all(h.get("generator") == "rules" for h in result["hypotheses"])
        assert all(h.get("evidence_refs") for h in result["hypotheses"])
    finally:
        runtime.close()