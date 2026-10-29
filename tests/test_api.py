from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api import app
from app.config import reset_settings_cache

ROOT = Path(__file__).resolve().parent.parent


def _configure(monkeypatch, tmp_path):
    monkeypatch.setenv("SESSION_BACKEND", "memory")
    monkeypatch.setenv("LONG_TERM_BACKEND", "memory")
    monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("TRACING_ENABLED", "false")
    monkeypatch.setenv("KB_PATH", str(ROOT / "data" / "kb" / "docs.jsonl"))
    monkeypatch.setenv("AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("LLM_API_KEY", "")
    reset_settings_cache()


def test_api_chat_approval_and_guards(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    try:
        with TestClient(app) as client:
            health = client.get("/health").json()
            assert health["status"] == "ok"
            assert health["llm"]["enabled"] is False
            viewer_tools = client.get("/tools", params={"role": "viewer"}).json()["tools"]
            assert {t["name"] for t in viewer_tools} == {"query_metrics", "search_knowledge"}

            response = client.post(
                "/chat",
                json={
                    "message": "支付服务 payment-service 从 14:00 开始 P99 延迟涨到 1.8s",
                    "user_id": "u1",
                    "team_id": "team-payments",
                    "role": "oncall",
                },
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "awaiting_approval"
            assert payload["hypotheses"][0]["root_cause"] == "config_change"
            assert payload["evidence"]
            assert payload["timeline"]
            session_id = payload["session_id"]

            conflict = client.post(
                "/chat",
                json={
                    "message": "顺便再看下别的",
                    "session_id": session_id,
                    "user_id": "u1",
                    "team_id": "team-payments",
                    "role": "oncall",
                },
            )
            assert conflict.status_code == 409

            denied = client.post(
                f"/approvals/{session_id}",
                json={"approved": True, "approver_id": "v", "approver_role": "viewer"},
            )
            assert denied.status_code == 403

            approved = client.post(
                f"/approvals/{session_id}",
                json={
                    "approved": True,
                    "approver_id": "lead",
                    "approver_role": "oncall",
                    "comment": "ok",
                },
            )
            assert approved.status_code == 200
            final = approved.json()
            assert final["status"] == "action_executed"
            assert final["final_report"]["status"] == "action_executed"

            duplicate = client.post(
                f"/approvals/{session_id}",
                json={"approved": True, "approver_id": "lead", "approver_role": "oncall"},
            )
            assert duplicate.status_code == 409

            session = client.get(f"/sessions/{session_id}").json()
            assert session["open_approval"] is None
            assert client.get("/audit/approvals").json()["approved"] == 1
    finally:
        reset_settings_cache()


def test_stream_works_with_sqlite_checkpointer(monkeypatch, tmp_path):
    """回归：同步 SqliteSaver 不支持 graph.astream，SSE 必须走同步 stream。"""
    _configure(monkeypatch, tmp_path)
    monkeypatch.setenv("CHECKPOINT_BACKEND", "sqlite")
    reset_settings_cache()
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/chat/stream",
                json={
                    "message": "支付服务 payment-service 从 14:00 开始 P99 延迟涨到 1.8s",
                    "user_id": "u3",
                    "team_id": "team-payments",
                    "role": "oncall",
                },
            ) as response:
                assert response.status_code == 200
                body = "".join(response.iter_text())
            assert "event: done" in body
            assert "awaiting_approval" in body
            assert "event: interrupt" in body
    finally:
        reset_settings_cache()


def test_api_clarification_followup(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    try:
        with TestClient(app) as client:
            first = client.post(
                "/chat",
                json={"message": "P99 延迟很高，帮我看一下", "user_id": "u2", "team_id": "t2"},
            ).json()
            assert first["status"] == "needs_clarification"
            session_id = first["session_id"]

            second = client.post(
                "/chat",
                json={
                    "message": "payment-service，14:00 开始",
                    "session_id": session_id,
                    "user_id": "u2",
                    "team_id": "t2",
                },
            ).json()
            assert second["intent"]["service"] == "payment-service"
            assert second["status"] == "awaiting_approval"
            assert second["hypotheses"][0]["root_cause"] == "config_change"
    finally:
        reset_settings_cache()