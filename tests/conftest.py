from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.runtime import build_runtime

ROOT = Path(__file__).resolve().parent.parent


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        llm_api_key="",
        llm_critic_enabled=False,
        session_backend="memory",
        long_term_backend="memory",
        checkpoint_backend="memory",
        tracing_enabled=False,
        kb_path=str(ROOT / "data" / "kb" / "docs.jsonl"),
        audit_path=str(tmp_path / "audit" / "approvals.jsonl"),
        trace_dir=str(tmp_path / "traces"),
        checkpoint_db=str(tmp_path / "checkpoints.sqlite"),
        mock_fail_tools="",
        mock_delay="",
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture()
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture()
def runtime(settings):
    rt = build_runtime(settings)
    yield rt
    rt.close()