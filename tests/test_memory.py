from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.timeutil import to_iso
from app.memory.long_term import InMemoryLongTermMemory, make_case


def _case(settings, *, symptom="payment-service P99 延迟升高", root_cause="config_change", team="team-a", service="payment-service", created_at=None):
    case = make_case(
        team_id=team,
        user_id="u",
        service=service,
        symptom=symptom,
        intent="latency",
        root_cause=root_cause,
        conclusion="回滚配置后恢复",
        evidence_sources=["metric", "change"],
        tools_called=["query_metrics", "get_changes"],
    )
    if created_at:
        case["created_at"] = created_at
    return case


def test_dedup_same_conclusion(settings):
    memory = InMemoryLongTermMemory(settings)
    first = memory.add_case(_case(settings))
    second = memory.add_case(_case(settings))
    assert first["deduped"] is False
    assert second["deduped"] is True
    assert memory.stats()["cases"] == 1
    assert second["case"]["seen_count"] == 2


def test_conflict_keeps_audit_and_supersedes(settings):
    memory = InMemoryLongTermMemory(settings)
    first = memory.add_case(_case(settings, root_cause="config_change"))
    second = memory.add_case(_case(settings, root_cause="bad_deploy"))
    assert second["deduped"] is False
    assert second["case"]["conflicts"][0]["previous_case_id"] == first["case_id"]
    stored_first = memory._cases[first["case_id"]]
    assert stored_first["superseded_by"] == second["case_id"]
    # 默认检索排除被取代的旧结论
    results = memory.search("payment-service P99", "team-a", top_k=5)
    assert all(r["case_id"] != first["case_id"] for r in results)


def test_team_isolation(settings):
    memory = InMemoryLongTermMemory(settings)
    memory.add_case(_case(settings, team="team-a"))
    assert memory.search("P99", "team-b", top_k=3) == []
    assert len(memory.search("P99", "team-a", top_k=3)) == 1


def test_ttl_prunes(settings):
    memory = InMemoryLongTermMemory(settings)
    old = to_iso(datetime.now(timezone.utc) - timedelta(days=200))
    memory.add_case(_case(settings, created_at=old))
    assert memory.stats()["cases"] == 0
    assert memory.search("P99", "team-a") == []