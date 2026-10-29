"""离线一键自检：6 个场景 + 澄清 + 权限降级 + 工具故障。

    python scripts/selfcheck.py

不调用 LLM，不写项目 data 目录；适合改代码后快速回归。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.config import Settings  # noqa: E402
from app.memory.session import session_context  # noqa: E402
from app.models import UserContext  # noqa: E402
from app.runtime import build_runtime  # noqa: E402

SCENARIOS = [
    ("payment-service", "支付服务 payment-service 从 14:00 开始 P99 延迟涨到 1.8s", "config_change", True),
    ("order-service", "order-service 09:15 错误率涨到 12.6%，日志有 OOMKilled", "bad_deploy", True),
    ("search-service", "search-service 11:40 QPS 从 1200 涨到 5200，CPU 96% 被限流", "traffic_surge", True),
    ("inventory-service", "inventory-service 16:10 错误率升高，Redis connection refused", "dependency_failure", False),
    ("notification-service", "notification-service 18:05 磁盘 99%，No space left on device", "resource_exhaustion", True),
    ("user-service", "user-service 20:00 DNS 解析失败，TCP 重传，P99 升高", "network_issue", False),
]


def offline_settings(tmp: Path, **overrides) -> Settings:
    return Settings(
        llm_api_key="",
        llm_critic_enabled=False,
        session_backend="memory",
        long_term_backend="memory",
        checkpoint_backend="memory",
        tracing_enabled=False,
        kb_path=str(ROOT / "data" / "kb" / "docs.jsonl"),
        audit_path=str(tmp / "audit.jsonl"),
        trace_dir=str(tmp / "traces"),
        **overrides,
    )


def run(runtime, query: str, session_id: str, role: str = "oncall"):
    user = UserContext(user_id="selfcheck", team_id="team-check", role=role)
    session = runtime.sessions.get_or_create(session_id, user)
    state = {
        "session_id": session.session_id,
        "thread_id": session.thread_id,
        "trace_id": f"selfcheck-{session_id}",
        "user": user.model_dump(mode="json"),
        "query": query,
        "conversation": session_context(session, 4),
        "status": "new",
        "retry_count": 0,
    }
    config = {"configurable": {"thread_id": session.thread_id}, "recursion_limit": 30}
    return runtime.graph.invoke(state, config)


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for idx, (service, query, expected, expect_approval) in enumerate(SCENARIOS, start=1):
            runtime = build_runtime(offline_settings(tmp))
            try:
                result = run(runtime, query, f"scenario-{idx}")
                ranked = [h.get("root_cause") for h in result.get("hypotheses") or []]
                interrupted = "__interrupt__" in result
                top1_ok = bool(ranked) and ranked[0] == expected
                top3_ok = expected in ranked[:3]
                approval_ok = interrupted == expect_approval
                status = "PASS" if (top1_ok and top3_ok and approval_ok) else "FAIL"
                print(
                    f"[{status}] {service:<22} top1={ranked[0] if ranked else None:<20} "
                    f"approval={interrupted} evidence={len(result.get('evidence') or [])} "
                    f"missing={result.get('coverage', {}).get('missing_sources')}"
                )
                if status == "FAIL":
                    failures.append(f"{service}: top1_ok={top1_ok} top3_ok={top3_ok} approval_ok={approval_ok}")
            finally:
                runtime.close()

        # 澄清路径
        runtime = build_runtime(offline_settings(tmp))
        try:
            result = run(runtime, "P99 延迟很高，帮我看一下", "clarify")
            ok = result.get("status") == "needs_clarification" and not result.get("tool_calls")
            print(f"[{'PASS' if ok else 'FAIL'}] clarification          status={result.get('status')}")
            if not ok:
                failures.append("clarification path")
        finally:
            runtime.close()

        # viewer 权限降级：不能查日志/变更，但流程不崩
        runtime = build_runtime(offline_settings(tmp))
        try:
            result = run(runtime, SCENARIOS[0][1], "viewer", role="viewer")
            missing = set(result.get("coverage", {}).get("missing_sources") or [])
            ok = {"log", "change"} <= missing and bool(result.get("hypotheses"))
            print(f"[{'PASS' if ok else 'FAIL'}] viewer degradation      missing={sorted(missing)}")
            if not ok:
                failures.append("viewer degradation")
        finally:
            runtime.close()

        # 指标工具故障：降级且报告缺失维度，不编造
        runtime = build_runtime(offline_settings(tmp, mock_fail_tools="query_metrics"))
        try:
            result = run(runtime, SCENARIOS[0][1], "tool-failure")
            missing = set(result.get("coverage", {}).get("missing_sources") or [])
            ok = "metric" in missing and bool(result.get("hypotheses"))
            print(f"[{'PASS' if ok else 'FAIL'}] tool failure            missing={sorted(missing)}")
            if not ok:
                failures.append("tool failure degradation")
        finally:
            runtime.close()

    print()
    if failures:
        print(f"SELF-CHECK FAILED: {len(failures)} issue(s)")
        for item in failures:
            print(f"- {item}")
        return 1
    print("SELF-CHECK PASSED: 6 scenarios + clarification + permission + tool failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())