"""命令行端到端演示（不启动 API / UI）。

    python scripts/demo.py --approve
    python scripts/demo.py --query "order-service 09:15 错误率升高" --reject
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from langgraph.types import Command  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.graph.state import DiagnosticState  # noqa: E402
from app.models import UserContext  # noqa: E402
from app.runtime import build_runtime  # noqa: E402

DEFAULT_QUERY = (
    "支付服务 payment-service 从 14:00 开始 P99 延迟从 200ms 涨到 1.8s，帮我看看可能原因。"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--role", default="oncall", choices=["viewer", "sre", "oncall", "admin"])
    parser.add_argument("--approve", action="store_true", help="自动批准首个高风险动作")
    parser.add_argument("--reject", action="store_true", help="拒绝高风险动作")
    parser.add_argument("--actor", default="demo-operator")
    args = parser.parse_args()

    runtime = build_runtime(get_settings())
    user = UserContext(user_id=args.actor, team_id="team-payments", role=args.role)
    session = runtime.sessions.get_or_create(f"demo-{uuid.uuid4().hex[:8]}", user)
    state = DiagnosticState(
        session_id=session.session_id,
        thread_id=session.thread_id,
        trace_id=uuid.uuid4().hex,
        user=user.model_dump(mode="json"),
        query=args.query,
        conversation=[],
        status="new",
        retry_count=0,
    )
    config = {"configurable": {"thread_id": session.thread_id}, "recursion_limit": 30}
    try:
        result = runtime.graph.invoke(state, config)
        _print_hypotheses(result)

        if "__interrupt__" in result:
            request = result.get("approval_request") or {}
            action = request.get("action") or {}
            print("\n=== 等待人工确认 ===")
            print(f"候选根因：{request.get('title')}（{request.get('confidence')}）")
            print(f"命令：{action.get('command')}")
            print("diff：")
            print(action.get("diff"))
            if args.approve or args.reject:
                decision = {
                    "approved": args.approve,
                    "approver_id": args.actor,
                    "approver_role": args.role,
                    "comment": "CLI demo 自动批准" if args.approve else "CLI demo 拒绝",
                }
                result = runtime.graph.invoke(Command(resume=decision), config)
            else:
                print("\n使用 --approve 或 --reject 继续。当前 Trace 已落盘。")
                print(f"trace_id={state['trace_id']} session_id={session.session_id}")
                return 0

        report = result.get("final_report") or {}
        print("\n=== 最终报告 ===\n")
        print(report.get("markdown") or json.dumps(result, ensure_ascii=False, indent=2)[:4000])
        print(f"\ntrace_id={state['trace_id']}  session_id={session.session_id}")
        print(f"memory={result.get('memory_result')}")
        return 0
    finally:
        runtime.close()


def _print_hypotheses(result: dict) -> None:
    intent = result.get("intent") or {}
    coverage = result.get("coverage") or {}
    print("=== 意图与覆盖度 ===")
    print(
        f"intent={intent.get('intent')} service={intent.get('service')} "
        f"window={intent.get('time_window', {}).get('start')} ~ {intent.get('time_window', {}).get('end')}"
    )
    print(f"present={coverage.get('present_sources')} missing={coverage.get('missing_sources')}")
    print(f"tool_calls={[(c.get('tool'), c.get('ok')) for c in result.get('tool_calls') or []]}")
    print("\n=== 候选根因 ===")
    for hyp in result.get("hypotheses") or []:
        print(
            f"{hyp.get('id')} {hyp.get('root_cause'):<20} conf={hyp.get('confidence')} "
            f"supported={hyp.get('supported')} action={hyp.get('action_required')} :: {hyp.get('title')}"
        )


if __name__ == "__main__":
    raise SystemExit(main())