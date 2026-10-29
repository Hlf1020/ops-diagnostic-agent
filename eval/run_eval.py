"""运行 Golden Dataset 评测。

用法：
    python -m eval.run_eval                 # 离线规则引擎（可复现）
    python -m eval.run_eval --llm           # 使用 .env 中的 LLM Key
    python -m eval.run_eval --limit 5       # 冒烟
    python -m eval.run_eval --fail-under-top3 0.9

输出：eval/out/report.json + eval/out/report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Windows 控制台默认 GBK，避免评测报告里的符号导致崩溃
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.config import Settings  # noqa: E402
from app.graph.state import DiagnosticState  # noqa: E402
from app.memory.session import session_context  # noqa: E402
from app.models import UserContext  # noqa: E402
from app.rag.knowledge import KnowledgeBase  # noqa: E402
from app.runtime import build_runtime  # noqa: E402
from eval.metrics import evaluate_ranked, hit_at_k, mean, percentile, reciprocal_rank  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_settings(*, use_llm: bool, tmp: Path) -> Settings:
    return Settings(
        llm_api_key="" if not use_llm else Settings().llm_api_key,
        llm_critic_enabled=use_llm,
        session_backend="memory",
        long_term_backend="memory",
        checkpoint_backend="memory",
        tracing_enabled=False,
        audit_path=str(tmp / "approvals.jsonl"),
        trace_dir=str(tmp / "traces"),
        kb_path=str(ROOT / "data" / "kb" / "docs.jsonl"),
    )


def run_one(case: dict, settings: Settings) -> dict:
    runtime = build_runtime(settings)
    user = UserContext(user_id="eval-runner", team_id="eval-team", role="oncall")
    session = runtime.sessions.get_or_create(f"eval-{case['id']}", user)
    state = DiagnosticState(
        session_id=session.session_id,
        thread_id=session.thread_id,
        trace_id=f"eval-{case['id']}",
        user=user.model_dump(mode="json"),
        query=case["query"],
        conversation=session_context(session, 4),
        status="new",
        retry_count=0,
    )
    config = {"configurable": {"thread_id": session.thread_id}, "recursion_limit": 30}
    started = time.perf_counter()
    try:
        result = runtime.graph.invoke(state, config)
    finally:
        latency_ms = (time.perf_counter() - started) * 1000
        runtime.close()

    hypotheses = result.get("hypotheses") or []
    ranked: list[str] = []
    for hyp in hypotheses:
        label = hyp.get("root_cause")
        if label and label not in ranked:
            ranked.append(label)
    top = hypotheses[0] if hypotheses else {}
    tool_calls = result.get("tool_calls") or []
    ok_calls = sum(1 for c in tool_calls if c.get("ok"))
    usage_records = result.get("llm_usage") or []
    tokens = sum(int(u.get("total_tokens", 0) or 0) for u in usage_records)
    cost = sum(float(u.get("cost_usd", 0) or 0) for u in usage_records)
    coverage = result.get("coverage") or {}
    critique = result.get("critique") or {}

    return {
        "id": case["id"],
        "scenario": case.get("scenario"),
        "expected_top1": case["expected_top1"],
        "expected_intent": case.get("expected_intent"),
        "expected_service": case.get("expected_service"),
        "ranked": ranked,
        "top1": ranked[0] if ranked else None,
        "top1_hit": bool(ranked) and ranked[0] == case["expected_top1"],
        "top3_hit": case["expected_top1"] in ranked[:3],
        "rr": reciprocal_rank(ranked, [case["expected_top1"]]),
        "intent": (result.get("intent") or {}).get("intent"),
        "intent_hit": (result.get("intent") or {}).get("intent") == case.get("expected_intent"),
        "service": (result.get("intent") or {}).get("service"),
        "service_hit": (result.get("intent") or {}).get("service") == case.get("expected_service"),
        "confidence": top.get("confidence"),
        "supported": top.get("supported"),
        "faithfulness": critique.get("faithfulness", 0.0),
        "approval_required": bool(result.get("needs_approval")),
        "action": top.get("action_required"),
        "evidence_count": len(result.get("evidence") or []),
        "missing_sources": coverage.get("missing_sources") or [],
        "tool_calls": len(tool_calls),
        "tool_ok": ok_calls,
        "tool_success_rate": (ok_calls / len(tool_calls)) if tool_calls else 0.0,
        "tokens": tokens,
        "cost_usd": cost,
        "latency_ms": round(latency_ms, 2),
        "status": result.get("status"),
        "error": result.get("tool_errors") or [],
    }


def run_retrieval(settings: Settings, cases: list[dict]) -> tuple[list[dict], dict]:
    kb = KnowledgeBase.load(settings.resolved_kb_path)
    records = []
    for case in cases:
        results = kb.search(case["query"], top_k=5)
        ranked = [r.document.id for r in results]
        records.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected": case["expected_doc_ids"],
                "ranked": ranked,
                "hit@3": hit_at_k(ranked, case["expected_doc_ids"], k=3),
                "rr": reciprocal_rank(ranked, case["expected_doc_ids"]),
                "scores": [round(r.score, 4) for r in results],
            }
        )
    summary = {
        "count": len(records),
        "hit@3": mean([1.0 if r["hit@3"] else 0.0 for r in records]),
        "mrr": mean([r["rr"] for r in records]),
        "hit@5": mean(
            [1.0 if hit_at_k(r["ranked"], r["expected"], k=5) else 0.0 for r in records]
        ),
    }
    return records, summary


def aggregate(records: list[dict], retrieval: dict, kb_summary: dict, *, llm_enabled: bool) -> dict:
    ranked_records = [{"ranked": r["ranked"], "expected": r["expected_top1"]} for r in records]
    diagnosis = evaluate_ranked(ranked_records, k=3)
    tool_total = sum(r["tool_calls"] for r in records)
    tool_ok = sum(r["tool_ok"] for r in records)
    latencies = [r["latency_ms"] for r in records]
    tokens = sum(r["tokens"] for r in records)
    cost = sum(r["cost_usd"] for r in records)
    return {
        "mode": "llm+rules" if llm_enabled else "rules-offline",
        "cases": len(records),
        "diagnosis": diagnosis,
        "retrieval": retrieval,
        "faithfulness": mean([r["faithfulness"] for r in records]),
        "supported_top1_rate": mean([1.0 if r["supported"] else 0.0 for r in records]),
        "intent_accuracy": mean([1.0 if r["intent_hit"] else 0.0 for r in records]),
        "service_accuracy": mean([1.0 if r["service_hit"] else 0.0 for r in records]),
        "tool_success_rate": round(tool_ok / tool_total, 4) if tool_total else 0.0,
        "tool_calls": tool_total,
        "p95_latency_ms": percentile(latencies, 95),
        "avg_latency_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "avg_evidence": mean([r["evidence_count"] for r in records]),
        "approval_required_rate": mean([1.0 if r["approval_required"] else 0.0 for r in records]),
        "tokens": tokens,
        "cost_usd": round(cost, 6),
        "cost_per_diagnosis_usd": round(cost / len(records), 6) if records else 0.0,
        "kb": kb_summary,
        "note": (
            "离线模式未调用 LLM，token/cost 为 0；人工驳回率需在生产审批日志中统计。"
            if not llm_enabled
            else "LLM 模式结果受模型与温度影响，建议固定版本做回归。"
        ),
    }


def render_markdown(summary: dict, records: list[dict], retrieval_records: list[dict]) -> str:
    lines = ["# Ops Diagnostic Agent 评测报告", ""]
    d = summary["diagnosis"]
    lines += [
        f"- 模式：**{summary['mode']}**",
        f"- 用例数：{summary['cases']}",
        f"- **Top-1 根因准确率：{d['top1_accuracy']:.2%}**",
        f"- **Top-3 根因命中率：{d['top3_accuracy']:.2%}**",
        f"- MRR：{d['mrr']:.4f}",
        f"- Faithfulness（平均）：{summary['faithfulness']:.3f}",
        f"- 首要假设被证据支撑比例：{summary['supported_top1_rate']:.2%}",
        f"- 意图准确率：{summary['intent_accuracy']:.2%}",
        f"- 服务识别准确率：{summary['service_accuracy']:.2%}",
        f"- 工具成功率：{summary['tool_success_rate']:.2%}（{summary['tool_calls']} 次调用）",
        f"- P95 诊断延迟：{summary['p95_latency_ms']} ms；平均：{summary['avg_latency_ms']} ms",
        f"- 平均证据数：{summary['avg_evidence']}",
        f"- 需要人工审批的比例：{summary['approval_required_rate']:.2%}",
        f"- Token / 成本：{summary['tokens']} / ${summary['cost_usd']}（单次 ${summary['cost_per_diagnosis_usd']}）",
        f"- 知识库：{summary['kb'].get('documents')} 篇",
        "",
        f"> {summary['note']}",
        "",
        "## 诊断用例",
        "",
        "| ID | 期望根因 | Top-1 | Top-3 | RR | 置信度 | 支撑度 | 工具 | 延迟(ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['id']} | {r['expected_top1']} | {r['top1']} | "
            f"{'✅' if r['top3_hit'] else '❌'} | {r['rr']:.2f} | {r['confidence']} | "
            f"{r['faithfulness']} | {r['tool_ok']}/{r['tool_calls']} | {r['latency_ms']} |"
        )
    lines += [
        "",
        "## 检索评测",
        "",
        f"- Hit@3：**{summary['retrieval']['hit@3']:.2%}**；Hit@5：{summary['retrieval']['hit@5']:.2%}；MRR：{summary['retrieval']['mrr']:.4f}",
        "",
        "| ID | 期望文档 | Top-3 命中 | RR | 返回前三 |",
        "|---|---|---|---|---|",
    ]
    for r in retrieval_records:
        lines.append(
            f"| {r['id']} | {', '.join(r['expected'])} | {'✅' if r['hit@3'] else '❌'} | "
            f"{r['rr']:.2f} | {', '.join(r['ranked'][:3])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="使用 .env 中的 LLM 配置")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-under-top3", type=float, default=0.0)
    parser.add_argument("--output", default=str(EVAL_DIR / "out"))
    args = parser.parse_args()

    cases = load_jsonl(EVAL_DIR / "golden_set.jsonl")
    retrieval_cases = load_jsonl(EVAL_DIR / "retrieval_set.jsonl")
    if args.limit:
        cases = cases[: args.limit]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        settings = make_settings(use_llm=args.llm, tmp=Path(tmp))
        actual_llm = settings.llm_enabled
        print(f"[eval] mode={'llm' if actual_llm else 'offline'} cases={len(cases)}")
        for idx, case in enumerate(cases, start=1):
            try:
                record = run_one(case, settings)
            except Exception as exc:
                record = {
                    "id": case["id"],
                    "expected_top1": case["expected_top1"],
                    "ranked": [],
                    "top1_hit": False,
                    "top3_hit": False,
                    "rr": 0.0,
                    "faithfulness": 0.0,
                    "supported": False,
                    "confidence": 0.0,
                    "intent_hit": False,
                    "service_hit": False,
                    "evidence_count": 0,
                    "tool_calls": 0,
                    "tool_ok": 0,
                    "tool_success_rate": 0.0,
                    "tokens": 0,
                    "cost_usd": 0,
                    "latency_ms": 0.0,
                    "approval_required": False,
                    "status": "error",
                    "error": str(exc),
                    "missing_sources": [],
                }
            records.append(record)
            print(
                f"[{idx}/{len(cases)}] {record['id']}: top1={record.get('top1')} "
                f"hit3={record['top3_hit']} faith={record['faithfulness']} "
                f"{record['latency_ms']}ms"
            )
        retrieval_records, retrieval_summary = run_retrieval(settings, retrieval_cases)

    summary = aggregate(
        records,
        retrieval_summary,
        KnowledgeBase.load(settings.resolved_kb_path).stats(),
        llm_enabled=actual_llm,
    )
    payload = {"summary": summary, "diagnosis_cases": records, "retrieval_cases": retrieval_records}
    (out_dir / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(summary, records, retrieval_records)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")

    print("\n" + markdown)
    print(f"[eval] wrote {out_dir / 'report.json'} and report.md")
    if args.fail_under_top3 and summary["diagnosis"]["top3_accuracy"] < args.fail_under_top3:
        print(
            f"[eval] FAIL: top3_accuracy {summary['diagnosis']['top3_accuracy']} < {args.fail_under_top3}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())