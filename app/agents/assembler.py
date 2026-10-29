"""证据对齐：去重、时间窗裁剪、来源限额、覆盖度计算、长期记忆召回。

这一层是控制 token 与抑制幻觉的关键：下游拿到的是 Evidence 集合，
不是四路工具的原始输出。
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.core.timeutil import expand_window, parse_iso, to_iso
from app.core.tracing import span
from app.memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)

EXPECTED_SOURCES = ["log", "metric", "change", "knowledge"]
CAPS = {"log": 5, "metric": 8, "change": 6, "knowledge": 4, "memory": 3}
TOOL_SOURCE = {
    "query_logs": "log",
    "query_metrics": "metric",
    "get_changes": "change",
    "search_knowledge": "knowledge",
    "execute_remediation": "action",
}


def _within(ts: str | None, start, end) -> bool:
    if not ts:
        return True
    try:
        value = parse_iso(ts)
    except Exception:
        return True
    return start <= value <= end


def _memory_evidence(case: dict) -> dict:
    similarity = float(case.get("similarity", 0) or 0)
    confidence = min(0.82, 0.4 + 0.4 * similarity)
    return {
        "id": f"ev-mem-{case.get('case_id')}",
        "source": "memory",
        "title": f"历史案例：{(case.get('symptom') or '')[:70]}",
        "timestamp": None,
        "service": case.get("service"),
        "raw_excerpt": (
            f"历史相似案例（相似度 {similarity:.3f}）：{case.get('symptom', '')}\n"
            f"结论：{case.get('conclusion', '')}"
        )[:700],
        "confidence": round(confidence, 3),
        "tool_trace_id": "memory:recall",
        "tool_name": "recall_memory",
        "metadata": {
            "case_id": case.get("case_id"),
            "root_cause": case.get("root_cause"),
            "similarity": similarity,
            "action": case.get("action"),
            "approved": case.get("approved"),
            "seen_count": case.get("seen_count"),
            "created_at": case.get("created_at"),
        },
        "untrusted": False,
        "injection_flags": [],
    }


def assemble_evidence(
    state: dict,
    *,
    settings: Settings,
    memory: LongTermMemory,
    trace_id: str | None,
) -> dict:
    intent = state.get("intent") or {}
    window = intent.get("time_window") or {}
    service = intent.get("service")
    team_id = (state.get("user") or {}).get("team_id", "team-default")

    if window.get("start") and window.get("end"):
        start, end = expand_window(window, settings.time_window_pad_minutes)
    else:
        start = end = None

    with span("node.evidence_assembler", trace_id=trace_id):
        deduped: dict[str, dict] = {}
        dropped_out_of_window = 0
        for item in state.get("raw_evidence") or []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            if start is not None and item.get("source") in {"log", "metric", "change"}:
                if not _within(item.get("timestamp"), start, end):
                    dropped_out_of_window += 1
                    continue
            existing = deduped.get(item["id"])
            if existing is None or item.get("confidence", 0) > existing.get("confidence", 0):
                deduped[item["id"]] = dict(item)

        # 长期记忆召回：只注入摘要与结论，控制在 top-k
        historical: list[dict] = []
        memory_evidence: list[dict] = []
        if service:
            query = f"{state.get('query', '')} {service} {intent.get('intent_label', '')}"
            try:
                cases = memory.search(query, team_id=team_id, top_k=settings.memory_top_k)
            except Exception:
                logger.exception("long-term memory recall failed")
                cases = []
            for case in cases:
                historical.append(
                    {
                        "case_id": case.get("case_id"),
                        "symptom": case.get("symptom"),
                        "conclusion": case.get("conclusion"),
                        "root_cause": case.get("root_cause"),
                        "similarity": case.get("similarity"),
                        "service": case.get("service"),
                        "created_at": case.get("created_at"),
                    }
                )
                memory_evidence.append(_memory_evidence(case))

        for item in memory_evidence:
            if item["id"] not in deduped:
                deduped[item["id"]] = item

        # 每来源按 confidence 截断，防止日志刷屏
        selected: list[dict] = []
        for source, cap in CAPS.items():
            items = [e for e in deduped.values() if e.get("source") == source]
            items.sort(key=lambda e: e.get("confidence", 0), reverse=True)
            selected.extend(items[:cap])

        selected.sort(key=lambda e: (e.get("timestamp") or "9999"), reverse=False)

        timeline = [
            {
                "time": e.get("timestamp"),
                "source": e.get("source"),
                "evidence_id": e.get("id"),
                "title": e.get("title"),
                "confidence": e.get("confidence"),
                "excerpt": (e.get("raw_excerpt") or "")[:160],
            }
            for e in selected
            if e.get("timestamp") and e.get("source") in {"log", "metric", "change"}
        ][:40]

        present = sorted({e.get("source") for e in selected})
        missing = [s for s in EXPECTED_SOURCES if s not in present]
        errors_by_source: dict[str, list[dict]] = {}
        for err in state.get("tool_errors") or []:
            source = TOOL_SOURCE.get(err.get("tool"), "other")
            errors_by_source.setdefault(source, []).append(err)

        missing_details = []
        for source in missing:
            errs = errors_by_source.get(source, [])
            retryable = any(bool(e.get("retryable")) for e in errs)
            if errs:
                reason = f"工具失败：{errs[0].get('error_type')} - {str(errs[0].get('message', ''))[:160]}"
            elif source == "change":
                reason = "查询窗口内没有变更记录"
            elif source == "knowledge":
                reason = "知识库未返回相关文档"
            elif source == "log":
                reason = "窗口内没有匹配的 WARN/ERROR 日志"
            elif source == "metric":
                reason = "窗口内没有检测到指标异常"
            else:
                reason = "该维度信息缺失"
            missing_details.append(
                {"source": source, "reason": reason, "retryable": retryable}
            )

        coverage = {
            "expected_sources": EXPECTED_SOURCES,
            "present_sources": present,
            "missing_sources": missing,
            "missing_details": missing_details,
            "tools_failed": sorted({e.get("tool") for e in state.get("tool_errors") or [] if e.get("tool")}),
            "window": {
                "start": to_iso(start) if start else window.get("start"),
                "end": to_iso(end) if end else window.get("end"),
                "label": window.get("label"),
                "detected_by": window.get("detected_by"),
            },
            "evidence_count": len(selected),
            "dropped_out_of_window": dropped_out_of_window,
        }

        return {
            "evidence": selected,
            "timeline": timeline,
            "coverage": coverage,
            "historical_cases": historical,
        }