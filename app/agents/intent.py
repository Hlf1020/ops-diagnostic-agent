"""意图路由与槽位抽取。

分类节点只负责“这是什么类型的问题”；服务名与时间窗口由确定性解析器优先，
LLM 只能补充，不能覆盖已解析出的明确值。低置信度且缺少服务名时走澄清，
不硬猜。
"""

from __future__ import annotations

import logging
import re

from app.config import Settings
from app.core.llm import LLMClient, LLMError
from app.core.timeutil import parse_time_window
from app.core.tracing import span
from app.models import IntentOut
from app.tools.scenario_data import KNOWN_SERVICES, resolve_service

logger = logging.getLogger(__name__)

INTENT_KEYWORDS: dict[str, list[str]] = {
    "latency": ["延迟", "变慢", "响应变慢", "很慢", "卡顿", "p99", "latency", "响应时间", "超时", "耗时"],
    "error_rate": ["错误率", "报错", "失败率", "失败", "5xx", "500", "error", "异常率", "不可用", "宕机"],
    "resource": [
        "cpu",
        "内存",
        "memory",
        "磁盘",
        "disk",
        "io",
        "io wait",
        "资源",
        "负载",
        "oom",
        "oomkilled",
        "重启",
        "heap",
        "qps",
        "流量",
        "排队",
        "扩容",
        "限流",
        "throttl",
        "oserror",
        "no space",
        "dns",
        "网络",
        "重传",
        "tcp",
        "no such host",
    ],
    "change_related": ["变更", "发布", "上线", "部署", "配置", "回滚", "deploy", "release", "change"],
}

# 显式资源/错误指标权重更高，避免“磁盘满了导致延迟”被误判为纯延迟问题
KEYWORD_WEIGHTS: dict[str, int] = {
    "错误率": 3,
    "失败率": 3,
    "不可用": 3,
    "p99": 3,
    "cpu": 3,
    "磁盘": 3,
    "qps": 3,
    "oom": 3,
    "oserror": 3,
    "no space": 3,
    "dns": 3,
    "延迟": 2,
    "响应变慢": 3,
    "超时": 2,
    "5xx": 2,
    "500": 2,
    "内存": 2,
    "io wait": 2,
    "网络": 2,
    "重传": 2,
    "流量": 2,
    "排队": 2,
    "扩容": 2,
    "限流": 2,
    "throttl": 2,
    "oomkilled": 2,
    "tcp": 1,
    "no such host": 2,
    "变更": 1,
    "发布": 1,
    "部署": 1,
    "配置": 1,
    "回滚": 1,
}


def _keyword_present(text: str, keyword: str) -> bool:
    """ASCII 关键词用词边界，避免 OSError 被 'error' 误命中。"""
    if re.fullmatch(r"[a-z0-9 ]+", keyword):
        return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None
    return keyword in text

INTENT_LABELS = {
    "latency": "延迟/性能类",
    "error_rate": "错误率/可用性类",
    "resource": "资源/容量类",
    "change_related": "变更关联类",
    "unknown": "未知类型",
}

SYSTEM_PROMPT = """你是企业运维诊断 Agent 的意图路由器。只做分类，不要诊断，不要编造事实。
把用户请求归入以下之一：
- latency：延迟、变慢、P99、超时、响应时间问题
- error_rate：错误率、5xx、报错、失败、不可用
- resource：CPU、内存、磁盘、IO、容量、OOM、重启
- change_related：明确询问发布/配置/变更是否导致问题
- unknown：信息不足或不属于以上类型

只返回 JSON：
{"intent": "...", "confidence": 0.0-1.0, "service": "服务名或 null", "rationale": "一句话中文理由"}
服务名只能从以下列表选择，无法确定时必须返回 null：""" + ", ".join(KNOWN_SERVICES)


def heuristic_intent(query: str) -> tuple[str, float, list[str]]:
    text = query.lower()
    scores: dict[str, int] = {}
    hits: dict[str, list[str]] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        matched = [kw for kw in keywords if _keyword_present(text, kw.lower())]
        if matched:
            scores[intent] = sum(KEYWORD_WEIGHTS.get(kw, 1) for kw in matched)
            hits[intent] = matched
    if not scores:
        return "unknown", 0.35, []
    best = max(scores, key=lambda k: scores[k])
    # 平票时按业务优先级：错误率 > 延迟 > 资源 > 变更
    priority = ["error_rate", "latency", "resource", "change_related"]
    top_score = scores[best]
    tied = [k for k, v in scores.items() if v == top_score]
    best = sorted(tied, key=lambda k: priority.index(k) if k in priority else 99)[0]
    confidence = min(0.95, 0.55 + 0.04 * top_score)
    return best, confidence, hits.get(best, [])


def classify_intent(
    query: str,
    *,
    settings: Settings,
    llm: LLMClient | None,
    trace_id: str | None = None,
) -> dict:
    window = parse_time_window(
        query,
        settings.mock_now,
        pad_minutes=settings.explicit_window_minutes,
        default_hours=settings.default_window_hours,
    )
    service = resolve_service(query)

    with span("node.intent_router", trace_id=trace_id, attrs={"app.query_chars": len(query)}):
        heuristic, confidence, hits = heuristic_intent(query)
        intent = heuristic
        rationale = f"规则命中：{', '.join(hits)}" if hits else "未命中明确关键词，降级为通用诊断"
        source = "rules"
        usage = None

        if llm is not None and llm.enabled:
            try:
                result = llm.complete_json(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    IntentOut,
                    trace_id=trace_id,
                    node="intent_router",
                )
                parsed = result.parsed
                llm_service = parsed.service if parsed.service in KNOWN_SERVICES else None
                llm_intent = parsed.intent
                # 明确服务名以确定性解析为准；LLM 只在解析失败时补充
                if service is None and llm_service:
                    service = llm_service
                if llm_intent != "unknown" or heuristic == "unknown":
                    intent = llm_intent
                confidence = max(0.0, min(1.0, parsed.confidence))
                rationale = parsed.rationale or rationale
                source = "llm"
                usage = result.usage.as_dict(settings)
            except LLMError as exc:
                logger.warning("LLM intent classification failed, fallback to rules: %s", exc)

        missing = [] if service else ["service"]
        question = None
        if missing:
            question = (
                "请提供服务名或确认目标服务。已知服务示例：payment-service、order-service、"
                "search-service。你也可以贴出监控告警里的服务名。"
            )

        return {
            "intent": intent,
            "intent_label": INTENT_LABELS.get(intent, INTENT_LABELS["unknown"]),
            "confidence": round(confidence, 3),
            "service": service,
            "time_window": window,
            "missing_slots": missing,
            "clarification_question": question,
            "rationale": rationale,
            "source": source,
            "usage": usage,
        }