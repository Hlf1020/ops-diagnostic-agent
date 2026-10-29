"""Faithfulness critic：检查每条假设是否真的被证据支撑。

确定性检查保证离线评测可复现；配置了 LLM 时再叠加一次 LLM-as-judge，
但 LLM 只能收紧结论，不能把无证据的假设“说成”有证据。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.config import Settings
from app.core.llm import LLMClient, LLMError
from app.core.timeutil import to_iso
from app.core.tracing import span
from app.models import CriticOut

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """你是严格的证据审查员（faithfulness critic）。给定假设与其引用的证据，判断假设中的关键结论是否能被这些证据直接支持。
规则：
- 只能依据提供的证据，不得引入外部知识或自行补充事实；
- 引用 id 不存在、引用证据与根因类别无关、或结论超出证据范围，都判定 supported=false；
- 只返回 JSON：
{"verdicts":[{"hypothesis_id":"hyp-1","supported":true,"reason":"中文理由","unsupported_claims":["..."]}],"notes":"..."}"""


def _label_support(hypothesis: dict, evidence_map: dict[str, dict], valid_refs: list[str]) -> tuple[bool, str]:
    label = hypothesis.get("root_cause")
    refs = [evidence_map[r] for r in valid_refs if r in evidence_map]
    sources = {e.get("source") for e in refs}
    categories = {
        (e.get("metadata") or {}).get("category")
        for e in refs
        if e.get("source") == "log"
    }
    metrics = {
        (e.get("metadata") or {}).get("metric")
        for e in refs
        if e.get("source") == "metric"
    }
    change_types = {
        (e.get("metadata") or {}).get("change_type")
        for e in refs
        if e.get("source") == "change"
    }
    if label == "config_change":
        ok = "config" in change_types or bool(categories & {"connection_pool", "gc"})
        return ok, "配置变更或连接池/GC 日志" if ok else "缺少配置变更或连接池相关证据"
    if label == "bad_deploy":
        ok = "deploy" in change_types or bool(categories & {"oom", "error"})
        return ok, "发布记录或 OOM/异常堆栈日志" if ok else "缺少发布记录或 OOM/异常堆栈证据"
    if label == "traffic_surge":
        ok = "request_rate_qps" in metrics
        return ok, "QPS 指标" if ok else "缺少 QPS 指标"
    if label == "dependency_failure":
        ok = "dependency_error_rate_pct" in metrics or "dependency" in categories
        return ok, "依赖错误率指标或依赖连接失败日志" if ok else "缺少依赖侧证据"
    if label == "resource_exhaustion":
        ok = bool(metrics & {"disk_usage_pct", "io_wait_pct", "cpu_utilization_pct", "cpu_throttled_pct", "memory_usage_pct", "pod_restarts"}) or bool(
            categories & {"disk", "oom"}
        )
        return ok, "资源指标或磁盘/OOM 日志" if ok else "缺少资源类证据"
    if label == "network_issue":
        ok = bool(metrics & {"dns_error_rate_pct", "network_retransmit_pct"}) or bool(categories & {"dns", "network"})
        return ok, "DNS/网络指标或日志" if ok else "缺少 DNS/网络证据"
    if label == "unknown":
        return True, "未知根因不要求类别证据"
    return False, "未知根因类别"


def _llm_critic(
    hypotheses: list[dict],
    evidence_map: dict[str, dict],
    *,
    llm: LLMClient,
    settings: Settings,
    trace_id: str | None,
) -> tuple[dict[str, dict], dict | None]:
    if not llm.enabled or not settings.llm_critic_enabled:
        return {}, None
    top = hypotheses[:3]
    if not top:
        return {}, None
    lines = []
    for hyp in top:
        evidence_text = "\n".join(
            f"- [{ref}] {evidence_map.get(ref, {}).get('source')}: {evidence_map.get(ref, {}).get('title')} | "
            f"{(evidence_map.get(ref, {}).get('raw_excerpt') or '')[:220]}"
            for ref in hyp.get("evidence_refs", [])
        )
        lines.append(
            f"假设 {hyp.get('id')}: {hyp.get('title')}\nroot_cause={hyp.get('root_cause')}\n"
            f"reasoning={hyp.get('reasoning')}\n引用证据:\n{evidence_text}"
        )
    try:
        result = llm.complete_json(
            [
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(lines)},
            ],
            CriticOut,
            trace_id=trace_id,
            node="faithfulness_critic",
        )
    except LLMError as exc:
        logger.warning("LLM critic failed, using deterministic checks only: %s", exc)
        return {}, None
    verdicts = {v.hypothesis_id: v.model_dump() for v in result.parsed.verdicts}
    return verdicts, {**result.usage.as_dict(settings), "node": "faithfulness_critic"}


def apply_critic(
    state: dict,
    *,
    llm: LLMClient,
    settings: Settings,
    trace_id: str | None,
) -> dict:
    hypotheses = [dict(h) for h in (state.get("hypotheses") or [])]
    evidence = state.get("evidence") or []
    evidence_map = {e.get("id"): e for e in evidence}
    coverage = state.get("coverage") or {}
    retry_count = int(state.get("retry_count", 0) or 0)

    with span("node.faithfulness_critic", trace_id=trace_id, attrs={"app.hypotheses": len(hypotheses)}):
        llm_verdicts, critic_usage = _llm_critic(
            hypotheses, evidence_map, llm=llm, settings=settings, trace_id=trace_id
        )
        checks = []
        for hyp in hypotheses:
            refs = list(hyp.get("evidence_refs") or [])
            valid = [r for r in refs if r in evidence_map]
            invalid = [r for r in refs if r not in evidence_map]
            operational = [
                r
                for r in valid
                if evidence_map[r].get("source") in {"log", "metric", "change"}
            ]
            ratio = len(valid) / len(refs) if refs else 0.0
            score = 0.5 * ratio + 0.5 * (1.0 if operational else 0.2)
            label_ok, label_reason = _label_support(hyp, evidence_map, valid)
            if not label_ok:
                score -= 0.25

            verdict = llm_verdicts.get(hyp.get("id"))
            llm_supported = None
            if verdict is not None:
                llm_supported = bool(verdict.get("supported"))
                if not llm_supported:
                    score -= 0.25

            supported = bool(valid) and not invalid and bool(operational) and label_ok
            if verdict is not None and not llm_supported:
                supported = False
            score = max(0.0, min(1.0, score))
            if not supported:
                hyp["confidence"] = round(min(float(hyp.get("confidence", 0.5)), 0.55), 3)
                hyp["caveats"] = list(
                    dict.fromkeys(
                        list(hyp.get("caveats") or [])
                        + [f"证据支撑不足：{label_reason}。"]
                    )
                )
            hyp["supported"] = supported
            hyp["faithfulness"] = round(score, 3)
            checks.append(
                {
                    "hypothesis_id": hyp.get("id"),
                    "supported": supported,
                    "faithfulness": round(score, 3),
                    "valid_refs": valid,
                    "invalid_refs": invalid,
                    "operational_refs": operational,
                    "label_check": label_reason,
                    "llm_supported": llm_supported,
                    "llm_reason": (verdict or {}).get("reason"),
                    "reason": (
                        f"引用 {len(refs)} 条证据，有效 {len(valid)} 条，其中运维证据 {len(operational)} 条；"
                        f"{label_reason}。"
                    ),
                }
            )

        overall = round(
            sum(c["faithfulness"] for c in checks) / len(checks), 3
        ) if checks else 0.0
        supported_count = sum(1 for c in checks if c["supported"])

        retryable_sources = [
            d.get("source")
            for d in coverage.get("missing_details") or []
            if d.get("retryable")
        ]
        retryable_sources = list(dict.fromkeys(retryable_sources))
        can_retry = bool(retryable_sources) and retry_count < settings.max_retries
        needs_retry = can_retry and (overall < 0.72 or supported_count == 0)

        next_retry_count = retry_count + 1 if needs_retry else retry_count
        critique = {
            "faithfulness": overall,
            "supported_count": supported_count,
            "hypothesis_count": len(checks),
            "checks": checks,
            "needs_retry": needs_retry,
            "retry_focus": retryable_sources if needs_retry else [],
            "notes": (
                "存在可重试的缺失维度，将回退补充取证。"
                if needs_retry
                else ("证据支撑不足，但缺失维度不可重试，直接输出带不确定性说明的结论。" if overall < 0.72 else "证据链完整度可接受。")
            ),
        }

        needs_approval = False
        approval_request = None
        top = hypotheses[0] if hypotheses else None
        if (
            top
            and top.get("supported")
            and top.get("action_required") not in {None, "none"}
            and top.get("proposed_action")
            and float(top.get("confidence", 0)) >= 0.6
        ):
            needs_approval = True
            action = top["proposed_action"]
            refs = [
                evidence_map[r]
                for r in top.get("evidence_refs", [])
                if r in evidence_map
            ][:5]
            approval_request = {
                "type": "approval_required",
                "hypothesis_id": top.get("id"),
                "title": top.get("title"),
                "root_cause": top.get("root_cause"),
                "confidence": top.get("confidence"),
                "action": action,
                "evidence": [
                    {
                        "id": e.get("id"),
                        "source": e.get("source"),
                        "title": e.get("title"),
                        "timestamp": e.get("timestamp"),
                        "excerpt": (e.get("raw_excerpt") or "")[:300],
                        "confidence": e.get("confidence"),
                    }
                    for e in refs
                ],
                "risks": [
                    f"该操作属于 {action.get('risk')} 风险变更，可能影响线上流量。",
                    "执行前必须核对实际命令、目标服务与 diff。",
                    "恢复执行使用幂等键，重复提交不会重复执行。",
                ],
                "expires_at": to_iso(
                    datetime.now(timezone.utc) + timedelta(minutes=settings.approval_ttl_minutes)
                ),
                "instructions": "请确认命令与 diff 后批准或拒绝；拒绝时请填写原因，将写入审计与长期记忆。",
            }

        updates = {
            "hypotheses": hypotheses,
            "critique": critique,
            "retry_count": next_retry_count,
            "retry_focus": retryable_sources if needs_retry else [],
            "needs_approval": needs_approval,
            "approval_request": approval_request,
        }
        if critic_usage:
            updates["llm_usage"] = [critic_usage]
        return updates