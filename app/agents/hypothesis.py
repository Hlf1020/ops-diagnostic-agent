"""假设生成：规则引擎（离线、可复现）+ LLM（有 key 时）。

规则引擎不是“玩具”：它把时间邻近性、指标变化、日志模式、变更风险、知识库
与历史案例组合成可解释分数。LLM 只在这个证据集上做归纳，输出仍要经过
evidence_refs 校验与 faithfulness critic。
"""

from __future__ import annotations

import logging

from app.agents.actions import build_action
from app.config import Settings
from app.core.llm import LLMClient, LLMError
from app.core.timeutil import minutes_between
from app.core.tracing import span
from app.models import HypothesisListOut

logger = logging.getLogger(__name__)

ROOT_CAUSE_LABELS = {
    "config_change": "配置变更",
    "bad_deploy": "异常部署",
    "traffic_surge": "流量突增",
    "dependency_failure": "依赖故障",
    "resource_exhaustion": "资源耗尽",
    "network_issue": "网络/DNS 异常",
    "unknown": "未确定",
}

SPECIFIC_LABELS = set(ROOT_CAUSE_LABELS) - {"unknown"}

LLM_SYSTEM_PROMPT = """你是资深 SRE 根因分析员。你只能基于用户提供的证据列表推理，禁止编造证据、时间、数值或变更 ID。
要求：
1. 给出 1-4 个候选根因，按置信度从高到低排序；
2. 每条假设的 evidence_refs 必须引用证据列表中真实存在的 id，至少一条运维证据（log/metric/change），不能只引用知识库；
3. 根因只能是：config_change, bad_deploy, traffic_surge, dependency_failure, resource_exhaustion, network_issue, unknown；
4. 只有在存在对应变更证据时才建议 rollback；证据不足时降低 confidence 并写入 caveats；
5. action_required 只能是 none, rollback, restart, scale, cleanup；
6. 只返回 JSON：
{"hypotheses":[{"title":"...","reasoning":"...","root_cause":"...","confidence":0.0,"evidence_refs":["ev-..."],"action_required":"none","caveats":["..."]}],"notes":"..."}
所有文本用中文。"""


def _clamp(value: float, low: float = 0.2, high: float = 0.95) -> float:
    return round(max(low, min(high, value)), 3)


def _num(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:g}%"
    if unit == "count":
        return f"{value:g} 次"
    return f"{value:g}{unit}"


def _evidence_by_source(evidence: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for item in evidence:
        out.setdefault(item.get("source", "other"), []).append(item)
    return out


def _anomaly_start(evidence: list[dict], window: dict) -> str | None:
    timestamps = [
        e.get("timestamp")
        for e in evidence
        if e.get("timestamp") and e.get("source") in {"metric", "log"}
    ]
    if timestamps:
        return min(timestamps)
    return window.get("anchor") or window.get("start")


def _metric_summary(metrics: list[dict], limit: int = 2) -> str:
    ranked = sorted(metrics, key=lambda e: (e.get("metadata") or {}).get("change_pct", 0), reverse=True)
    parts = []
    for item in ranked[:limit]:
        meta = item.get("metadata") or {}
        parts.append(
            f"{meta.get('metric')} {_num(float(meta.get('baseline', 0)), meta.get('unit', ''))}"
            f"->{_num(float(meta.get('peak', 0)), meta.get('unit', ''))}"
        )
    return "、".join(parts)


def _log_categories(logs: list[dict]) -> set[str]:
    return {((e.get("metadata") or {}).get("category")) for e in logs}


def _coverage_caveats(coverage: dict) -> list[str]:
    details = coverage.get("missing_details") or []
    if not details:
        return []
    parts = [f"{d.get('source')}（{d.get('reason')}）" for d in details]
    return [f"信息缺失：{'；'.join(parts)}。当前假设仅基于已获取的证据，未对缺失维度做推断。"]


class _CandidateBook:
    def __init__(self):
        self.items: dict[str, dict] = {}

    def add(
        self,
        label: str,
        *,
        title: str,
        reasoning: str,
        score: float,
        refs: list[str],
        caveats: list[str] | None = None,
    ) -> None:
        if label not in SPECIFIC_LABELS:
            return
        current = self.items.get(label)
        if current is None:
            self.items[label] = {
                "root_cause": label,
                "title": title,
                "reasoning": reasoning,
                "score": _clamp(score),
                "refs": list(dict.fromkeys(refs)),
                "caveats": list(caveats or []),
            }
            return
        current["refs"] = list(dict.fromkeys(current["refs"] + refs))
        current["caveats"] = list(dict.fromkeys(current["caveats"] + (caveats or [])))
        if score > current["score"]:
            current["score"] = _clamp(score)
            current["title"] = title
            current["reasoning"] = reasoning

    def boost(self, label: str, refs: list[str], amount: float, note: str = "") -> None:
        current = self.items.get(label)
        if not current:
            return
        current["refs"] = list(dict.fromkeys(current["refs"] + refs))
        current["score"] = _clamp(current["score"] + amount)
        if note and note not in current["reasoning"]:
            current["reasoning"] += f" {note}"

    def all(self) -> list[dict]:
        return sorted(self.items.values(), key=lambda c: c["score"], reverse=True)


def _derive_action(label: str, evidence: list[dict]) -> str:
    changes = [e for e in evidence if e.get("source") == "change"]
    change_types = {(e.get("metadata") or {}).get("change_type") for e in changes}
    if label == "config_change":
        return "rollback" if "config" in change_types else "none"
    if label == "bad_deploy":
        return "rollback" if "deploy" in change_types else "none"
    if label == "traffic_surge":
        return "scale"
    if label == "resource_exhaustion":
        categories = _log_categories([e for e in evidence if e.get("source") == "log"])
        has_disk = "disk" in categories or any(
            (e.get("metadata") or {}).get("metric") in {"disk_usage_pct", "io_wait_pct"}
            for e in evidence
            if e.get("source") == "metric"
        )
        return "cleanup" if has_disk else "scale"
    return "none"


def _build_rules(state: dict) -> list[dict]:
    evidence = state.get("evidence") or []
    coverage = state.get("coverage") or {}
    intent = state.get("intent") or {}
    service = intent.get("service")
    window = intent.get("time_window") or {}
    by_source = _evidence_by_source(evidence)
    metrics = by_source.get("metric", [])
    logs = by_source.get("log", [])
    changes = by_source.get("change", [])
    kb = by_source.get("knowledge", [])
    memories = by_source.get("memory", [])
    categories = _log_categories(logs)
    anomaly_start = _anomaly_start(evidence, window)
    book = _CandidateBook()

    # ---- 1) 变更时间线 ----
    for change_ev in changes:
        meta = change_ev.get("metadata") or {}
        ts = change_ev.get("timestamp")
        if not ts or not anomaly_start:
            continue
        try:
            delta = minutes_between(ts, anomaly_start)
        except Exception:
            continue
        if delta < -2 or delta > 30:
            continue  # 异常开始后才发生（缓解措施）或距异常过远的变更
        change_type = meta.get("change_type")
        if change_type not in {"config", "deploy"}:
            continue
        label = "config_change" if change_type == "config" else "bad_deploy"
        base = 0.55
        if delta <= 2:
            base += 0.20
        elif delta <= 5:
            base += 0.17
        elif delta <= 10:
            base += 0.13
        elif delta <= 20:
            base += 0.08
        else:
            base += 0.04
        risk = meta.get("risk", "low")
        base += {"high": 0.10, "medium": 0.05}.get(risk, 0.0)
        if metrics:
            base += 0.06
        if label == "config_change" and any(
            (m.get("metadata") or {}).get("metric") in {"gc_pause_p99_ms", "db_connection_wait_ms"}
            for m in metrics
        ):
            base += 0.04
        if label == "bad_deploy" and any(
            (m.get("metadata") or {}).get("metric") in {"memory_usage_pct", "pod_restarts"}
            for m in metrics
        ):
            base += 0.04
        if label == "config_change" and categories & {"connection_pool", "gc", "dependency", "deploy"}:
            base += 0.06
        if label == "bad_deploy" and categories & {"oom", "error", "dependency"}:
            base += 0.08

        diff = (meta.get("diff") or "").replace("\n", " | ")
        kind_cn = "配置" if change_type == "config" else "部署"
        metric_text = _metric_summary(metrics) or "相关指标出现异常"
        primary = "首要嫌疑" if base >= 0.85 else "候选原因之一"
        if label == "config_change":
            log_support = categories & {"connection_pool", "gc", "dependency", "deploy"}
            support_text = (
                "连接池/GC 相关日志与变更时间线一致"
                if log_support
                else "未观察到连接池/GC 的直接日志证据，该假设主要来自时间邻近性"
            )
            title = f"{kind_cn}变更 {meta.get('change_id')} 与指标异常时间吻合"
            reasoning = (
                f"{ts} 由 {meta.get('operator')} 执行配置变更「{meta.get('summary')}」（风险 {risk}）；"
                f"约 {delta:.0f} 分钟后开始出现异常：{metric_text}。"
                f"变更 diff：{diff[:240]}。{support_text}，配置错误是{primary}。"
            )
        else:
            log_support = categories & {"oom", "error", "dependency"}
            support_text = (
                "日志中的 OOM/异常堆栈与发布内容吻合"
                if log_support
                else "未观察到与新版本直接相关的 OOM/异常堆栈，该假设主要来自时间邻近性"
            )
            title = f"{kind_cn} {meta.get('change_id')} 后出现异常模式"
            reasoning = (
                f"{ts} 发布「{meta.get('summary')}」（风险 {risk}），约 {delta:.0f} 分钟后指标恶化：{metric_text}。"
                f"{support_text}，diff：{diff[:240]}。新版本代码缺陷是{primary}。"
            )
        caveats = ["时间相关性不等于因果，回滚前需结合 diff 与现场日志确认。"]
        if not log_support:
            caveats.append("日志未提供与该变更类型匹配的直接证据。")
        book.add(
            label,
            title=title,
            reasoning=reasoning,
            score=base,
            refs=[change_ev.get("id")],
            caveats=caveats,
        )

    # ---- 2) 指标模式 ----
    metric_map = {(m.get("metadata") or {}).get("metric"): m for m in metrics}
    qps = metric_map.get("request_rate_qps")
    if qps and float((qps.get("metadata") or {}).get("change_pct", 0)) >= 100:
        book.add(
            "traffic_surge",
            title="请求量突增触发 CPU 限流与延迟上升",
            reasoning=(
                f"QPS 异常：{_metric_summary([qps])}，且时间窗口内没有匹配的发布变更。"
                "流量突增是更直接的解释，扩容/限流是优先处置方向。"
            ),
            score=0.87,
            refs=[qps.get("id")],
        )

    dep = metric_map.get("dependency_error_rate_pct")
    if dep and float((dep.get("metadata") or {}).get("change_pct", 0)) >= 50:
        error_ev = metric_map.get("error_rate_pct")
        refs = [dep.get("id")] + ([error_ev.get("id")] if error_ev else [])
        book.add(
            "dependency_failure",
            title="下游依赖错误率飙升，调用方被拖垮",
            reasoning=(
                f"依赖错误率指标异常：{_metric_summary([dep])}；调用方错误率同步上升。"
                "日志中的 connection refused / 熔断器 OPEN 指向依赖侧故障，而非自身代码发布。"
            ),
            score=0.86,
            refs=refs,
        )

    disk = metric_map.get("disk_usage_pct")
    io_wait = metric_map.get("io_wait_pct")
    if disk and (
        float((disk.get("metadata") or {}).get("peak", 0)) >= 95
        or float((disk.get("metadata") or {}).get("change_pct", 0)) >= 15
    ):
        refs = [disk.get("id")] + ([io_wait.get("id")] if io_wait else [])
        book.add(
            "resource_exhaustion",
            title="磁盘空间耗尽导致写入失败与延迟上升",
            reasoning=(
                f"磁盘指标异常：{_metric_summary([disk])}；IO wait 同步上升。"
                "日志中的 No space left on device / OSError 28 与指标一致。"
            ),
            score=0.88,
            refs=refs,
        )

    dns = metric_map.get("dns_error_rate_pct")
    retransmit = metric_map.get("network_retransmit_pct")
    if (dns and float((dns.get("metadata") or {}).get("change_pct", 0)) >= 50) or (
        retransmit and float((retransmit.get("metadata") or {}).get("change_pct", 0)) >= 100
    ):
        refs = [e.get("id") for e in [dns, retransmit] if e]
        book.add(
            "network_issue",
            title="DNS 解析与网络重传异常",
            reasoning=(
                f"网络指标异常：{_metric_summary([e for e in [dns, retransmit] if e])}；"
                "日志中的 no such host / connection reset 指向 DNS 或节点网络问题。"
            ),
            score=0.89,
            refs=refs,
        )

    memory_ev = metric_map.get("memory_usage_pct")
    restart_ev = metric_map.get("pod_restarts")
    if restart_ev and float((restart_ev.get("metadata") or {}).get("peak", 0)) > 0:
        refs = [restart_ev.get("id")] + ([memory_ev.get("id")] if memory_ev else [])
        book.add(
            "resource_exhaustion",
            title="内存耗尽触发 OOMKilled 与重启",
            reasoning=(
                f"容器重启次数异常：{_metric_summary([restart_ev])}，内存使用率同步接近上限。"
                "若同窗口存在发布变更，应优先考虑新版本内存泄漏。"
            ),
            score=0.68,
            refs=refs,
        )
    cpu = metric_map.get("cpu_utilization_pct")
    throttled = metric_map.get("cpu_throttled_pct")
    if (cpu and float((cpu.get("metadata") or {}).get("peak", 0)) >= 85) or (
        throttled and float((throttled.get("metadata") or {}).get("peak", 0)) >= 10
    ):
        refs = [e.get("id") for e in [cpu, throttled] if e]
        book.add(
            "resource_exhaustion",
            title="CPU 饱和/限流导致处理能力下降",
            reasoning=(
                f"CPU 相关指标异常：{_metric_summary([e for e in [cpu, throttled] if e])}。"
                "若 QPS 同时翻倍，则更可能是流量突增；否则检查 CPU limit 与实例规格。"
            ),
            score=0.64,
            refs=refs,
        )

    # ---- 3) 日志模式补强（在指标/变更不足时的兜底） ----
    log_refs_by_category: dict[str, list[str]] = {}
    for ev in logs:
        cat = (ev.get("metadata") or {}).get("category")
        log_refs_by_category.setdefault(cat, []).append(ev.get("id"))

    if categories & {"oom", "error"}:
        refs = log_refs_by_category.get("oom", []) + log_refs_by_category.get("error", [])
        if "bad_deploy" in book.items:
            book.boost("bad_deploy", refs, 0.02, "日志中的 OOM/异常堆栈进一步支持该假设。")
        else:
            book.add(
                "bad_deploy",
                title="日志显示 OOM/空指针，疑似新版本缺陷",
                reasoning="日志窗口内出现 OOMKilled 或未处理异常堆栈；未取到对应发布记录，需人工核对发布时间。",
                score=0.69,
                refs=refs,
                caveats=["缺少发布变更记录，bad_deploy 假设无法完全闭环。"],
            )
    if categories & {"connection_pool", "gc"}:
        refs = log_refs_by_category.get("connection_pool", []) + log_refs_by_category.get("gc", [])
        if "config_change" in book.items:
            book.boost("config_change", refs, 0.02, "连接池/GC 日志支持配置错误假设。")
        else:
            book.add(
                "config_change",
                title="连接池/GC 日志异常，疑似配置问题",
                reasoning="日志显示连接池等待或 GC 停顿异常；未取到对应配置变更，建议核对连接池与 JVM 参数。",
                score=0.66,
                refs=refs,
                caveats=["缺少配置变更记录，config_change 假设无法完全闭环。"],
            )
    if "disk" in categories:
        refs = log_refs_by_category.get("disk", [])
        if "resource_exhaustion" in book.items:
            book.boost("resource_exhaustion", refs, 0.02, "磁盘写入失败日志支持资源耗尽假设。")
        else:
            book.add(
                "resource_exhaustion",
                title="磁盘写入失败日志",
                reasoning="日志出现 No space left on device / OSError 28，指向磁盘空间耗尽。",
                score=0.76,
                refs=refs,
            )
    if categories & {"dns", "network"}:
        refs = log_refs_by_category.get("dns", []) + log_refs_by_category.get("network", [])
        if "network_issue" in book.items:
            book.boost("network_issue", refs, 0.02, "DNS/网络日志支持该假设。")
        else:
            book.add(
                "network_issue",
                title="DNS/网络异常日志",
                reasoning="日志出现 DNS 解析失败或 TCP 重传，指向网络层问题。",
                score=0.77,
                refs=refs,
            )
    if "dependency" in categories:
        refs = log_refs_by_category.get("dependency", [])
        if "dependency_failure" in book.items:
            book.boost("dependency_failure", refs, 0.02, "依赖连接失败/熔断日志支持该假设。")
        else:
            book.add(
                "dependency_failure",
                title="依赖连接失败或熔断",
                reasoning="日志出现 connection refused / circuit breaker OPEN，指向下游依赖故障。",
                score=0.76,
                refs=refs,
            )

    # ---- 4) 知识库与历史案例 ----
    for ev in kb:
        label = (ev.get("metadata") or {}).get("root_cause")
        if label in book.items:
            book.boost(label, [ev.get("id")], 0.03, f"知识库文档「{ev.get('title')}」描述了相同模式。")
        elif label in SPECIFIC_LABELS and len(evidence) >= 2:
            book.add(
                label,
                title=f"知识库匹配：{ev.get('title')}",
                reasoning=f"混合检索命中知识库文档「{ev.get('title')}」，其描述的模式与当前症状相近，但需要运维证据进一步验证。",
                score=0.52,
                refs=[ev.get("id")],
                caveats=["该假设主要来自知识库，缺少充分的现场证据。"],
            )

    for ev in memories:
        label = (ev.get("metadata") or {}).get("root_cause")
        if label in book.items:
            similarity = float((ev.get("metadata") or {}).get("similarity", 0) or 0)
            book.boost(
                label,
                [ev.get("id")],
                0.02,
                f"长期记忆中存在相似历史案例（相似度 {similarity:.2f}）。",
            )

    candidates = book.all()
    global_caveats = _coverage_caveats(coverage)
    if not candidates:
        refs = [e.get("id") for e in sorted(evidence, key=lambda x: x.get("confidence", 0), reverse=True)[:3]]
        missing = "、".join(coverage.get("missing_sources") or []) or "无"
        return [
            {
                "title": "证据不足，无法确认根因",
                "reasoning": (
                    f"已收集 {len(evidence)} 条证据，但未形成可支撑的根因模式；缺失维度：{missing}。"
                    "建议补充日志/指标/变更数据后重试，或升级给对应领域专家。"
                ),
                "root_cause": "unknown",
                "confidence": 0.3,
                "evidence_refs": refs,
                "action_required": "none",
                "caveats": global_caveats + ["未达到可执行结论的置信度阈值。"],
                "generator": "rules",
            }
        ]

    results = []
    for idx, cand in enumerate(candidates[:4], start=1):
        label = cand["root_cause"]
        action_kind = _derive_action(label, evidence)
        proposed = build_action(
            action_kind,
            service=service,
            evidence=evidence,
            session_id=state.get("session_id", "session"),
            root_cause=label,
            anomaly_start=anomaly_start,
        )
        if proposed is None:
            action_kind = "none"
        results.append(
            {
                "id": f"hyp-{idx}",
                "title": cand["title"],
                "reasoning": cand["reasoning"],
                "root_cause": label,
                "confidence": cand["score"],
                "evidence_refs": cand["refs"],
                "action_required": action_kind,
                "proposed_action": proposed,
                "caveats": list(dict.fromkeys(cand["caveats"] + global_caveats)),
                "generator": "rules",
            }
        )
    return results


def _compact_evidence(evidence: list[dict]) -> str:
    lines = []
    for item in evidence:
        excerpt = (item.get("raw_excerpt") or "").replace("\n", " ")[:280]
        lines.append(
            f"[{item.get('id')}] source={item.get('source')} ts={item.get('timestamp')} "
            f"confidence={item.get('confidence')} title={item.get('title')} | {excerpt}"
        )
    return "\n".join(lines)


def _llm_prompt(state: dict) -> tuple[str, str]:
    intent = state.get("intent") or {}
    window = intent.get("time_window") or {}
    coverage = state.get("coverage") or {}
    historical = state.get("historical_cases") or []
    history_text = "\n".join(
        f"- {h.get('symptom')} -> {h.get('root_cause')}：{h.get('conclusion')}"
        for h in historical[:3]
    ) or "无"
    user = (
        f"用户描述：{state.get('query', '')}\n"
        f"服务：{intent.get('service')}\n"
        f"意图：{intent.get('intent')} / {intent.get('intent_label')}\n"
        f"时间窗口：{window.get('start')} ~ {window.get('end')}（UTC+8）\n"
        f"缺失维度：{coverage.get('missing_sources')}（详情：{coverage.get('missing_details')}）\n"
        f"历史相似案例：\n{history_text}\n\n"
        f"证据列表（只能引用这些 id）：\n{_compact_evidence(state.get('evidence') or [])}"
    )
    return LLM_SYSTEM_PROMPT, user


def _finalize_llm(
    hypotheses: list[dict],
    *,
    state: dict,
    evidence_ids: set[str],
    coverage: dict,
    anomaly_start: str | None,
) -> list[dict]:
    service = (state.get("intent") or {}).get("service")
    evidence = state.get("evidence") or []
    operational_ids = {
        e.get("id") for e in evidence if e.get("source") in {"log", "metric", "change"}
    }
    global_caveats = _coverage_caveats(coverage)
    results = []
    for idx, item in enumerate(hypotheses[:4], start=1):
        refs = [ref for ref in item.get("evidence_refs", []) if ref in evidence_ids]
        if not refs:
            continue
        caveats = list(item.get("caveats") or []) + global_caveats
        confidence = float(item.get("confidence", 0.5))
        if not set(refs) & operational_ids:
            confidence = min(confidence, 0.45)
            caveats.append("该假设只引用了知识库/记忆，缺少现场运维证据。")
        action_kind = item.get("action_required", "none")
        proposed = build_action(
            action_kind,
            service=service,
            evidence=evidence,
            session_id=state.get("session_id", "session"),
            root_cause=item.get("root_cause", ""),
            anomaly_start=anomaly_start,
        )
        if proposed is None:
            action_kind = "none"
        results.append(
            {
                "id": f"hyp-{idx}",
                "title": item.get("title"),
                "reasoning": item.get("reasoning", ""),
                "root_cause": item.get("root_cause", "unknown"),
                "confidence": round(max(0.0, min(0.95, confidence)), 3),
                "evidence_refs": refs,
                "action_required": action_kind,
                "proposed_action": proposed,
                "caveats": list(dict.fromkeys(caveats)),
                "generator": "llm",
            }
        )
    return results


def generate_hypotheses(
    state: dict,
    *,
    llm: LLMClient,
    settings: Settings,
    trace_id: str | None,
) -> dict:
    with span("node.hypothesis_generator", trace_id=trace_id, attrs={"app.evidence_count": len(state.get("evidence") or [])}):
        if llm.enabled:
            try:
                system, user = _llm_prompt(state)
                result = llm.complete_json(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    HypothesisListOut,
                    trace_id=trace_id,
                    node="hypothesis_generator",
                )
                ids = {e.get("id") for e in state.get("evidence") or []}
                evidence = state.get("evidence") or []
                window = (state.get("intent") or {}).get("time_window") or {}
                finalized = _finalize_llm(
                    [h.model_dump() for h in result.parsed.hypotheses],
                    state=state,
                    evidence_ids=ids,
                    coverage=state.get("coverage") or {},
                    anomaly_start=_anomaly_start(evidence, window),
                )
                if finalized:
                    return {
                        "hypotheses": finalized,
                        "hypothesis_notes": result.parsed.notes,
                        "llm_usage": [
                            {**result.usage.as_dict(settings), "node": "hypothesis_generator"}
                        ],
                    }
                logger.warning("LLM hypotheses had no valid evidence refs, falling back to rules")
            except LLMError as exc:
                logger.warning("LLM hypothesis generation failed, fallback to rules: %s", exc)
            except Exception:
                logger.exception("unexpected LLM hypothesis error, fallback to rules")

        return {
            "hypotheses": _build_rules(state),
            "hypothesis_notes": "rules-engine",
        }