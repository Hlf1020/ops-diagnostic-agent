"""最终报告生成。使用确定性模板，避免 critic 之后再由 LLM 引入新事实。"""

from __future__ import annotations

from app.agents.hypothesis import ROOT_CAUSE_LABELS
from app.core.text import safe_json_dumps


def _severity(intent: dict, hypotheses: list[dict], supported: bool) -> str:
    if not supported:
        return "P3"
    top = hypotheses[0] if hypotheses else {}
    confidence = float(top.get("confidence", 0))
    action = top.get("action_required")
    if confidence >= 0.7 and (
        intent.get("intent") in {"error_rate", "latency"} or action in {"rollback", "restart"}
    ):
        return "P1"
    if confidence >= 0.55:
        return "P2"
    return "P3"


def _recommended_actions(state: dict, coverage: dict) -> list[str]:
    hypotheses = state.get("hypotheses") or []
    action_result = state.get("action_result")
    approval_request = state.get("approval_request")
    top = hypotheses[0] if hypotheses else None
    actions: list[str] = []

    if action_result:
        status = action_result.get("status")
        action = (approval_request or {}).get("action") or {}
        if status in {"simulated_success", "success", "replayed"}:
            actions.append(f"已执行动作：{action.get('command', '')}")
            actions.append("执行后持续观察 5-10 分钟核心指标，确认是否恢复。")
        elif status == "rejected":
            comment = action_result.get("comment") or "未填写原因"
            actions.append(f"人工拒绝执行高危动作（原因：{comment}）。建议保留现场证据并升级复核。")
        else:
            actions.append(f"动作执行失败：{action_result.get('error', 'unknown error')}。请人工介入。")
    elif approval_request:
        action = approval_request.get("action") or {}
        actions.append(f"等待人工审批后执行：{action.get('command', '')}")
    elif top and top.get("action_required") not in {None, "none"} and top.get("proposed_action"):
        actions.append(f"建议动作（需审批）：{top['proposed_action'].get('command')}")
    else:
        actions.append("当前没有可自动执行的高风险动作；请按候选根因补充验证。")

    metrics = [
        (e.get("metadata") or {}).get("metric")
        for e in state.get("evidence") or []
        if e.get("source") == "metric"
    ]
    metrics = list(dict.fromkeys([m for m in metrics if m]))[:4]
    if metrics:
        actions.append(f"观察指标：{', '.join(metrics)}")
    if coverage.get("missing_sources"):
        actions.append(f"补充缺失维度的数据：{', '.join(coverage['missing_sources'])}。")
    return actions


def build_final_report(state: dict) -> dict:
    intent = state.get("intent") or {}
    hypotheses = state.get("hypotheses") or []
    critique = state.get("critique") or {}
    coverage = state.get("coverage") or {}
    approval_request = state.get("approval_request")
    approval = state.get("approval")
    action_result = state.get("action_result")
    evidence = state.get("evidence") or []
    evidence_map = {e.get("id"): e for e in evidence}
    historical = state.get("historical_cases") or []
    supported = bool(critique.get("supported_count"))
    top = hypotheses[0] if hypotheses else None
    service = intent.get("service") or "unknown"

    if supported and top:
        summary = (
            f"首要根因假设：{top.get('title')}（{ROOT_CAUSE_LABELS.get(top.get('root_cause'), top.get('root_cause'))}，"
            f"置信度 {top.get('confidence')}，证据支撑度 {top.get('faithfulness')}）。{top.get('reasoning', '')[:220]}"
        )
    else:
        summary = (
            "当前证据不足以确认根因。以下候选方向按现有证据的支撑度排序，"
            "请勿据此直接执行高风险变更。"
        )

    if action_result:
        status = {
            "simulated_success": "action_executed",
            "success": "action_executed",
            "replayed": "action_executed",
            "rejected": "action_rejected",
        }.get(action_result.get("status"), "action_failed")
    elif not supported:
        status = "completed_low_confidence"
    else:
        status = "completed"

    root_causes = []
    for idx, hyp in enumerate(hypotheses[:3], start=1):
        refs = [
            {
                "id": ref,
                "source": evidence_map.get(ref, {}).get("source"),
                "title": evidence_map.get(ref, {}).get("title"),
                "timestamp": evidence_map.get(ref, {}).get("timestamp"),
                "confidence": evidence_map.get(ref, {}).get("confidence"),
            }
            for ref in hyp.get("evidence_refs", [])
        ]
        root_causes.append(
            {
                "rank": idx,
                "id": hyp.get("id"),
                "title": hyp.get("title"),
                "root_cause": hyp.get("root_cause"),
                "root_cause_label": ROOT_CAUSE_LABELS.get(hyp.get("root_cause"), hyp.get("root_cause")),
                "confidence": hyp.get("confidence"),
                "faithfulness": hyp.get("faithfulness"),
                "supported": hyp.get("supported"),
                "reasoning": hyp.get("reasoning"),
                "evidence_refs": refs,
                "action_required": hyp.get("action_required"),
                "proposed_action": hyp.get("proposed_action"),
                "caveats": hyp.get("caveats", []),
            }
        )

    missing = [
        f"{d.get('source')}：{d.get('reason')}" for d in coverage.get("missing_details") or []
    ]
    recommended = _recommended_actions(state, coverage)
    severity = _severity(intent, hypotheses, supported)

    report = {
        "title": f"{service} 诊断报告",
        "service": service,
        "intent": intent.get("intent"),
        "intent_label": intent.get("intent_label"),
        "window": intent.get("time_window"),
        "severity": severity,
        "status": status,
        "summary": summary,
        "confidence": top.get("confidence") if top else 0.0,
        "faithfulness": critique.get("faithfulness", 0.0),
        "root_causes": root_causes,
        "evidence": [
            {
                "id": e.get("id"),
                "source": e.get("source"),
                "title": e.get("title"),
                "timestamp": e.get("timestamp"),
                "excerpt": (e.get("raw_excerpt") or "")[:400],
                "confidence": e.get("confidence"),
                "tool_trace_id": e.get("tool_trace_id"),
                "injection_flags": e.get("injection_flags", []),
            }
            for e in evidence
        ],
        "timeline": state.get("timeline") or [],
        "missing_information": missing,
        "recommended_actions": recommended,
        "approval": {
            "required": approval_request is not None,
            "request": approval_request,
            "decision": approval,
            "result": action_result,
        },
        "historical_cases": historical,
        "critique": critique,
        "disclaimer": (
            "本报告是基于工具证据的推断，不等同于已确认根因；"
            "涉及回滚、扩容、清理等生产操作必须经人工审批。"
        ),
    }
    report["markdown"] = render_markdown(report)
    report["markdown_chars"] = len(report["markdown"])
    return report


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    window = report.get("window") or {}
    lines.append(f"# {report.get('title')}")
    lines.append("")
    lines.append(f"- 状态：**{report.get('status')}**")
    lines.append(f"- 严重级别：**{report.get('severity')}**")
    lines.append(f"- 服务：`{report.get('service')}`")
    lines.append(f"- 时间窗口：{window.get('start')} ~ {window.get('end')}（{window.get('label', '')}）")
    lines.append(f"- 总体置信度：{report.get('confidence')}；证据支撑度：{report.get('faithfulness')}")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(report.get("summary", ""))
    lines.append("")
    lines.append("## 候选根因 Top 3")
    lines.append("")
    for item in report.get("root_causes") or []:
        flag = "✅" if item.get("supported") else "⚠️"
        lines.append(
            f"{item.get('rank')}. {flag} **{item.get('title')}** "
            f"（{item.get('root_cause_label')}，置信度 {item.get('confidence')}，支撑度 {item.get('faithfulness')}）"
        )
        lines.append(f"   - 推理：{item.get('reasoning')}")
        if item.get("evidence_refs"):
            refs = "、".join(
                f"`{r.get('id')}`({r.get('source')})" for r in item["evidence_refs"]
            )
            lines.append(f"   - 证据：{refs}")
        if item.get("caveats"):
            lines.append(f"   - 不确定性：{'；'.join(item['caveats'])}")
        lines.append("")
    lines.append("## 证据时间线")
    lines.append("")
    for item in (report.get("timeline") or [])[:15]:
        lines.append(
            f"- `{item.get('time')}` [{item.get('source')}] {item.get('title')} "
            f"（置信度 {item.get('confidence')}，`{item.get('evidence_id')}`）"
        )
    if not report.get("timeline"):
        lines.append("- 无可对齐的时间序列证据。")
    lines.append("")
    if report.get("missing_information"):
        lines.append("## 信息缺失")
        lines.append("")
        for item in report["missing_information"]:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("## 建议动作")
    lines.append("")
    for item in report.get("recommended_actions") or []:
        lines.append(f"- {item}")
    lines.append("")
    approval = report.get("approval") or {}
    if approval.get("required"):
        action = (approval.get("request") or {}).get("action") or {}
        lines.append("## 人工确认")
        lines.append("")
        lines.append(f"- 待执行命令：`{action.get('command')}`")
        lines.append(f"- 变更 diff：\n```diff\n{action.get('diff', '')}\n```")
        lines.append(f"- 幂等键：`{action.get('idempotency_key')}`")
        decision = approval.get("decision")
        if decision:
            lines.append(f"- 审批结果：{'批准' if decision.get('approved') else '拒绝'}，操作人 {decision.get('approver_id')}，意见：{decision.get('comment', '')}")
        lines.append("")
    if report.get("historical_cases"):
        lines.append("## 历史参考")
        lines.append("")
        for case in report["historical_cases"][:3]:
            lines.append(
                f"- {case.get('created_at')} {case.get('symptom')} -> {case.get('root_cause')}"
                f"（相似度 {case.get('similarity')}）"
            )
        lines.append("")
    lines.append("---")
    lines.append(report.get("disclaimer", ""))
    return "\n".join(lines)


def report_for_prompt(report: dict) -> str:
    return safe_json_dumps(
        {
            "title": report.get("title"),
            "root_causes": report.get("root_causes"),
            "missing_information": report.get("missing_information"),
        },
        max_chars=3000,
    )