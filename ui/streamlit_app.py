"""Streamlit 演示台：诊断、证据面板、人工审批、Trace 时间线。

启动：streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_QUERY = (
    "支付服务 payment-service 从 14:00 开始 P99 延迟从 200ms 涨到 1.8s，"
    "帮我看看可能原因。"
)

st.set_page_config(page_title="Ops Diagnostic Agent", page_icon="🛠️", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.small-muted {color:#6b7280; font-size:0.82rem;}
.badge {display:inline-block; padding:2px 8px; border-radius:10px; font-size:0.75rem; margin-right:6px;}
.badge-ok {background:#dcfce7; color:#166534;}
.badge-warn {background:#fef3c7; color:#92400e;}
.badge-bad {background:#fee2e2; color:#991b1b;}
.badge-info {background:#dbeafe; color:#1e40af;}
.trace-row {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:0.8rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- API helpers
def api_get(path: str) -> Any:
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(f"{API_BASE}{path}")
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        return {"_error": str(exc)}


def api_post(path: str, payload: dict, timeout: float = 180.0) -> dict:
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{API_BASE}{path}", json=payload)
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                return {"_error": detail, "_status": response.status_code}
            return response.json()
    except Exception as exc:
        return {"_error": str(exc), "_status": 0}


def stream_post(path: str, payload: dict, status_box, timeout: float = 180.0) -> dict:
    """消费 SSE，返回 done 事件中的完整响应。失败时回退为普通 POST。"""
    final: dict | None = None
    node_lines: list[str] = []
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{API_BASE}{path}", json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    return {"_error": response.text[:800], "_status": response.status_code}
                event = None
                for line in response.iter_lines():
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        raw = line.split(":", 1)[1].strip()
                        if event == "node":
                            try:
                                item = json.loads(raw)
                                node_lines.append(f"{item.get('node')} …")
                                status_box.markdown("**执行中**：" + " → ".join(node_lines[-6:]))
                            except Exception:
                                pass
                        elif event == "done":
                            final = json.loads(raw)
                        elif event == "error":
                            return {"_error": raw[:800], "_status": 0}
    except Exception as exc:
        return {"_error": f"stream failed: {exc}", "_status": 0}
    return final or {"_error": "stream ended without done event", "_status": 0}


# ------------------------------------------------------------------ rendering
def badge(text: str, kind: str = "info") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def render_header(resp: dict) -> None:
    intent = resp.get("intent") or {}
    coverage = resp.get("coverage") or {}
    usage = resp.get("usage") or {}
    status = resp.get("status", "unknown")
    kind = "ok" if status.startswith("completed") or status == "action_executed" else (
        "warn" if status in {"awaiting_approval", "needs_clarification"} else "info"
    )
    cols = st.columns(5)
    cols[0].markdown(badge(status, kind), unsafe_allow_html=True)
    cols[0].caption("状态")
    cols[1].metric("意图", intent.get("intent", "-"))
    cols[2].metric("服务", intent.get("service") or "-")
    cols[3].metric("证据数", coverage.get("evidence_count", len([])))
    cols[4].metric("Token / 成本", f"{usage.get('total_tokens', 0)} / ${usage.get('cost_usd', 0):.4f}")
    window = (intent.get("time_window") or {})
    if window:
        st.caption(
            f"时间窗口：{window.get('start')} ~ {window.get('end')} ｜ 解析方式：{window.get('detected_by')}"
        )
    if coverage.get("missing_sources"):
        st.warning("信息缺失维度：" + "、".join(coverage["missing_sources"]))


def render_hypotheses(resp: dict) -> None:
    hypotheses = resp.get("hypotheses") or []
    if not hypotheses:
        st.info("暂无候选根因。")
        return
    for hyp in hypotheses:
        supported = hyp.get("supported")
        flag = "证据充分" if supported else "证据不足"
        kind = "ok" if supported else "warn"
        with st.expander(
            f"{hyp.get('title')} ｜ {hyp.get('root_cause')} ｜ 置信度 {hyp.get('confidence')}",
            expanded=hyp is hypotheses[0],
        ):
            st.markdown(
                badge(flag, kind)
                + badge(f"faithfulness {hyp.get('faithfulness')}", "info")
                + badge(f"action: {hyp.get('action_required')}", "warn" if hyp.get("action_required") != "none" else "info"),
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, float(hyp.get("confidence") or 0)))
            st.markdown((hyp.get("reasoning") or "").strip() or "_无推理文本_")
            refs = hyp.get("evidence_refs") or []
            if refs:
                st.caption("引用证据：" + "、".join(f"`{r}`" for r in refs))
            if hyp.get("caveats"):
                st.markdown("**不确定性**")
                for caveat in hyp["caveats"]:
                    st.markdown(f"- {caveat}")


def render_evidence(resp: dict) -> None:
    report = resp.get("final_report") or {}
    evidence = report.get("evidence") or resp.get("evidence") or []
    if not evidence:
        coverage = resp.get("coverage") or {}
        st.info("最终报告尚未生成（可能在等待审批），当前覆盖度：" + json.dumps(coverage, ensure_ascii=False))
        return
    by_source: dict[str, list[dict]] = {}
    for item in evidence:
        by_source.setdefault(item.get("source") or "other", []).append(item)
    tabs = st.tabs([f"{name}（{len(items)}）" for name, items in by_source.items()] or ["空"])
    for tab, (name, items) in zip(tabs, by_source.items()):
        with tab:
            for item in items:
                flags = item.get("injection_flags") or []
                flag_html = badge("prompt-injection 已标记", "bad") if flags else ""
                st.markdown(
                    f"**{item.get('title')}**  \n"
                    f"<span class='small-muted'>{item.get('timestamp') or '历史/知识'} · "
                    f"置信度 {item.get('confidence')} · `{item.get('id')}`</span> {flag_html}",
                    unsafe_allow_html=True,
                )
                st.code(item.get("excerpt") or "", language="text")
                st.caption(f"tool_trace_id: {item.get('tool_trace_id')}")
                st.divider()


def render_timeline(report: dict) -> None:
    timeline = report.get("timeline") or []
    if not timeline:
        st.info("无可对齐的时间线证据。")
        return
    for item in timeline:
        st.markdown(
            f"<div class='trace-row'><b>{item.get('time')}</b> "
            f"[{item.get('source')}] {item.get('title')} "
            f"<span class='small-muted'>({item.get('evidence_id')}, conf={item.get('confidence')})</span></div>",
            unsafe_allow_html=True,
        )


def render_approval(resp: dict, session_id: str) -> dict | None:
    request = resp.get("approval_request")
    if not request:
        return None
    action = request.get("action") or {}
    st.error("检测到高风险动作，工作流已暂停，等待人工确认。")
    st.markdown(f"**候选根因**：{request.get('title')}（置信度 {request.get('confidence')}）")
    st.markdown("**将执行的命令**")
    st.code(action.get("command") or "", language="bash")
    st.markdown("**变更 diff**")
    st.code(action.get("diff") or "", language="diff")
    st.caption(
        f"目标：{action.get('target')} ｜ 风险：{action.get('risk')} ｜ "
        f"幂等键：{action.get('idempotency_key')} ｜ 过期时间：{request.get('expires_at')}"
    )
    for risk in request.get("risks") or []:
        st.markdown(f"- {risk}")
    with st.expander("审批所依据的证据", expanded=False):
        for item in request.get("evidence") or []:
            st.markdown(f"**{item.get('title')}**（{item.get('source')}，{item.get('timestamp')}）")
            st.code(item.get("excerpt") or "", language="text")

    confirmed = st.checkbox("我已核对命令、目标服务与 diff", key=f"confirm-{request.get('hypothesis_id')}")
    comment = st.text_area("审批意见（拒绝时必填）", key=f"comment-{request.get('hypothesis_id')}")
    col1, col2 = st.columns(2)
    approve_clicked = col1.button("批准并执行", type="primary", disabled=not confirmed)
    reject_clicked = col2.button("拒绝", disabled=not comment.strip())

    if approve_clicked or reject_clicked:
        payload = {
            "approved": approve_clicked,
            "approver_id": st.session_state.get("approver_id", "oncall-lead"),
            "approver_role": st.session_state.get("approver_role", "oncall"),
            "comment": comment,
        }
        result = api_post(f"/approvals/{session_id}", payload, timeout=180)
        if result.get("_error"):
            st.error(f"审批失败：{result['_error']}")
            return None
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (result.get("final_report") or {}).get("summary") or result.get("message", ""),
            }
        )
        return result
    return None


def render_trace(trace_id: str) -> None:
    data = api_get(f"/traces/{trace_id}")
    if data.get("_error"):
        st.info("Trace 尚不可用：" + str(data["_error"]))
        return
    spans = data.get("spans") or []
    rows = []
    for span in spans:
        attrs = span.get("attributes") or {}
        rows.append(
            {
                "span": span.get("name"),
                "duration_ms": span.get("duration_ms"),
                "start": span.get("start_time"),
                "tool": attrs.get("tool.name"),
                "tokens": attrs.get("gen_ai.usage.input_tokens"),
                "cost_usd": attrs.get("app.cost_usd"),
                "error": (span.get("status") or {}).get("status_code"),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)
    selected = st.selectbox("查看 span 详情", [s.get("name") for s in spans], key=f"span-{trace_id}")
    if selected:
        match = next(s for s in spans if s.get("name") == selected)
        st.json(match)


def render_report(resp: dict) -> None:
    report = resp.get("final_report")
    if report:
        st.markdown(report.get("markdown") or report.get("summary") or "")
    else:
        st.info("尚未生成最终报告；完成审批后会输出带执行结果的报告。")


# --------------------------------------------------------------------- sidebar
st.title("企业级运维诊断 Agent")
st.caption("可审计、可评估的故障初筛与根因分析 ｜ LangGraph · Tool Gateway · OTel")

with st.sidebar:
    st.subheader("身份与权限")
    st.session_state.setdefault("messages", [])
    role = st.selectbox("请求角色", ["viewer", "sre", "oncall", "admin"], index=2)
    st.session_state["role"] = role
    st.text_input("用户", value="oncall-zhang", key="user_id")
    st.text_input("团队", value="team-payments", key="team_id")
    st.selectbox("审批角色", ["sre", "oncall", "admin"], index=1, key="approver_role")
    st.text_input("审批人", value="oncall-lead", key="approver_id")
    stream_mode = st.toggle("流式显示节点进度", value=True)
    if st.button("新会话", width="stretch"):
        for key in ["session_id", "last_response"]:
            st.session_state.pop(key, None)
        st.session_state["messages"] = []
        st.rerun()
    st.divider()
    health = api_get("/health")
    if health.get("_error"):
        st.error(f"API 不可用：{health['_error']}")
    else:
        st.success("API 在线")
        st.caption(
            f"LLM：{health.get('llm', {}).get('model')} ｜ "
            f"会话：{health.get('backends', {}).get('sessions')} ｜ "
            f"记忆：{health.get('backends', {}).get('long_term_memory')}"
        )
        st.caption(f"知识库：{health.get('knowledge', {}).get('documents')} 篇 ｜ 工具：{health.get('tools')} 个")
    st.divider()
    if st.session_state.get("session_id"):
        st.caption(f"session: {st.session_state['session_id']}")
        if st.session_state.get("last_response"):
            st.caption(f"trace: {st.session_state['last_response'].get('trace_id')}")

# ------------------------------------------------------------------ main input
with st.form("query-form", clear_on_submit=False):
    query = st.text_area("描述异常现象", value=DEFAULT_QUERY, height=110)
    submitted = st.form_submit_button("开始诊断", type="primary", width="stretch")

if submitted and query.strip():
    payload = {
        "message": query.strip(),
        "session_id": st.session_state.get("session_id"),
        "user_id": st.session_state.get("user_id", "oncall-zhang"),
        "team_id": st.session_state.get("team_id", "team-payments"),
        "role": st.session_state.get("role", "oncall"),
    }
    st.session_state.messages.append({"role": "user", "content": query.strip()})
    status_box = st.status("正在诊断…", expanded=True)
    if stream_mode:
        with st.spinner("编排图中…"):
            response = stream_post("/chat/stream", payload, status_box)
    else:
        with st.spinner("诊断中（工具取证 / 假设生成 / 审查）…"):
            response = api_post("/chat", payload)
    status_box.update(label="诊断阶段完成", state="complete")

    if response.get("_error"):
        st.error(f"请求失败（{response.get('_status')}）：{response['_error']}")
        if response.get("_status") == 409 and st.session_state.get("last_response"):
            st.warning("当前会话有待审批动作，请先在下方处理。")
    else:
        st.session_state["session_id"] = response["session_id"]
        st.session_state["last_response"] = response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (response.get("final_report") or {}).get("summary") or response.get("message", ""),
            }
        )

# ------------------------------------------------------------------- last view
response = st.session_state.get("last_response")
if response:
    render_header(response)

    if response.get("status") == "needs_clarification":
        st.warning(response.get("clarification_question") or response.get("message") or "请补充信息。")

    approval_result = render_approval(response, response.get("session_id"))
    if approval_result:
        st.session_state["last_response"] = approval_result
        response = approval_result
        st.rerun()

    tab_report, tab_hyp, tab_evidence, tab_timeline, tab_trace = st.tabs(
        ["报告", "候选根因", "证据面板", "时间线", "Trace"]
    )
    with tab_report:
        render_report(response)
    with tab_hyp:
        render_hypotheses(response)
    with tab_evidence:
        render_evidence(response)
    with tab_timeline:
        render_timeline(
            {"timeline": (response.get("final_report") or {}).get("timeline") or response.get("timeline") or []}
        )
    with tab_trace:
        render_trace(response.get("trace_id", ""))

    with st.expander("原始响应（调试）", expanded=False):
        st.json({k: v for k, v in response.items() if k != "final_report"})
else:
    st.info("输入现象后开始诊断。离线模式下无需 LLM Key，规则引擎会给出可复现的候选根因。")

if st.session_state.get("messages"):
    with st.sidebar:
        st.divider()
        st.subheader("会话记录")
        for message in st.session_state["messages"][-8:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])