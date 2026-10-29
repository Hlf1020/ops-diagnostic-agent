"""FastAPI 接入层。

- POST /chat：同步诊断（LLM/工具较慢时在线程池执行，不阻塞事件循环）
- POST /approvals/{session_id}：恢复 interrupt
- GET  /traces/{trace_id}：证据链回放
- POST /chat/stream：SSE 推送节点进度
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator


from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command

from app.config import get_settings
from app.core.timeutil import parse_iso, to_iso
from app.core.tracing import set_span_attrs, span
from app.graph.state import DiagnosticState
from app.memory.session import SessionMismatch, session_context
from app.models import ApprovalRequest, ChatRequest, ChatResponse, Role, UserContext
from app.runtime import build_runtime
from app.tools.gateway import has_permission

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    app.state.runtime = build_runtime()
    yield
    app.state.runtime.close()


# 默认 Swagger UI 从 jsdelivr CDN 加载 CSS/JS，内网或受限网络下 /docs 会白屏。
# 改为使用 swagger-ui-bundle 自带的本地静态资源。
_SWAGGER_DIR = Path(__file__).resolve().parent / "static" / "swagger"


app = FastAPI(
    title="Ops Diagnostic Agent",
    version="1.0.0",
    description="可审计、可评估的企业运维诊断 Agent（LangGraph + Tool Gateway + OTel）",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
app.mount("/static/swagger", StaticFiles(directory=_SWAGGER_DIR), name="swagger-static")


@app.get("/docs", include_in_schema=False)
async def swagger_docs():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Ops Diagnostic Agent - Swagger UI",
        swagger_js_url="/static/swagger/swagger-ui-bundle.js?v=5",
        swagger_css_url="/static/swagger/swagger-ui.css?v=5",
        swagger_favicon_url="/static/swagger/favicon-32x32.png",
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _rt(request: Request):
    return request.app.state.runtime


def _get_session(runtime, session_id: str, user: UserContext):
    try:
        return runtime.sessions.get_or_create(session_id, user)
    except SessionMismatch as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _prepare_turn(runtime, req: ChatRequest, session_id: str):
    user = UserContext(user_id=req.user_id, team_id=req.team_id, role=req.role)
    session = _get_session(runtime, session_id, user)
    if session.open_approval:
        expires = session.open_approval.get("expires_at")
        expired = False
        if expires:
            try:
                expired = parse_iso(expires) < datetime.now(timezone.utc)
            except Exception:
                expired = False
        if not expired:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "当前会话有待人工确认的高风险动作，请先批准或拒绝。",
                    "approval_request": session.open_approval.get("request"),
                },
            )
        session.open_approval = None
        runtime.sessions.save(session)

    pending_query, _ = session.pop_pending()
    effective_query = f"{pending_query}\n补充信息：{req.message}" if pending_query else req.message

    if session.turn_counter == 0 or session.turns:
        thread_id = session.start_new_turn()
    else:
        thread_id = session.thread_id
    session.append_turn("user", req.message, {"effective_query": effective_query[:500]})
    runtime.sessions.save(session)

    trace_id = uuid.uuid4().hex
    state = DiagnosticState(
        session_id=session.session_id,
        thread_id=thread_id,
        trace_id=trace_id,
        user=user.model_dump(mode="json"),
        query=effective_query,
        conversation=session_context(session, runtime.settings.context_turn_limit),
        status="new",
        retry_count=0,
    )
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    return session, thread_id, trace_id, state, config, effective_query


def _usage(result: dict) -> dict:
    records = result.get("llm_usage") or []
    input_tokens = sum(int(r.get("input_tokens", 0) or 0) for r in records)
    output_tokens = sum(int(r.get("output_tokens", 0) or 0) for r in records)
    cost = sum(float(r.get("cost_usd", 0) or 0) for r in records)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(cost, 6),
        "calls": len(records),
    }


def _public_hypotheses(result: dict) -> list[dict]:
    keys = [
        "id",
        "title",
        "root_cause",
        "confidence",
        "supported",
        "faithfulness",
        "action_required",
        "proposed_action",
        "caveats",
    ]
    return [{k: h.get(k) for k in keys} for h in (result.get("hypotheses") or [])[:4]]


def _public_evidence(result: dict) -> list[dict]:
    items = []
    for item in (result.get("evidence") or [])[:30]:
        items.append(
            {
                "id": item.get("id"),
                "source": item.get("source"),
                "title": item.get("title"),
                "timestamp": item.get("timestamp"),
                "confidence": item.get("confidence"),
                "excerpt": (item.get("raw_excerpt") or "")[:500],
                "tool_trace_id": item.get("tool_trace_id"),
                "injection_flags": item.get("injection_flags") or [],
            }
        )
    return items


def _build_response(
    *,
    session_id: str,
    thread_id: str,
    trace_id: str,
    result: dict,
    interrupted: bool,
    latency_ms: float,
) -> ChatResponse:
    if interrupted:
        status = "awaiting_approval"
        message = "候选根因已生成，但首要假设涉及高风险动作，已暂停等待人工确认。"
    elif result.get("status") == "needs_clarification":
        status = "needs_clarification"
        message = result.get("clarification_question")
    else:
        report = result.get("final_report") or {}
        status = report.get("status") or result.get("status") or "completed"
        message = report.get("summary")

    return ChatResponse(
        session_id=session_id,
        thread_id=thread_id,
        trace_id=trace_id,
        status=status,
        message=message,
        clarification_question=result.get("clarification_question"),
        intent=result.get("intent"),
        approval_request=result.get("approval_request") if interrupted else None,
        hypotheses=_public_hypotheses(result),
        evidence=_public_evidence(result),
        timeline=(result.get("timeline") or [])[:40],
        final_report=result.get("final_report") if not interrupted else None,
        coverage=result.get("coverage"),
        usage=_usage(result),
        latency_ms=round(latency_ms, 2),
    )


def _persist_after(runtime, session, result: dict, *, interrupted: bool, thread_id: str, trace_id: str):
    if interrupted:
        session.open_approval = {
            "thread_id": thread_id,
            "trace_id": trace_id,
            "request": result.get("approval_request"),
            "created_at": to_iso(datetime.now(timezone.utc)),
            "expires_at": (result.get("approval_request") or {}).get("expires_at"),
        }
    else:
        session.open_approval = None
        if result.get("status") == "needs_clarification":
            intent = result.get("intent") or {}
            if intent.get("missing_slots"):
                session.set_pending(
                    result.get("query", ""),
                    intent.get("missing_slots") or [],
                )
    runtime.sessions.save(session)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    runtime = _rt(request)
    session_id = req.session_id or f"sess-{uuid.uuid4().hex[:16]}"
    session, thread_id, trace_id, state, config, _ = _prepare_turn(runtime, req, session_id)
    started = time.perf_counter()

    with span(
        "request.chat",
        trace_id=trace_id,
        attrs={
            "app.session_id": session_id,
            "app.user_id": req.user_id,
            "app.team_id": req.team_id,
            "app.role": req.role.value,
            "app.query_chars": len(state.get("query", "")),
        },
    ) as current:
        try:
            result = await asyncio.to_thread(runtime.graph.invoke, state, config)
        except Exception as exc:
            logger.exception("diagnosis failed")
            set_span_attrs(current, {"app.status": "error", "error.type": type(exc).__name__})
            raise HTTPException(status_code=500, detail=f"diagnosis failed: {exc}") from exc

        interrupted = "__interrupt__" in result
        _persist_after(
            runtime, session, result, interrupted=interrupted, thread_id=thread_id, trace_id=trace_id
        )
        set_span_attrs(
            current,
            {
                "app.status": "awaiting_approval" if interrupted else result.get("status"),
                "app.intent": (result.get("intent") or {}).get("intent"),
                "app.service": (result.get("intent") or {}).get("service"),
                "app.evidence_count": len(result.get("evidence") or []),
            },
        )

    return _build_response(
        session_id=session_id,
        thread_id=thread_id,
        trace_id=trace_id,
        result=result,
        interrupted=interrupted,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


@app.post("/approvals/{session_id}", response_model=ChatResponse)
async def approve(session_id: str, req: ApprovalRequest, request: Request):
    runtime = _rt(request)
    session = runtime.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    open_approval = session.open_approval
    if not open_approval:
        raise HTTPException(status_code=409, detail="no pending approval for this session")

    expires = open_approval.get("expires_at")
    if expires:
        try:
            if parse_iso(expires) < datetime.now(timezone.utc):
                session.open_approval = None
                runtime.sessions.save(session)
                raise HTTPException(status_code=410, detail="approval request expired")
        except HTTPException:
            raise
        except Exception:
            pass

    if req.approved and not has_permission(req.approver_role, "remediation:write"):
        raise HTTPException(
            status_code=403,
            detail=f"approver role {req.approver_role.value} cannot execute remediation",
        )

    thread_id = open_approval["thread_id"]
    trace_id = open_approval.get("trace_id") or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    started = time.perf_counter()

    session.append_turn(
        "user",
        f"审批：{'批准' if req.approved else '拒绝'}（{req.approver_id}，{req.approver_role.value}）{req.comment}",
        {"trace_id": trace_id},
    )
    runtime.sessions.save(session)

    resume = {
        "approved": req.approved,
        "approver_id": req.approver_id,
        "approver_role": req.approver_role.value,
        "comment": req.comment,
        "confirm_token": req.confirm_token,
    }
    with span(
        "request.approval",
        trace_id=trace_id,
        attrs={
            "app.session_id": session_id,
            "app.approver_id": req.approver_id,
            "app.approver_role": req.approver_role.value,
            "app.approved": req.approved,
        },
    ) as current:
        try:
            result = await asyncio.to_thread(
                runtime.graph.invoke, Command(resume=resume), config
            )
        except Exception as exc:
            logger.exception("approval resume failed")
            set_span_attrs(current, {"app.status": "error", "error.type": type(exc).__name__})
            raise HTTPException(status_code=500, detail=f"approval resume failed: {exc}") from exc

        interrupted = "__interrupt__" in result
        if not interrupted:
            session.open_approval = None
            runtime.sessions.save(session)
        set_span_attrs(
            current,
            {
                "app.status": result.get("status"),
                "app.action_result": (result.get("action_result") or {}).get("status"),
            },
        )

    return _build_response(
        session_id=session_id,
        thread_id=thread_id,
        trace_id=trace_id,
        result=result,
        interrupted=interrupted,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    runtime = _rt(request)
    session_id = req.session_id or f"sess-{uuid.uuid4().hex[:16]}"
    session, thread_id, trace_id, state, config, _ = _prepare_turn(runtime, req, session_id)
    started = time.perf_counter()

    async def events() -> AsyncIterator[str]:
        # SqliteSaver 是同步 checkpointer，graph.astream 会直接报错；
        # 因此在后台线程跑同步 graph.stream，通过队列把分片交回事件循环。
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        saw_interrupt = False
        last_values: dict = dict(state)

        def emit(item) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def run_stream() -> None:
            try:
                for chunk in runtime.graph.stream(state, config, stream_mode="updates"):
                    emit(("chunk", chunk))
                emit(("snapshot", runtime.graph.get_state(config)))
            except Exception as exc:  # 交给事件循环侧统一处理
                emit(("error", exc))
            finally:
                emit(("end", None))

        threading.Thread(target=run_stream, name=f"graph-stream-{trace_id[:8]}", daemon=True).start()

        try:
            while True:
                kind, payload = await queue.get()
                if kind == "end":
                    break
                if kind == "error":
                    raise payload
                if kind == "snapshot":
                    snapshot = payload
                    if snapshot and snapshot.values:
                        last_values.update(snapshot.values)
                    if snapshot and getattr(snapshot, "next", None) and "human_approval" in snapshot.next:
                        saw_interrupt = True
                    continue
                for node, update in (payload or {}).items():
                    if node == "__interrupt__":
                        saw_interrupt = True
                        payload_text = json.dumps(
                            {"node": node, "approval": result_safe(update)},
                            ensure_ascii=False,
                        )
                        yield f"event: interrupt\ndata: {payload_text}\n\n"
                        continue
                    if isinstance(update, dict):
                        last_values.update(update)
                    summary = {
                        "node": node,
                        "status": (update or {}).get("status") if isinstance(update, dict) else None,
                        "evidence_added": len((update or {}).get("raw_evidence") or [])
                        if isinstance(update, dict)
                        else 0,
                    }
                    yield f"event: node\ndata: {json.dumps(summary, ensure_ascii=False)}\n\n"
            interrupted = saw_interrupt or "__interrupt__" in last_values
            _persist_after(
                runtime, session, last_values, interrupted=interrupted, thread_id=thread_id, trace_id=trace_id
            )
            response = _build_response(
                session_id=session_id,
                thread_id=thread_id,
                trace_id=trace_id,
                result=last_values,
                interrupted=interrupted,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            yield f"event: done\ndata: {response.model_dump_json()}\n\n"
        except Exception as exc:
            logger.exception("stream failed")
            error = json.dumps({"status": "error", "message": str(exc)[:500]}, ensure_ascii=False)
            yield f"event: error\ndata: {error}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def result_safe(value) -> dict:
    try:
        return {"payload": str(value)[:2000]}
    except Exception:
        return {}


@app.get("/health")
async def health(request: Request):
    runtime = _rt(request)
    return {
        "status": "ok",
        "app": runtime.settings.app_name,
        "version": runtime.settings.app_version,
        "llm": {
            "enabled": runtime.llm.enabled,
            "model": runtime.settings.llm_model if runtime.llm.enabled else "offline-fallback",
        },
        "backends": {
            "sessions": runtime.sessions.backend_name(),
            "long_term_memory": runtime.memory.backend_name(),
            "checkpointer": runtime.settings.checkpoint_backend,
        },
        "knowledge": runtime.kb.stats(),
        "tools": len(runtime.gateway.registry.all()),
    }


@app.get("/tools")
async def tools(request: Request, role: Role = Query(default=Role.ONCALL)):
    return {"role": role.value, "tools": _rt(request).gateway.describe_tools(role)}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    session = _rt(request).sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "team_id": session.team_id,
        "role": session.role,
        "turn_counter": session.turn_counter,
        "thread_id": session.thread_id,
        "summary": session.summary,
        "pending_query": session.pending_query,
        "pending_missing_slots": session.pending_missing_slots,
        "open_approval": session.open_approval,
        "turns": session.turns[-6:],
    }


@app.get("/traces")
async def list_traces(request: Request, limit: int = Query(default=50, ge=1, le=200)):
    return {"traces": _rt(request).traces.list_traces(limit=limit)}


@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str, request: Request):
    spans = _rt(request).traces.get(trace_id)
    if not spans:
        raise HTTPException(status_code=404, detail="trace not found")
    return {"trace_id": trace_id, "span_count": len(spans), "spans": spans}


@app.get("/audit/approvals")
async def audit_approvals(request: Request):
    return _rt(request).audit.approval_stats()


@app.get("/memory/stats")
async def memory_stats(request: Request):
    return _rt(request).memory.stats()