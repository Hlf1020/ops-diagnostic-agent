"""OpenTelemetry 接入 + JSONL Trace 落盘。

每个 LangGraph 节点、每次工具调用、每次 LLM 调用都会开 span，并强制写入
`app.trace_id`。TraceStore 按该属性聚合，因此即使 span 分布在 LangGraph 的
线程池中，也能还原出同一次诊断的完整证据链。
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, TracerProvider
from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.trace import Status, StatusCode

from app.core.text import safe_json_dumps

logger = logging.getLogger(__name__)

_INIT_LOCK = threading.Lock()
_INITIALIZED = False
_TRACER_NAME = "ops-diagnostic-agent"


class JsonlSpanProcessor(SpanProcessor):
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = self.directory / "spans.jsonl"

    def on_start(self, span: Span, parent_context=None) -> None:  # noqa: ARG002
        return None

    def on_end(self, span: ReadableSpan) -> None:
        try:
            data = json.loads(span.to_json())
            data["app.trace_id"] = span.attributes.get("app.trace_id", format(span.context.trace_id, "032x"))
            line = json.dumps(data, ensure_ascii=False)
            with self._lock:
                with self._file.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:  # tracing 绝不能影响主流程
            logger.exception("failed to export span")

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 3000) -> bool:
        return True


def init_tracing(settings) -> None:
    global _INITIALIZED
    with _INIT_LOCK:
        if _INITIALIZED or not getattr(settings, "tracing_enabled", False):
            return
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": settings.app_name,
                    "service.version": settings.app_version,
                    "deployment.environment": "local",
                }
            )
        )
        provider.add_span_processor(JsonlSpanProcessor(settings.resolved_trace_dir))
        trace.set_tracer_provider(provider)
        _INITIALIZED = True


def get_tracer():
    return trace.get_tracer(_TRACER_NAME)


def _coerce_attrs(attrs: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (attrs or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            out[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, (str, bool, int, float)) for v in value):
            out[key] = list(value)
        else:
            out[key] = safe_json_dumps(value, max_chars=1500)
    return out


@contextmanager
def span(
    name: str,
    *,
    trace_id: str | None = None,
    attrs: dict[str, Any] | None = None,
) -> Iterator[Span]:
    attributes = _coerce_attrs(attrs)
    if trace_id:
        attributes["app.trace_id"] = trace_id
    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes) as current:
        try:
            yield current
        except Exception as exc:
            current.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
            current.record_exception(exc)
            raise


def set_span_attrs(current: Span, attrs: dict[str, Any]) -> None:
    for key, value in _coerce_attrs(attrs).items():
        current.set_attribute(key, value)


def add_span_event(current: Span, name: str, attrs: dict[str, Any] | None = None) -> None:
    current.add_event(name, _coerce_attrs(attrs))


class TraceStore:
    """按 app.trace_id 聚合 JSONL span，供 API 与 Streamlit 回放。"""

    def __init__(self, directory: Path):
        self.directory = directory
        self.file = directory / "spans.jsonl"

    def _load(self) -> list[dict]:
        if not self.file.exists():
            return []
        spans: list[dict] = []
        with self.file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    spans.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return spans

    def get(self, trace_id: str) -> list[dict]:
        spans = [s for s in self._load() if s.get("app.trace_id") == trace_id]
        spans.sort(key=lambda s: s.get("start_time", 0))
        return [self._shape(s) for s in spans]

    def list_traces(self, limit: int = 50) -> list[dict]:
        grouped: dict[str, list[dict]] = {}
        for s in self._load():
            # 只展示本 Agent 显式打标的 trace，过滤 FastAPI/ASGI 自动埋点
            tid = s.get("app.trace_id")
            if tid:
                grouped.setdefault(tid, []).append(s)
        summaries = []
        for tid, items in grouped.items():
            items.sort(key=lambda s: s.get("start_time", 0))
            root = next((s for s in items if s.get("name") == "request.chat"), items[0])
            attrs = root.get("attributes", {})
            error_count = sum(1 for s in items if s.get("status", {}).get("status_code") == "ERROR")
            duration_ms = _duration_ms(root.get("start_time"), root.get("end_time"))
            summaries.append(
                {
                    "trace_id": tid,
                    "name": root.get("name"),
                    "start_time": _ns_to_iso(root.get("start_time", 0)),
                    "duration_ms": duration_ms,
                    "span_count": len(items),
                    "error_count": error_count,
                    "session_id": attrs.get("app.session_id"),
                    "intent": attrs.get("app.intent"),
                    "service": attrs.get("app.service"),
                    "status": attrs.get("app.status"),
                }
            )
        summaries.sort(key=lambda x: x["start_time"] or "", reverse=True)
        return summaries[:limit]

    def _shape(self, raw: dict) -> dict:
        start = raw.get("start_time", 0)
        end = raw.get("end_time", 0)
        return {
            "name": raw.get("name"),
            "trace_id": raw.get("app.trace_id") or raw.get("trace_id"),
            "span_id": raw.get("span_id"),
            "parent_span_id": raw.get("parent_span_id"),
            "start_time": _ns_to_iso(start),
            "end_time": _ns_to_iso(end),
            "duration_ms": _duration_ms(start, end),
            "attributes": raw.get("attributes", {}),
            "events": raw.get("events", []),
            "status": raw.get("status", {}),
        }


def _to_datetime(value) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1e9, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _ns_to_iso(value) -> str | None:
    dt = _to_datetime(value)
    return dt.isoformat(timespec="milliseconds") if dt else None


def _duration_ms(start, end) -> float:
    start_dt = _to_datetime(start)
    end_dt = _to_datetime(end)
    if not start_dt or not end_dt:
        return 0.0
    return round(max(0.0, (end_dt - start_dt).total_seconds() * 1000), 2)