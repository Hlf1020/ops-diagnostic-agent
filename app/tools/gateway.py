"""MCP 风格工具网关：Schema 校验、权限裁剪、超时、输出净化与审计。

Agent 永远不直接持有后端凭证，也不直接调用后端 SDK。所有调用都经过这里，
这也是回答“用户越权请求删除/重启怎么办”的工程锚点。
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from app.config import Settings
from app.core.text import safe_json_dumps, sanitize_value, short_hash
from app.core.timeutil import to_iso
from app.core.tracing import span
from app.models import Role, UserContext
from app.tools.registry import ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)

ROLE_PERMISSIONS: dict[str, set[str]] = {
    Role.VIEWER.value: {"kb:read", "metrics:read"},
    Role.SRE.value: {"kb:read", "metrics:read", "logs:read", "changes:read"},
    Role.ONCALL.value: {"kb:read", "metrics:read", "logs:read", "changes:read", "remediation:write"},
    Role.ADMIN.value: {"*"},
}

_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool-gw")


def permissions_for(role: str | Role) -> set[str]:
    value = role.value if isinstance(role, Role) else str(role)
    return ROLE_PERMISSIONS.get(value, set())


def has_permission(role: str | Role, permission: str) -> bool:
    perms = permissions_for(role)
    return "*" in perms or permission in perms


@dataclass
class ToolResult:
    ok: bool
    tool: str
    output: dict | None
    error: dict | None
    duration_ms: float
    truncated: bool
    injection_flags: list[str]
    trace_id: str | None
    span_id: str | None
    args_preview: str
    started_at: str

    @property
    def call_id(self) -> str:
        return f"{self.trace_id or 'no-trace'}/{self.span_id or 'no-span'}"

    def record(self) -> dict:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "duration_ms": round(self.duration_ms, 2),
            "error_type": (self.error or {}).get("type"),
            "retryable": (self.error or {}).get("retryable", False),
            "output_chars": len(safe_json_dumps(self.output)) if self.output is not None else 0,
            "truncated": self.truncated,
            "injection_flags": self.injection_flags,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "call_id": self.call_id,
            "started_at": self.started_at,
            "args_preview": self.args_preview,
        }

    def error_record(self) -> dict | None:
        if self.ok or not self.error:
            return None
        return {
            "tool": self.tool,
            "error_type": self.error["type"],
            "message": self.error["message"][:500],
            "retryable": bool(self.error.get("retryable")),
            "duration_ms": round(self.duration_ms, 2),
            "trace_id": self.trace_id,
        }


class ToolGateway:
    def __init__(self, registry: ToolRegistry, settings: Settings):
        self.registry = registry
        self.settings = settings
        self._fail_tools = {x.strip() for x in settings.mock_fail_tools.split(",") if x.strip()}
        self._delays: dict[str, int] = {}
        for item in settings.mock_delay.split(","):
            if ":" in item:
                name, _, value = item.partition(":")
                try:
                    self._delays[name.strip()] = int(value)
                except ValueError:
                    logger.warning("invalid MOCK_DELAY entry: %s", item)

    # -------------------------------------------------------------- exposure
    def visible_specs(self, role: str | Role, allowed: set[str] | None = None) -> list[ToolSpec]:
        specs = [s for s in self.registry.all() if has_permission(role, s.required_permission)]
        if allowed is not None:
            specs = [s for s in specs if s.name in allowed]
        return specs

    def openai_tools(self, role: str | Role, allowed: set[str] | None = None) -> list[dict]:
        return [spec.openai_schema() for spec in self.visible_specs(role, allowed)]

    def describe_tools(self, role: str | Role) -> list[dict]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "permission": spec.required_permission,
                "tags": spec.tags,
                "input_schema": spec.json_schema(),
            }
            for spec in self.visible_specs(role)
        ]

    # ------------------------------------------------------------------ call
    def call(
        self,
        tool_name: str,
        args: dict,
        user: UserContext | dict,
        *,
        trace_id: str | None = None,
        node: str = "tool",
    ) -> ToolResult:
        user_ctx = user if isinstance(user, UserContext) else UserContext.model_validate(user)
        started_at = to_iso(datetime.now().astimezone())
        started = time.perf_counter()

        spec = self.registry.get(tool_name)
        if spec is None:
            return self._fail(
                tool_name, args, started, started_at, trace_id, None,
                "not_found", f"unknown tool: {tool_name}", retryable=False,
            )
        if not has_permission(user_ctx.role, spec.required_permission):
            return self._fail(
                tool_name, args, started, started_at, trace_id, None,
                "permission_denied",
                f"role={user_ctx.role.value} lacks permission {spec.required_permission}",
                retryable=False,
            )

        try:
            validated = spec.input_model.model_validate(args)
        except ValidationError as exc:
            detail = json.dumps(exc.errors(), ensure_ascii=False, default=str)
            return self._fail(
                tool_name, args, started, started_at, trace_id, None,
                "validation_error", detail[:1200], retryable=False,
            )

        attrs = {
            "tool.name": tool_name,
            "tool.permission": spec.required_permission,
            "tool.args_hash": short_hash(safe_json_dumps(args)),
            "tool.timeout_s": spec.timeout_s,
            "app.node": node,
            "app.role": user_ctx.role.value,
        }
        with span(f"tool.{tool_name}", trace_id=trace_id, attrs=attrs) as current:
            span_context = current.get_span_context()
            span_id = format(span_context.span_id, "016x")
            try:
                if tool_name in self._fail_tools:
                    raise RuntimeError("injected tool failure (MOCK_FAIL_TOOLS)")
                delay_ms = self._delays.get(tool_name, 0)
                future = _EXECUTOR.submit(self._invoke, spec, validated, delay_ms)
                output = future.result(timeout=spec.timeout_s)
            except FutureTimeout:
                current.set_attribute("error.type", "timeout")
                return self._fail(
                    tool_name, args, started, started_at, trace_id, span_id,
                    "timeout", f"tool timed out after {spec.timeout_s:.1f}s", retryable=True,
                )
            except Exception as exc:
                current.set_attribute("error.type", type(exc).__name__)
                return self._fail(
                    tool_name, args, started, started_at, trace_id, span_id,
                    "injected_failure" if "injected" in str(exc) else "handler_error",
                    f"{type(exc).__name__}: {exc}", retryable="injected" not in str(exc),
                )

            flags: set[str] = set()
            clean_output = sanitize_value(output, flags)
            serialized = safe_json_dumps(clean_output)
            truncated = len(serialized) > spec.max_output_chars
            if truncated:
                clean_output = {
                    "_truncated": True,
                    "_original_chars": len(serialized),
                    "_limit": spec.max_output_chars,
                    "preview": serialized[: spec.max_output_chars // 2],
                }
                serialized = safe_json_dumps(clean_output)

            duration_ms = (time.perf_counter() - started) * 1000
            current.set_attribute("tool.ok", True)
            current.set_attribute("tool.duration_ms", round(duration_ms, 2))
            current.set_attribute("tool.output_chars", len(serialized))
            current.set_attribute("tool.truncated", truncated)
            current.set_attribute("tool.injection_flags", ",".join(sorted(flags)) or "none")
            current.set_attribute("tool.call_id", f"{trace_id or ''}/{span_id}")

            return ToolResult(
                ok=True,
                tool=tool_name,
                output=clean_output,
                error=None,
                duration_ms=duration_ms,
                truncated=truncated,
                injection_flags=sorted(flags),
                trace_id=trace_id,
                span_id=span_id,
                args_preview=safe_json_dumps(args, max_chars=500),
                started_at=started_at,
            )

    def _invoke(self, spec: ToolSpec, validated, delay_ms: int):
        if delay_ms:
            time.sleep(delay_ms / 1000)
        return spec.handler(validated)

    def _fail(
        self,
        tool: str,
        args: dict,
        started: float,
        started_at: str,
        trace_id: str | None,
        span_id: str | None,
        error_type: str,
        message: str,
        *,
        retryable: bool,
    ) -> ToolResult:
        duration_ms = (time.perf_counter() - started) * 1000
        logger.warning("tool call failed tool=%s type=%s message=%s", tool, error_type, message)
        return ToolResult(
            ok=False,
            tool=tool,
            output=None,
            error={"type": error_type, "message": message, "retryable": retryable},
            duration_ms=duration_ms,
            truncated=False,
            injection_flags=[],
            trace_id=trace_id,
            span_id=span_id,
            args_preview=safe_json_dumps(args, max_chars=500),
            started_at=started_at,
        )