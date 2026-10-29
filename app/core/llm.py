"""OpenAI-compatible LLM 客户端（DeepSeek / Qwen / GPT / Claude 兼容网关均可）。

设计要点：
- 没有 API key 时不算失败，而是显式暴露 `enabled=False`，让编排层进入确定性
  降级路径（离线演示、评测可复现）。
- 超时 / 429 / 5xx 重试；参数错误不重试。
- JSON 输出先解析再 Pydantic 校验，校验失败追加一次“修复”请求，仍失败则抛错，
  上层回退到规则引擎，绝不把坏 JSON 塞进状态。
- 每次调用写 OTel span，记录 token 与成本。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.core.tracing import add_span_event, set_span_attrs, span
from app.core.text import extract_json_object, safe_json_dumps

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, settings: Settings) -> float:
        return (
            self.input_tokens / 1_000_000 * settings.llm_input_price_per_mtok
            + self.output_tokens / 1_000_000 * settings.llm_output_price_per_mtok
        )

    def as_dict(self, settings: Settings) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd(settings), 6),
        }


@dataclass
class LLMMessage:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict)


@dataclass
class JSONResult:
    parsed: BaseModel
    text: str
    usage: Usage
    attempts: int


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.llm_base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.llm_timeout_s),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def enabled(self) -> bool:
        return self.settings.llm_enabled

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        trace_id: str | None = None,
        node: str = "llm",
        temperature: float | None = None,
    ) -> LLMMessage:
        if not self.enabled:
            raise LLMError("LLM is not configured (LLM_API_KEY empty)")

        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format

        last_error: Exception | None = None
        with span(
            f"llm.{node}",
            trace_id=trace_id,
            attrs={
                "gen_ai.system": "openai_compatible",
                "gen_ai.request.model": self.settings.llm_model,
                "gen_ai.request.temperature": payload["temperature"],
                "app.node": node,
                "llm.tool_count": len(tools or []),
            },
        ) as current:
            for attempt in range(self.settings.llm_max_retries + 1):
                try:
                    response = self._client.post("/chat/completions", json=payload)
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    last_error = LLMError(f"network error: {exc}", retryable=True)
                    add_span_event(current, "llm.retry", {"attempt": attempt, "error": type(exc).__name__})
                else:
                    if response.status_code == 429 or response.status_code >= 500:
                        last_error = LLMError(
                            f"upstream {response.status_code}: {response.text[:300]}",
                            retryable=True,
                            status_code=response.status_code,
                        )
                        add_span_event(current, "llm.retry", {"attempt": attempt, "status": response.status_code})
                    elif response.status_code >= 400:
                        raise LLMError(
                            f"LLM request failed {response.status_code}: {response.text[:500]}",
                            retryable=False,
                            status_code=response.status_code,
                        )
                    else:
                        data = response.json()
                        message = (data.get("choices") or [{}])[0].get("message", {})
                        usage_raw = data.get("usage") or {}
                        usage = Usage(
                            input_tokens=int(usage_raw.get("prompt_tokens", 0) or 0),
                            output_tokens=int(usage_raw.get("completion_tokens", 0) or 0),
                            model=data.get("model", self.settings.llm_model),
                        )
                        tool_calls = []
                        for call in message.get("tool_calls") or []:
                            fn = call.get("function") or {}
                            tool_calls.append(
                                {
                                    "id": call.get("id") or f"call_{len(tool_calls)}",
                                    "name": fn.get("name"),
                                    "arguments": _parse_args(fn.get("arguments")),
                                }
                            )
                        set_span_attrs(
                            current,
                            {
                                "gen_ai.usage.input_tokens": usage.input_tokens,
                                "gen_ai.usage.output_tokens": usage.output_tokens,
                                "gen_ai.response.finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
                                "app.cost_usd": round(usage.cost_usd(self.settings), 6),
                                "llm.attempt": attempt + 1,
                            },
                        )
                        return LLMMessage(
                            content=message.get("content") or "",
                            tool_calls=tool_calls,
                            usage=usage,
                            finish_reason=(data.get("choices") or [{}])[0].get("finish_reason") or "stop",
                            raw=data,
                        )

                if attempt < self.settings.llm_max_retries:
                    time.sleep(min(3.0, 0.4 * (2**attempt)))

            set_span_attrs(current, {"error.type": "LLMError"})
            raise last_error or LLMError("LLM call failed", retryable=True)

    # ------------------------------------------------------------- json mode
    def complete_json(
        self,
        messages: list[dict],
        model_cls: type[T],
        *,
        trace_id: str | None = None,
        node: str = "llm_json",
        use_json_mode: bool = True,
    ) -> JSONResult:
        attempts = 0
        usage = Usage()
        text = ""
        last_exc: Exception | None = None

        for repair in range(2):
            request_messages = list(messages)
            if repair and last_exc is not None:
                request_messages = list(messages) + [
                    {"role": "assistant", "content": text[:4000]},
                    {
                        "role": "user",
                        "content": (
                            "上一次输出无法通过 JSON Schema 校验，错误如下：\n"
                            f"{type(last_exc).__name__}: {last_exc}\n"
                            f"期望的 JSON Schema：{safe_json_dumps(model_cls.model_json_schema(), max_chars=4000)}\n"
                            "请只返回一个严格合法的 JSON 对象，不要 markdown ```，不要解释。"
                        ),
                    },
                ]
            try:
                message = self.chat(
                    request_messages,
                    response_format={"type": "json_object"} if use_json_mode else None,
                    trace_id=trace_id,
                    node=node,
                )
            except LLMError as exc:
                if use_json_mode and exc.status_code == 400:
                    # 部分兼容网关不支持 response_format，退化为提示词约束
                    message = self.chat(
                        request_messages,
                        response_format=None,
                        trace_id=trace_id,
                        node=node,
                    )
                else:
                    raise
            attempts += 1
            usage.input_tokens += message.usage.input_tokens
            usage.output_tokens += message.usage.output_tokens
            usage.model = message.usage.model or self.settings.llm_model
            text = message.content
            try:
                payload = extract_json_object(text)
                parsed = model_cls.model_validate(payload)
                return JSONResult(parsed=parsed, text=text, usage=usage, attempts=attempts)
            except (ValueError, ValidationError) as exc:
                last_exc = exc
                logger.warning("LLM JSON validation failed on attempt %s: %s", attempts, exc)

        raise LLMError(f"LLM returned invalid structured output after {attempts} attempts: {last_exc}", retryable=False)


def _parse_args(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"_raw": parsed}
    except (json.JSONDecodeError, TypeError):
        return {"_parse_error": str(value)[:500]}