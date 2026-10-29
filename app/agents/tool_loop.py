"""通用 LLM 工具循环（子 Agent 共用）。

约束：
- 工具白名单由编排层给定，LLM 不能自行扩大；
- 最多 3 步，避免无限工具调用；
- 任何 LLM 异常都返回已收集的执行结果，交由上层决定是否降级到规则计划。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config import Settings
from app.core.llm import LLMClient, LLMError, Usage
from app.core.tracing import span
from app.models import UserContext
from app.tools.gateway import ToolGateway, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolLoopOutcome:
    executions: list[ToolResult] = field(default_factory=list)
    final_text: str = ""
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    used_llm: bool = False


def run_tool_loop(
    *,
    llm: LLMClient,
    gateway: ToolGateway,
    user: UserContext,
    system_prompt: str,
    user_prompt: str,
    allowed_tools: set[str],
    trace_id: str | None,
    node: str,
    settings: Settings,
    max_steps: int = 3,
) -> ToolLoopOutcome:
    outcome = ToolLoopOutcome()
    if not llm.enabled:
        outcome.error = "llm_disabled"
        return outcome

    tools = gateway.openai_tools(user.role, allowed=allowed_tools)
    if not tools:
        outcome.error = "no_tools_available"
        return outcome

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    with span(f"agent.{node}", trace_id=trace_id, attrs={"app.allowed_tools": sorted(allowed_tools)}):
        for step in range(max_steps):
            try:
                message = llm.chat(
                    messages,
                    tools=tools,
                    trace_id=trace_id,
                    node=f"{node}.step{step + 1}",
                )
            except LLMError as exc:
                outcome.error = f"LLMError: {exc}"
                logger.warning("%s tool loop LLM failure: %s", node, exc)
                return outcome

            outcome.used_llm = True
            outcome.usage.input_tokens += message.usage.input_tokens
            outcome.usage.output_tokens += message.usage.output_tokens
            outcome.usage.model = message.usage.model or outcome.usage.model
            outcome.final_text = message.content or outcome.final_text

            if not message.tool_calls:
                return outcome

            assistant_message: dict = {"role": "assistant", "content": message.content or ""}
            assistant_message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
            messages.append(assistant_message)

            for call in message.tool_calls:
                name = call.get("name") or ""
                args = call.get("arguments") or {}
                if name not in allowed_tools:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(
                                {"error": f"tool {name} is not allowed for this sub-agent"},
                                ensure_ascii=False,
                            ),
                        }
                    )
                    continue
                result = gateway.call(name, args, user, trace_id=trace_id, node=node)
                outcome.executions.append(result)
                tool_payload = (
                    result.output
                    if result.ok
                    else {"error": result.error, "hint": "修正参数后可以重试；不要编造结果"}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(tool_payload, ensure_ascii=False)[:7000],
                    }
                )

        return outcome


def usage_state(outcome: ToolLoopOutcome, settings: Settings, node: str) -> list[dict]:
    if outcome.usage.total_tokens == 0 and not outcome.used_llm:
        return []
    return [{**outcome.usage.as_dict(settings), "node": node}]