"""LLM 连通性检查：只发一条最小请求，不打印 Key。

    python scripts/check_llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.config import get_settings  # noqa: E402
from app.core.llm import LLMClient, LLMError  # noqa: E402


def main() -> int:
    settings = get_settings()
    print(f"base_url = {settings.llm_base_url}")
    print(f"model    = {settings.llm_model}")
    print(f"api_key  = {'已配置' if settings.llm_enabled else '未配置'} (.env / 环境变量)")
    if not settings.llm_enabled:
        print("\nLLM_API_KEY 为空。当前会走离线规则引擎。")
        print("把 Key 写入项目根目录的 .env（不是 .env.example），再运行本脚本。")
        return 1

    client = LLMClient(settings)
    try:
        message = client.chat(
            [
                {"role": "system", "content": "You are a connectivity probe."},
                {"role": "user", "content": "只回复 OK"},
            ],
            temperature=0,
            node="connectivity_check",
        )
    except LLMError as exc:
        print(f"\nFAILED: {exc}")
        if exc.status_code in {401, 403}:
            print("提示：Key 无效或无权限。")
        elif exc.status_code == 404:
            print("提示：Base URL 或模型名可能不匹配。先查询 /models 列表，确认模型 ID。")
        elif exc.status_code == 400:
            print("提示：请求参数或模型名不被该网关接受。")
        return 1
    finally:
        client.close()

    print("\nOK")
    print(f"response = {message.content[:200]}")
    print(f"usage    = input={message.usage.input_tokens} output={message.usage.output_tokens}")
    print(f"cost_usd = {message.usage.cost_usd(settings):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())