"""全局配置：所有可变行为都通过环境变量 / .env 注入，便于评测与故障注入。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "ops-diagnostic-agent"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # ---- LLM ----
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout_s: float = 45.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.1
    llm_input_price_per_mtok: float = 0.27
    llm_output_price_per_mtok: float = 1.10
    # 让 LLM 参与 critic 会增加一次调用；无 key 时自动关闭
    llm_critic_enabled: bool = True

    # ---- 存储 ----
    redis_url: str = ""
    qdrant_url: str = ""
    qdrant_path: str = ""
    session_backend: str = "auto"  # auto | memory | redis
    long_term_backend: str = "auto"  # auto | memory | qdrant
    checkpoint_backend: str = "sqlite"  # sqlite | memory
    checkpoint_db: str = "data/checkpoints.sqlite"
    qdrant_collection: str = "ops_cases"

    # ---- 业务参数 ----
    mock_now: str = "2026-10-20T22:00:00+08:00"
    time_window_pad_minutes: int = 30
    explicit_window_minutes: int = 60
    default_window_hours: int = 2
    max_retries: int = 2
    memory_top_k: int = 3
    memory_ttl_days: int = 90
    context_turn_limit: int = 4
    approval_ttl_minutes: int = 30

    # ---- 工具网关 ----
    tool_max_output_chars: int = 24_000
    tool_timeout_s: float = 8.0
    mock_fail_tools: str = ""
    mock_delay: str = ""

    # ---- 可观测 / 评测 ----
    tracing_enabled: bool = True
    trace_dir: str = "data/traces"
    audit_path: str = "data/audit/approvals.jsonl"
    kb_path: str = "data/kb/docs.jsonl"
    eval_output_dir: str = "data/eval"

    # ---- API ----
    cors_origins: str = "*"

    # 运行期由 CLI 覆盖；测试中直接构造 Settings(...)
    repository_root: Path = Field(default=BASE_DIR, exclude=True)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key.strip())

    @property
    def resolved_trace_dir(self) -> Path:
        return self._resolve(self.trace_dir)

    @property
    def resolved_audit_path(self) -> Path:
        return self._resolve(self.audit_path)

    @property
    def resolved_kb_path(self) -> Path:
        return self._resolve(self.kb_path)

    @property
    def resolved_checkpoint_db(self) -> Path:
        return self._resolve(self.checkpoint_db)

    @property
    def resolved_eval_dir(self) -> Path:
        return self._resolve(self.eval_output_dir)

    def _resolve(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.repository_root / p)

    def resolved_qdrant_path(self) -> Path | None:
        return self._resolve(self.qdrant_path) if self.qdrant_path else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()