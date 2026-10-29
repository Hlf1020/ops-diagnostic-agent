"""LangGraph checkpointer 工厂：SQLite 持久化优先，失败降级内存。

只有持久化 checkpoint 才能支撑“中断后进程重启仍可恢复审批”。
"""

from __future__ import annotations

import logging
import sqlite3

from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings

logger = logging.getLogger(__name__)


def build_checkpointer(settings: Settings):
    if settings.checkpoint_backend.lower() == "memory":
        return InMemorySaver()
    try:
        path = settings.resolved_checkpoint_db
        path.parent.mkdir(parents=True, exist_ok=True)
        # LangGraph 的 SqliteSaver 会在多个 async 任务中访问同一连接
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        from langgraph.checkpoint.sqlite import SqliteSaver

        saver = SqliteSaver(conn)
        saver.setup()
        logger.info("checkpointer: sqlite %s", path)
        return saver
    except Exception as exc:
        logger.warning("sqlite checkpointer unavailable (%s), falling back to memory", exc)
        return InMemorySaver()