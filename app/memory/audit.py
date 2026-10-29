"""审批与高风险动作审计。JSONL 追加写，便于 crontab 归档与离线计算驳回率。"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.core.timeutil import to_iso

logger = logging.getLogger(__name__)


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event: dict) -> dict:
        entry = {
            "ts": to_iso(datetime.now(timezone.utc)),
            **event,
        }
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.exception("failed to write audit log")
        return entry

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    def approval_stats(self) -> dict:
        rows = [r for r in self.read_all() if r.get("event") == "approval_decision"]
        approved = sum(1 for r in rows if r.get("approved"))
        rejected = len(rows) - approved
        return {
            "decisions": len(rows),
            "approved": approved,
            "rejected": rejected,
            "rejection_rate": round(rejected / len(rows), 4) if rows else 0.0,
        }