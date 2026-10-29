"""短期记忆：会话 buffer + 滚动摘要 + 待确认操作元数据。

真实部署用 Redis；本地/测试用进程内实现，接口一致。每次诊断使用独立
LangGraph thread（session:turn），避免上一轮的 evidence 污染下一轮。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from app.config import Settings
from app.core.timeutil import to_iso
from app.models import UserContext

logger = logging.getLogger(__name__)


class SessionError(RuntimeError):
    pass


class SessionMismatch(SessionError):
    pass


class SessionNotFound(SessionError):
    pass


@dataclass
class Session:
    session_id: str
    user_id: str
    team_id: str
    role: str
    turn_counter: int = 0
    thread_id: str = ""
    summary: str = ""
    turns: list[dict] = field(default_factory=list)
    pending_query: str | None = None
    pending_missing_slots: list[str] = field(default_factory=list)
    open_approval: dict | None = None
    executed_actions: dict[str, dict] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def create(cls, session_id: str, user: UserContext) -> "Session":
        now = to_iso(datetime.now(timezone.utc))
        session = cls(
            session_id=session_id,
            user_id=user.user_id,
            team_id=user.team_id,
            role=user.role.value,
            created_at=now,
            updated_at=now,
        )
        session.start_new_turn()
        return session

    def start_new_turn(self) -> str:
        self.turn_counter += 1
        self.thread_id = f"{self.session_id}:t{self.turn_counter}"
        self.updated_at = to_iso(datetime.now(timezone.utc))
        return self.thread_id

    def append_turn(self, role: str, content: str, meta: dict | None = None) -> None:
        self.turns.append(
            {
                "role": role,
                "content": content[:3000],
                "ts": to_iso(datetime.now(timezone.utc)),
                "meta": meta or {},
            }
        )
        if len(self.turns) > 20:
            self.turns = self.turns[-20:]
        self.updated_at = to_iso(datetime.now(timezone.utc))

    def recent_turns(self, limit: int) -> list[dict]:
        return self.turns[-limit:]

    def set_pending(self, query: str, missing: list[str]) -> None:
        self.pending_query = query
        self.pending_missing_slots = missing
        self.updated_at = to_iso(datetime.now(timezone.utc))

    def pop_pending(self) -> tuple[str | None, list[str]]:
        query, missing = self.pending_query, self.pending_missing_slots
        self.pending_query = None
        self.pending_missing_slots = []
        return query, missing

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class SessionStore(Protocol):
    def get_or_create(self, session_id: str, user: UserContext) -> Session: ...
    def get(self, session_id: str) -> Session | None: ...
    def save(self, session: Session) -> None: ...
    def record_executed(self, session: Session, key: str, result: dict) -> None: ...
    def backend_name(self) -> str: ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def get_or_create(self, session_id: str, user: UserContext) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session.create(session_id, user)
                self._sessions[session_id] = session
                return session
            self._verify(session, user)
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def save(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def record_executed(self, session: Session, key: str, result: dict) -> None:
        with self._lock:
            session.executed_actions[key] = result
            if len(session.executed_actions) > 50:
                session.executed_actions = dict(list(session.executed_actions.items())[-50:])
            self.save(session)

    @staticmethod
    def _verify(session: Session, user: UserContext) -> None:
        if session.user_id != user.user_id or session.team_id != user.team_id:
            raise SessionMismatch("session belongs to a different user or team")
        if session.role != user.role.value:
            raise SessionMismatch(
                "role changed within an existing session; start a new session to switch roles"
            )

    def backend_name(self) -> str:
        return "memory"


class RedisSessionStore(InMemorySessionStore):
    """Redis 持久化 + 本地 LRU 缓存；读取失败时降级为当前进程缓存。"""

    def __init__(self, settings: Settings):
        super().__init__()
        import redis  # 延迟导入，memory 模式不需要

        self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self._redis.ping()
        self._ttl_s = 7 * 24 * 3600

    def _key(self, session_id: str) -> str:
        return f"opsagent:session:{session_id}"

    def get_or_create(self, session_id: str, user: UserContext) -> Session:
        session = self.get(session_id)
        if session is None:
            session = Session.create(session_id, user)
            self.save(session)
            return session
        self._verify(session, user)
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            cached = self._sessions.get(session_id)
            if cached is not None:
                return cached
        raw = self._redis.get(self._key(session_id))
        if not raw:
            return None
        session = Session.from_dict(json.loads(raw))
        with self._lock:
            self._sessions[session_id] = session
        return session

    def save(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.session_id] = session
        self._redis.set(self._key(session.session_id), json.dumps(session.to_dict(), ensure_ascii=False), ex=self._ttl_s)

    def backend_name(self) -> str:
        return "redis"


def build_session_store(settings: Settings) -> SessionStore:
    backend = settings.session_backend.lower()
    if backend == "memory":
        return InMemorySessionStore()
    if backend in {"auto", "redis"} and settings.redis_url:
        try:
            store = RedisSessionStore(settings)
            logger.info("session store: redis")
            return store
        except Exception as exc:
            if backend == "redis":
                raise
            logger.warning("redis unavailable (%s), falling back to memory session store", exc)
    return InMemorySessionStore()


def session_context(session: Session, limit: int) -> list[dict]:
    turns = session.recent_turns(limit)
    if not session.summary:
        return turns
    return [{"role": "system", "content": f"此前会话摘要：{session.summary}"}] + turns


def utc_now_iso() -> str:
    return to_iso(datetime.now(timezone.utc))


def is_expired(iso_ts: str, ttl_minutes: int) -> bool:
    try:
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return True
    return datetime.now(timezone.utc) > ts + timedelta(minutes=ttl_minutes)