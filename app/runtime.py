"""组合根：把所有适配器装配成可运行的 Runtime。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.core.llm import LLMClient
from app.core.tracing import TraceStore, init_tracing
from app.graph.build import build_graph
from app.graph.nodes import NodeDeps, Nodes
from app.memory.audit import AuditLog
from app.memory.checkpoint import build_checkpointer
from app.memory.long_term import LongTermMemory, build_long_term_memory
from app.memory.session import SessionStore, build_session_store
from app.rag.knowledge import KnowledgeBase
from app.tools.gateway import ToolGateway
from app.tools.registry import build_registry

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    llm: LLMClient
    kb: KnowledgeBase
    gateway: ToolGateway
    memory: LongTermMemory
    sessions: SessionStore
    audit: AuditLog
    traces: TraceStore
    graph: object

    def close(self) -> None:
        self.llm.close()
        close_memory = getattr(self.memory, "close", None)
        if callable(close_memory):
            close_memory()


def build_runtime(settings: Settings | None = None) -> Runtime:
    settings = settings or get_settings()
    init_tracing(settings)
    kb = KnowledgeBase.load(settings.resolved_kb_path)
    gateway = ToolGateway(build_registry(settings, kb), settings)
    memory = build_long_term_memory(settings)
    sessions = build_session_store(settings)
    audit = AuditLog(settings.resolved_audit_path)
    checkpointer = build_checkpointer(settings)
    nodes = Nodes(
        NodeDeps(
            settings=settings,
            llm=LLMClient(settings),
            gateway=gateway,
            memory=memory,
            sessions=sessions,
            audit=audit,
        )
    )
    graph = build_graph(nodes, checkpointer=checkpointer)
    traces = TraceStore(settings.resolved_trace_dir)
    logger.info(
        "runtime ready env=%s llm=%s sessions=%s memory=%s",
        settings.repository_root,
        settings.llm_model if settings.llm_enabled else "offline-fallback",
        sessions.backend_name(),
        memory.backend_name(),
    )
    return Runtime(
        settings=settings,
        llm=nodes.deps.llm,
        kb=kb,
        gateway=gateway,
        memory=memory,
        sessions=sessions,
        audit=audit,
        traces=traces,
        graph=graph,
    )