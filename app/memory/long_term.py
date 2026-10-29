"""长期记忆：按 team namespace 隔离的诊断案例库。

能力：
- 写入时去重（同症状同结论只累加 seen_count）；
- 冲突审计（同症状不同结论保留双向引用，不静默覆盖）；
- TTL 淘汰；
- 检索只返回 top-k 摘要与结论，不把全文塞进 prompt。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid as uuid_lib
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from app.config import Settings
from app.core.text import normalize_symptom, short_hash
from app.core.timeutil import parse_iso, to_iso
from app.rag.hybrid import Document, HybridRetriever, embed_text, tokenize

logger = logging.getLogger(__name__)


class LongTermMemory(Protocol):
    def add_case(self, case: dict) -> dict: ...
    def search(self, query: str, team_id: str, top_k: int = 3) -> list[dict]: ...
    def stats(self) -> dict: ...
    def backend_name(self) -> str: ...


def make_case(
    *,
    team_id: str,
    user_id: str,
    service: str | None,
    symptom: str,
    intent: str,
    root_cause: str,
    conclusion: str,
    evidence_sources: list[str],
    tools_called: list[str],
    human_feedback: str = "",
    action: str = "none",
    approved: bool | None = None,
    ttl_days: int = 90,
) -> dict:
    now = to_iso(datetime.now(timezone.utc))
    normalized = normalize_symptom(symptom)
    return {
        "case_id": f"case-{short_hash(f'{team_id}:{normalized}:{root_cause}:{now}:{uuid_lib.uuid4()}', 16)}",
        "team_id": team_id,
        "user_id": user_id,
        "service": service,
        "symptom": symptom[:1000],
        "normalized_symptom": normalized,
        "intent": intent,
        "root_cause": root_cause,
        "conclusion": conclusion[:2000],
        "evidence_sources": sorted(set(evidence_sources)),
        "tools_called": sorted(set(tools_called)),
        "human_feedback": human_feedback[:1000],
        "action": action,
        "approved": approved,
        "created_at": now,
        "last_seen": now,
        "ttl_days": ttl_days,
        "seen_count": 1,
        "superseded_by": None,
        "conflicts": [],
    }


def _expired(case: dict, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    try:
        created = parse_iso(case.get("created_at", ""))
    except Exception:
        return True
    return now > created + timedelta(days=int(case.get("ttl_days", 90)))


class InMemoryLongTermMemory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cases: dict[str, dict] = {}
        self._lock = threading.RLock()

    def add_case(self, case: dict) -> dict:
        with self._lock:
            self._prune_locked()
            same = self._find_same_symptom(case)
            if same:
                if same.get("root_cause") == case.get("root_cause"):
                    same["seen_count"] = int(same.get("seen_count", 1)) + 1
                    same["last_seen"] = case.get("created_at")
                    if case.get("human_feedback"):
                        same["human_feedback"] = case["human_feedback"]
                    if case.get("approved") is not None:
                        same["approved"] = case.get("approved")
                    return {"deduped": True, "case_id": same["case_id"], "case": same}
                case["conflicts"] = [
                    {
                        "previous_case_id": same["case_id"],
                        "previous_root_cause": same.get("root_cause"),
                        "detected_at": case.get("created_at"),
                        "note": "same symptom, different conclusion; previous case retained for audit",
                    }
                ]
                same["superseded_by"] = case["case_id"]
                same["last_seen"] = case.get("created_at")
            self._cases[case["case_id"]] = case
            self._prune_locked()  # 拒绝写入已过 TTL 的记录
            return {"deduped": False, "case_id": case["case_id"], "case": case}

    def _find_same_symptom(self, case: dict) -> dict | None:
        for existing in self._cases.values():
            if existing.get("superseded_by"):
                continue
            if (
                existing.get("team_id") == case.get("team_id")
                and existing.get("service") == case.get("service")
                and existing.get("normalized_symptom") == case.get("normalized_symptom")
            ):
                return existing
        return None

    def search(self, query: str, team_id: str, top_k: int = 3) -> list[dict]:
        with self._lock:
            self._prune_locked()
            cases = [
                c
                for c in self._cases.values()
                if c.get("team_id") == team_id and not c.get("superseded_by")
            ]
        if not cases:
            return []
        docs = [
            Document(
                id=c["case_id"],
                title=f"{c.get('service') or 'unknown'} {c.get('symptom', '')[:80]}",
                content=f"{c.get('symptom', '')}\n结论：{c.get('conclusion', '')}\n根因：{c.get('root_cause')}",
                tags=[c.get("root_cause", "unknown"), c.get("intent", "unknown")],
                metadata=c,
            )
            for c in cases
        ]
        results = HybridRetriever(docs).search(query, top_k=top_k)
        out = []
        for item in results:
            case = dict(item.document.metadata)
            case["similarity"] = round(item.score, 4)
            case["retrieval"] = item.as_dict()
            out.append(case)
        return out

    def _prune_locked(self) -> None:
        expired = [cid for cid, c in self._cases.items() if _expired(c)]
        for cid in expired:
            del self._cases[cid]

    def stats(self) -> dict:
        with self._lock:
            return {
                "backend": "memory",
                "cases": len(self._cases),
                "active": sum(1 for c in self._cases.values() if not c.get("superseded_by")),
            }

    def backend_name(self) -> str:
        return "memory"


class QdrantLongTermMemory(InMemoryLongTermMemory):
    """Qdrant 持久化 + 向量召回，再用轻量词法覆盖度重排。"""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        from qdrant_client import QdrantClient

        if settings.qdrant_url:
            self._client = QdrantClient(url=settings.qdrant_url)
        else:
            path = settings.resolved_qdrant_path()
            assert path is not None
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))
        self.collection = settings.qdrant_collection
        last_error: Exception | None = None
        for attempt in range(30):
            try:
                self._ensure_collection()
                self._load_all()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt == 29:
                    break
                time.sleep(1)
        if last_error is not None:
            raise last_error

    def _ensure_collection(self) -> None:
        from qdrant_client.http import models as qm

        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=512, distance=qm.Distance.COSINE),
            )
        for field_name in ("team_id", "normalized_symptom", "root_cause"):
            try:
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def _load_all(self) -> None:
        from qdrant_client.http import models as qm

        offset = None
        loaded = 0
        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=qm.Filter(),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                if payload.get("case_id"):
                    self._cases[payload["case_id"]] = payload
                    loaded += 1
            if offset is None:
                break
        logger.info("loaded %s cases from qdrant collection %s", loaded, self.collection)

    def _point_id(self, case_id: str) -> str:
        return str(uuid_lib.uuid5(uuid_lib.NAMESPACE_URL, f"opsagent:{case_id}"))

    def _upsert(self, case: dict) -> None:
        from qdrant_client.http import models as qm

        self._client.upsert(
            collection_name=self.collection,
            points=[
                qm.PointStruct(
                    id=self._point_id(case["case_id"]),
                    vector=embed_text(f"{case.get('symptom', '')} {case.get('conclusion', '')}"),
                    payload=case,
                )
            ],
        )

    def add_case(self, case: dict) -> dict:
        result = super().add_case(case)
        stored = result["case"]
        if result["deduped"]:
            self._upsert(stored)
            return result
        # 冲突场景：旧 case 的 superseded_by 也要落盘
        for existing in self._cases.values():
            if existing.get("superseded_by") == stored["case_id"]:
                self._upsert(existing)
        self._upsert(stored)
        return result

    def search(self, query: str, team_id: str, top_k: int = 3) -> list[dict]:
        from qdrant_client.http import models as qm

        self._prune_local_and_remote(team_id)
        try:
            response = self._client.query_points(
                collection_name=self.collection,
                query=embed_text(query),
                query_filter=qm.Filter(
                    must=[qm.FieldCondition(key="team_id", match=qm.MatchValue(value=team_id))]
                ),
                limit=max(20, top_k * 5),
                with_payload=True,
            )
            points = response.points
        except Exception as exc:
            logger.warning("qdrant query failed, fallback to local index: %s", exc)
            return super().search(query, team_id, top_k)

        query_tokens = set(tokenize(query))
        scored = []
        for point in points:
            payload = dict(point.payload or {})
            if payload.get("superseded_by") or _expired(payload):
                continue
            text_tokens = set(tokenize(f"{payload.get('symptom', '')} {payload.get('conclusion', '')}"))
            coverage = len(query_tokens & text_tokens) / len(query_tokens) if query_tokens else 0.0
            score = 0.75 * float(point.score) + 0.25 * coverage
            payload["similarity"] = round(score, 4)
            payload["retrieval"] = {"vector_score": round(float(point.score), 4), "lexical_coverage": round(coverage, 4)}
            scored.append((score, payload))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def _prune_local_and_remote(self, team_id: str) -> None:
        from qdrant_client.http import models as qm

        now = datetime.now(timezone.utc)
        expired_ids = []
        for case_id, case in list(self._cases.items()):
            if case.get("team_id") == team_id and _expired(case, now):
                expired_ids.append(self._point_id(case_id))
                del self._cases[case_id]
        if expired_ids:
            try:
                self._client.delete(
                    collection_name=self.collection,
                    points_selector=qm.PointIdsList(points=expired_ids),
                )
            except Exception as exc:
                logger.warning("qdrant TTL delete failed: %s", exc)

    def stats(self) -> dict:
        stats = super().stats()
        stats["backend"] = "qdrant"
        stats["collection"] = self.collection
        return stats

    def backend_name(self) -> str:
        return "qdrant"

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


def build_long_term_memory(settings: Settings) -> LongTermMemory:
    backend = settings.long_term_backend.lower()
    if backend == "memory":
        return InMemoryLongTermMemory(settings)
    if backend in {"auto", "qdrant"} and (settings.qdrant_url or settings.qdrant_path):
        try:
            memory = QdrantLongTermMemory(settings)
            logger.info("long-term memory: qdrant")
            return memory
        except Exception as exc:
            if backend == "qdrant":
                raise
            logger.warning("qdrant unavailable (%s), falling back to memory", exc)
    return InMemoryLongTermMemory(settings)