"""运维知识库加载与检索。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.rag.hybrid import Document, HybridRetriever, ScoredDocument

logger = logging.getLogger(__name__)


class KBDoc(BaseModel):
    id: str
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    root_cause: str = "unknown"
    updated_at: str = ""


class KnowledgeBase:
    def __init__(self, docs: list[KBDoc], *, rrf_k: int = 60, path: Path | None = None):
        self.docs = docs
        self._path = path
        self._by_id = {d.id: d for d in docs}
        self.retriever = HybridRetriever(
            [
                Document(
                    id=d.id,
                    title=d.title,
                    content=d.content,
                    tags=d.tags,
                    metadata={"root_cause": d.root_cause, "updated_at": d.updated_at},
                )
                for d in docs
            ],
            rrf_k=rrf_k,
        )

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        if not path.exists():
            logger.warning("knowledge base not found: %s", path)
            return cls([])
        docs: list[KBDoc] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(KBDoc.model_validate(json.loads(line)))
                except Exception as exc:
                    logger.warning("skip invalid KB line %s: %s", line_no, exc)
        logger.info("loaded %s knowledge docs from %s", len(docs), path)
        return cls(docs, path=path)

    def search(self, query: str, top_k: int = 5) -> list[ScoredDocument]:
        return self.retriever.search(query, top_k=top_k)

    def get(self, doc_id: str) -> KBDoc | None:
        return self._by_id.get(doc_id)

    def stats(self) -> dict[str, Any]:
        return {"documents": len(self.docs), "path": str(self._path or "")}