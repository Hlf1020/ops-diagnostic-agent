"""离线可复现的混合检索：BM25 + 哈希向量 + RRF 融合 + 轻量重排。

为什么不用 embedding API：评测必须可重复、零外部依赖。生产替换点很清晰：
`_vectorize` 换成 BGE/Cohere embedding，`_rerank` 换成 BGE-reranker，
融合与 top-k 逻辑保持不变。
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9_.:/-]*", re.IGNORECASE)
_CJK_CHAR = re.compile(r"[一-鿿]")
VECTOR_DIM = 512


def tokenize(text: str) -> list[str]:
    lowered = (text or "").lower()
    tokens = _ASCII_WORD.findall(lowered)
    cjk = _CJK_CHAR.findall(lowered)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens


def _hash_index(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def embed_text(text: str, dim: int = VECTOR_DIM) -> list[float]:
    """确定性哈希 TF-IDF 向量。够用、可离线、可单测；不是语义 embedding 的替代品。"""
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * dim
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    vec = [0.0] * dim
    for token, count in counts.items():
        vec[_hash_index(token, dim)] += 1.0 + math.log(count)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Document:
    id: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return f"{self.title} {' '.join(self.tags)} {self.content}"


@dataclass
class ScoredDocument:
    document: Document
    score: float
    bm25_score: float
    vector_score: float
    rrf_score: float
    rerank_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.document.id,
            "title": self.document.title,
            "score": round(self.score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "vector_score": round(self.vector_score, 4),
            "rrf_score": round(self.rrf_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
        }


class HybridRetriever:
    def __init__(self, documents: list[Document], *, rrf_k: int = 60):
        self.documents = documents
        self.rrf_k = rrf_k
        self._corpus_tokens = [tokenize(d.searchable_text) for d in documents]
        self._bm25 = BM25Okapi(self._corpus_tokens) if documents else None
        self._doc_vectors = [embed_text(d.searchable_text) for d in documents]

    def search(self, query: str, top_k: int = 5, *, rerank_pool: int = 10) -> list[ScoredDocument]:
        if not self.documents or not query.strip():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        bm25_scores = list(self._bm25.get_scores(query_tokens)) if self._bm25 else [0.0] * len(self.documents)
        query_vec = embed_text(query)
        vector_scores = [cosine(query_vec, vec) for vec in self._doc_vectors]

        bm25_order = sorted(range(len(self.documents)), key=lambda i: bm25_scores[i], reverse=True)
        vector_order = sorted(range(len(self.documents)), key=lambda i: vector_scores[i], reverse=True)
        bm25_rank = {idx: rank + 1 for rank, idx in enumerate(bm25_order)}
        vector_rank = {idx: rank + 1 for rank, idx in enumerate(vector_order)}

        fused: dict[int, float] = {}
        for idx in range(len(self.documents)):
            rrf = 0.0
            if bm25_scores[idx] > 0:
                rrf += 1.0 / (self.rrf_k + bm25_rank[idx])
            if vector_scores[idx] > 0:
                rrf += 1.0 / (self.rrf_k + vector_rank[idx])
            fused[idx] = rrf

        fused_order = sorted(fused, key=lambda i: fused[i], reverse=True)
        pool = fused_order[: max(top_k, rerank_pool)]
        if not pool:
            return []
        max_rrf = max(fused[i] for i in pool) or 1.0

        scored: list[ScoredDocument] = []
        for idx in pool:
            doc = self.documents[idx]
            rerank = self._rerank(query, query_tokens, doc)
            score = 0.6 * (fused[idx] / max_rrf) + 0.4 * rerank
            scored.append(
                ScoredDocument(
                    document=doc,
                    score=score,
                    bm25_score=float(bm25_scores[idx]),
                    vector_score=float(vector_scores[idx]),
                    rrf_score=float(fused[idx]),
                    rerank_score=rerank,
                    bm25_rank=bm25_rank[idx],
                    vector_rank=vector_rank[idx],
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _rerank(self, query: str, query_tokens: list[str], doc: Document) -> float:
        doc_tokens = set(tokenize(doc.searchable_text))
        if not query_tokens:
            return 0.0
        query_set = set(query_tokens)
        coverage = len(query_set & doc_tokens) / len(query_set)
        title_tokens = set(tokenize(doc.title))
        title_coverage = len(query_set & title_tokens) / len(query_set)
        tag_tokens = set(tokenize(" ".join(doc.tags)))
        tag_coverage = len(query_set & tag_tokens) / len(query_set)
        phrase_bonus = 1.0 if query.strip().lower() in doc.searchable_text.lower() else 0.0
        return min(1.0, 0.5 * coverage + 0.25 * title_coverage + 0.15 * tag_coverage + 0.1 * phrase_bonus)