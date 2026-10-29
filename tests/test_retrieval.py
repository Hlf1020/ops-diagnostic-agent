from __future__ import annotations

from pathlib import Path

from app.rag.hybrid import HybridRetriever, Document, tokenize
from app.rag.knowledge import KnowledgeBase

ROOT = Path(__file__).resolve().parent.parent


def test_tokenize_mixes_ascii_and_chinese_bigrams():
    tokens = tokenize("HikariPool 连接池")
    assert "hikaripool" in tokens
    assert "连" in tokens
    assert "连接" in tokens


def test_hybrid_retrieval_hits_expected_docs():
    kb = KnowledgeBase.load(ROOT / "data" / "kb" / "docs.jsonl")
    results = kb.search("HikariPool Connection is not available 连接池 超时", top_k=3)
    ids = [r.document.id for r in results]
    assert "incident-2026-08-payment-pool" in ids
    assert all(r.rrf_score >= 0 for r in results)
    assert results == sorted(results, key=lambda r: r.score, reverse=True)


def test_rrf_prefers_documents_ranked_by_both_retrievers():
    docs = [
        Document(id="a", title="Redis connection refused", content="依赖故障 熔断"),
        Document(id="b", title="完全无关", content="天气 预报"),
        Document(id="c", title="Redis 熔断", content="connection refused dependency"),
    ]
    retriever = HybridRetriever(docs)
    results = retriever.search("redis connection refused 熔断", top_k=2)
    assert results[0].document.id in {"a", "c"}
    assert results[0].rerank_score > 0


def test_empty_query_returns_empty():
    retriever = HybridRetriever([Document(id="a", title="x", content="y")])
    assert retriever.search("   ") == []