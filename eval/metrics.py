"""评测指标实现。纯函数，便于单测。"""

from __future__ import annotations

import statistics


def reciprocal_rank(ranked_ids: list[str], expected_ids: list[str]) -> float:
    for idx, item in enumerate(ranked_ids, start=1):
        if item in expected_ids:
            return 1.0 / idx
    return 0.0


def hit_at_k(ranked_ids: list[str], expected_ids: list[str], k: int = 3) -> bool:
    return any(item in expected_ids for item in ranked_ids[:k])


def top1_accuracy(ranked_ids: list[str], expected_id: str) -> bool:
    return bool(ranked_ids) and ranked_ids[0] == expected_id


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[index], 2)


def mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def evaluate_ranked(records: list[dict], *, expected_key: str = "expected_top1", k: int = 3) -> dict:
    """records: [{ranked: [labels], expected: label}]"""
    top1 = [r["ranked"][0] == r["expected"] for r in records if r.get("ranked")]
    topk = [r["expected"] in (r.get("ranked") or [])[:k] for r in records]
    rr = [reciprocal_rank(r.get("ranked") or [], [r["expected"]]) for r in records]
    return {
        "count": len(records),
        "top1_accuracy": mean([1.0 if x else 0.0 for x in top1]),
        f"top{k}_accuracy": mean([1.0 if x else 0.0 for x in topk]),
        "mrr": mean(rr),
    }