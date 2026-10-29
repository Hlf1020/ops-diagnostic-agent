"""时间窗口解析：把中文运维口语映射成可审计的 [start, end]。

这些规则故意保持确定性、可单测。LLM 只负责语义分类，不负责时间计算，
避免模型把“14:00”算错却没有任何人发现。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

CN_TZ = timezone(timedelta(hours=8))

_TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3])[:：]([0-5]\d)(?!\d)")
_DATE_RE = re.compile(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})日?")
_REL_RE = re.compile(r"(?:过去|最近|近)\s*(\d+)\s*(分钟|小时|天)")
_WINDOW_RE = re.compile(r"从?\s*([01]?\d|2[0-3])[:：]([0-5]\d)\s*(?:开始|起)")


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def parse_iso(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=CN_TZ)
    return dt


def to_iso(dt: datetime) -> str:
    return parse_iso(dt).isoformat(timespec="seconds")


def fmt_cn(dt: datetime | str) -> str:
    d = parse_iso(dt)
    return d.strftime("%m-%d %H:%M:%S")


def fmt_clock(dt: datetime | str) -> str:
    d = parse_iso(dt)
    return d.strftime("%H:%M")


def parse_time_window(
    query: str,
    now: datetime | str,
    *,
    pad_minutes: int = 60,
    default_hours: int = 2,
) -> dict:
    """返回 {start, end, label, detected_by, as_of}，全部为 ISO 字符串。"""
    now_dt = parse_iso(now)
    text = query.strip()

    # 1) 显式日期 + 时刻（最高优先级）
    date_match = _DATE_RE.search(text)
    time_match = _TIME_RE.search(text)
    if time_match:
        hour, minute = int(time_match.group(1)), int(time_match.group(2))
        if date_match:
            year, month, day = (int(x) for x in date_match.groups())
            anchor = now_dt.replace(year=year, month=month, day=day)
        else:
            anchor = now_dt
        anchor = anchor.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # 用户描述的是“当天已发生”的现象；若时刻明显在未来，回退一天
        if not date_match and anchor > now_dt + timedelta(minutes=30):
            anchor -= timedelta(days=1)
        start = anchor - timedelta(minutes=pad_minutes)
        end = min(anchor + timedelta(minutes=pad_minutes), now_dt)
        if end <= start:
            end = start + timedelta(minutes=2 * pad_minutes)
        return {
            "start": to_iso(start),
            "end": to_iso(end),
            "anchor": to_iso(anchor),
            "label": f"{fmt_clock(anchor)} 前后 {pad_minutes} 分钟",
            "detected_by": "explicit_clock",
            "as_of": to_iso(now_dt),
        }

    # 2) 相对区间：过去 N 分钟/小时/天
    rel = _REL_RE.search(text)
    if rel:
        n = max(1, int(rel.group(1)))
        unit = rel.group(2)
        delta = {
            "分钟": timedelta(minutes=n),
            "小时": timedelta(hours=n),
            "天": timedelta(days=n),
        }[unit]
        start, end = now_dt - delta, now_dt
        return {
            "start": to_iso(start),
            "end": to_iso(end),
            "anchor": to_iso(end),
            "label": f"过去 {n} {unit}",
            "detected_by": "relative_range",
            "as_of": to_iso(now_dt),
        }

    # 3) 兜底：最近 N 小时
    start = now_dt - timedelta(hours=default_hours)
    return {
        "start": to_iso(start),
        "end": to_iso(now_dt),
        "anchor": to_iso(now_dt),
        "label": f"最近 {default_hours} 小时",
        "detected_by": "default",
        "as_of": to_iso(now_dt),
    }


def expand_window(window: dict, pad_minutes: int) -> tuple[datetime, datetime]:
    start = parse_iso(window["start"]) - timedelta(minutes=pad_minutes)
    end = parse_iso(window["end"]) + timedelta(minutes=pad_minutes)
    return start, end


def window_overlaps(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    a0, a1 = parse_iso(a_start), parse_iso(a_end)
    b0, b1 = parse_iso(b_start), parse_iso(b_end)
    return a0 <= b1 and b0 <= a1


def minutes_between(a: str | datetime, b: str | datetime) -> float:
    return (parse_iso(b) - parse_iso(a)).total_seconds() / 60.0