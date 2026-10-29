"""工具注册表：Schema、权限、超时、输出限额与 Handler 绑定。

新增真实工具只需在此注册；编排层、评测与权限模型不需要改。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

from app.config import Settings
from app.core.timeutil import parse_iso, to_iso
from app.rag.knowledge import KnowledgeBase
from app.tools import scenario_data


class ToolInput(BaseModel):
    pass


class QueryLogsInput(ToolInput):
    service: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    start: datetime
    end: datetime
    levels: list[Literal["DEBUG", "INFO", "WARN", "ERROR"]] = Field(
        default_factory=lambda: ["WARN", "ERROR"], max_length=4
    )
    pattern: str | None = Field(default=None, max_length=200)
    limit: int = Field(default=60, ge=1, le=200)

    @model_validator(mode="after")
    def _check_window(self):
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if self.end - self.start > timedelta(hours=24):
            raise ValueError("window must not exceed 24 hours")
        return self


class QueryMetricsInput(ToolInput):
    service: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    start: datetime
    end: datetime
    metric_names: list[str] | None = Field(default=None, max_length=12)
    step_s: int = Field(default=300, ge=60, le=3600)

    @model_validator(mode="after")
    def _check_window(self):
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if self.end - self.start > timedelta(hours=24):
            raise ValueError("window must not exceed 24 hours")
        return self


class GetChangesInput(ToolInput):
    service: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _check_window(self):
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if self.end - self.start > timedelta(hours=48):
            raise ValueError("window must not exceed 48 hours")
        return self


class SearchKnowledgeInput(ToolInput):
    query: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class ExecuteRemediationInput(ToolInput):
    action: Literal["rollback", "restart", "scale", "cleanup"]
    service: str = Field(..., min_length=2, max_length=64)
    target: str = Field(..., min_length=1, max_length=200)
    command: str = Field(..., min_length=3, max_length=1000)
    diff: str = Field(default="", max_length=4000)
    idempotency_key: str = Field(..., min_length=8, max_length=128)
    approved_by: str = Field(..., min_length=1, max_length=64)
    approval_note: str = Field(default="", max_length=500)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_model: type[ToolInput]
    required_permission: str
    handler: Callable[[ToolInput], dict]
    max_output_chars: int
    timeout_s: float
    tags: list[str]

    def json_schema(self) -> dict:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return schema

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]):
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def names(self) -> list[str]:
        return list(self._specs)


# ----------------------------------------------------------------- handlers
def _window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    return parse_iso(start), parse_iso(end)


_LOG_CATEGORIES: list[tuple[str, str, str]] = [
    ("connection_pool", "连接池耗尽", r"hikaripool|connection pool|acquire jdbc|waiting"),
    ("timeout", "调用超时", r"timed out|timeout|elapsed=\d+ms"),
    ("oom", "内存溢出/OOM", r"outofmemory|oomkilled|exit code 137|heap space"),
    ("gc", "GC 停顿", r"gc pause|g1 |gc overhead"),
    ("disk", "磁盘/IO", r"no space left|oserror 28|disk usage|io wait|appender queue"),
    ("dns", "DNS 解析", r"no such host|dns resolution|lookup .*:53"),
    ("network", "网络异常", r"retransmit|connection reset|packet loss"),
    ("dependency", "依赖故障", r"connection refused|circuit breaker|dependency_unavailable|redis"),
    ("deploy", "发布/配置", r"deployment|configmap|rolling update|image:"),
    ("error", "应用错误", r"nullpointer|exception|error rate|500 rate"),
]


def _query_logs(settings: Settings):
    def handler(args: QueryLogsInput) -> dict:
        incident = scenario_data.get_incident(args.service)
        if incident is None:
            return {
                "service": args.service,
                "window": {"start": to_iso(args.start), "end": to_iso(args.end)},
                "incident_found": False,
                "total": 0,
                "entries": [],
                "groups": [],
                "note": "no telemetry source registered for this service",
            }
        start, end = _window(args.start, args.end)
        rows = [
            row
            for row in scenario_data.INCIDENT_LOGS[args.service]
            if start <= parse_iso(row["ts"]) <= end
            and row["level"] in args.levels
            and (not args.pattern or args.pattern.lower() in row["message"].lower())
        ]
        total = len(rows)
        rows = rows[: args.limit]

        groups: dict[str, dict] = {}
        import re

        for row in rows:
            category = "other"
            label = "其他日志"
            for cat, cat_label, pattern in _LOG_CATEGORIES:
                if re.search(pattern, row["message"], re.IGNORECASE):
                    category, label = cat, cat_label
                    break
            group = groups.setdefault(
                category,
                {"category": category, "label": label, "count": 0, "sample": row["message"], "first_ts": row["ts"]},
            )
            group["count"] += 1
            if row["ts"] < group["first_ts"]:
                group["first_ts"] = row["ts"]
                group["sample"] = row["message"]
        return {
            "service": args.service,
            "window": {"start": to_iso(start), "end": to_iso(end)},
            "incident_found": True,
            "total": total,
            "entries": rows,
            "groups": sorted(groups.values(), key=lambda g: (-g["count"], g["first_ts"])),
            "filters": {"levels": args.levels, "pattern": args.pattern},
        }

    return handler


def _query_metrics(settings: Settings):
    def handler(args: QueryMetricsInput) -> dict:
        incident = scenario_data.get_incident(args.service)
        if incident is None:
            return {
                "service": args.service,
                "incident_found": False,
                "metrics": [],
                "anomalies": [],
                "note": "no metrics backend registered for this service",
            }
        start, end = _window(args.start, args.end)
        requested = set(args.metric_names or [m["metric"] for m in scenario_data.INCIDENT_METRICS[args.service]])
        metrics = []
        anomalies = []
        for series in scenario_data.INCIDENT_METRICS[args.service]:
            if series["metric"] not in requested:
                continue
            points = [p for p in series["points"] if start <= parse_iso(p["ts"]) <= end]
            item = {
                "metric": series["metric"],
                "unit": series["unit"],
                "points": points,
                "anomaly": series["anomaly"],
            }
            metrics.append(item)
            anomaly = series["anomaly"]
            if anomaly and parse_iso(anomaly["start"]) <= end and parse_iso(anomaly["end"]) >= start:
                anomalies.append(anomaly)
        return {
            "service": args.service,
            "incident_found": True,
            "window": {"start": to_iso(start), "end": to_iso(end)},
            "metrics": metrics,
            "anomalies": anomalies,
            "note": "metrics are 5-minute aggregates (UTC+8)",
        }

    return handler


def _get_changes(settings: Settings):
    def handler(args: GetChangesInput) -> dict:
        incident = scenario_data.get_incident(args.service)
        start, end = _window(args.start, args.end)
        if incident is None:
            return {
                "service": args.service,
                "incident_found": False,
                "changes": [],
                "total": 0,
                "note": "no change source registered for this service",
            }
        rows = [
            row
            for row in scenario_data.INCIDENT_CHANGES[args.service]
            if start <= parse_iso(row["ts"]) <= end
        ]
        return {
            "service": args.service,
            "incident_found": True,
            "window": {"start": to_iso(start), "end": to_iso(end)},
            "changes": rows,
            "total": len(rows),
        }

    return handler


def _search_knowledge(kb: KnowledgeBase):
    def handler(args: SearchKnowledgeInput) -> dict:
        results = kb.search(args.query, top_k=args.top_k)
        return {
            "query": args.query,
            "total": len(results),
            "results": [
                {
                    **item.as_dict(),
                    "title": item.document.title,
                    "content": item.document.content,
                    "tags": item.document.tags,
                    "root_cause": item.document.metadata.get("root_cause"),
                }
                for item in results
            ],
        }

    return handler


def _execute_remediation(settings: Settings):
    def handler(args: ExecuteRemediationInput) -> dict:
        # 这个 mock handler 永远不会触碰真实系统。替换 adapter 时必须保留
        # 审批、幂等键、审计和最小权限四个约束。
        return {
            "status": "simulated_success",
            "simulation": True,
            "action": args.action,
            "service": args.service,
            "target": args.target,
            "command": args.command,
            "diff": args.diff,
            "idempotency_key": args.idempotency_key,
            "approved_by": args.approved_by,
            "message": "演练环境：命令已通过网关校验，未对真实基础设施执行。",
        }

    return handler


def build_registry(settings: Settings, kb: KnowledgeBase) -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="query_logs",
                description=(
                    "查询指定服务在时间窗口内的日志。返回 ERROR/WARN 明细、按故障类别聚合的 groups，"
                    "以及日志中是否出现连接池、OOM、磁盘、DNS 等关键模式。"
                ),
                input_model=QueryLogsInput,
                required_permission="logs:read",
                handler=_query_logs(settings),
                max_output_chars=settings.tool_max_output_chars,
                timeout_s=settings.tool_timeout_s,
                tags=["log", "observability"],
            ),
            ToolSpec(
                name="query_metrics",
                description=(
                    "查询服务的时序指标（延迟、错误率、CPU、内存、GC、依赖错误率、磁盘等），"
                    "返回时间序列和检测到的异常区间。"
                ),
                input_model=QueryMetricsInput,
                required_permission="metrics:read",
                handler=_query_metrics(settings),
                max_output_chars=settings.tool_max_output_chars,
                timeout_s=settings.tool_timeout_s,
                tags=["metric", "observability"],
            ),
            ToolSpec(
                name="get_changes",
                description="查询服务在时间窗口内的部署与配置变更记录，包含操作人、风险等级和 diff。",
                input_model=GetChangesInput,
                required_permission="changes:read",
                handler=_get_changes(settings),
                max_output_chars=settings.tool_max_output_chars,
                timeout_s=settings.tool_timeout_s,
                tags=["change", "cmdb"],
            ),
            ToolSpec(
                name="search_knowledge",
                description="混合检索运维知识库（BM25 + 向量 + RRF + 重排），返回 runbook 与历史案例。",
                input_model=SearchKnowledgeInput,
                required_permission="kb:read",
                handler=_search_knowledge(kb),
                max_output_chars=settings.tool_max_output_chars,
                timeout_s=settings.tool_timeout_s,
                tags=["knowledge", "rag"],
            ),
            ToolSpec(
                name="execute_remediation",
                description=(
                    "执行高风险修复动作（回滚/重启/扩容/清理）。必须经过人工审批并携带幂等键。"
                    "只读角色不可见。"
                ),
                input_model=ExecuteRemediationInput,
                required_permission="remediation:write",
                handler=_execute_remediation(settings),
                max_output_chars=settings.tool_max_output_chars,
                timeout_s=max(settings.tool_timeout_s, 15.0),
                tags=["action", "high-risk"],
            ),
        ]
    )


def schema_example(spec: ToolSpec) -> str:
    return json.dumps(spec.json_schema(), ensure_ascii=False, indent=2)[:1500]