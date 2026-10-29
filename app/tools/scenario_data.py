"""离线演练数据：6 个典型故障场景，覆盖真实面试中最常被追问的根因类型。

所有数据固定在 2026-10-20（+08:00），便于 Golden Dataset 可复现。
真实落地时，这一层会被 Prometheus / Loki / CMDB / 变更平台的 adapter 替换；
上层 Agent、工具网关、证据链和评测代码不需要改。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.core.timeutil import CN_TZ, to_iso

INCIDENT_DATE = datetime(2026, 10, 20, tzinfo=CN_TZ)
MOCK_NOW = datetime(2026, 10, 20, 22, 0, tzinfo=CN_TZ)

KNOWN_SERVICES = [
    "payment-service",
    "order-service",
    "search-service",
    "inventory-service",
    "notification-service",
    "user-service",
]

SERVICE_ALIASES: dict[str, str] = {
    "payment": "payment-service",
    "payment-service": "payment-service",
    "支付": "payment-service",
    "支付服务": "payment-service",
    "order": "order-service",
    "order-service": "order-service",
    "订单": "order-service",
    "订单服务": "order-service",
    "search": "search-service",
    "search-service": "search-service",
    "搜索": "search-service",
    "搜索服务": "search-service",
    "inventory": "inventory-service",
    "inventory-service": "inventory-service",
    "库存": "inventory-service",
    "库存服务": "inventory-service",
    "notification": "notification-service",
    "notification-service": "notification-service",
    "通知": "notification-service",
    "通知服务": "notification-service",
    "user": "user-service",
    "user-service": "user-service",
    "用户": "user-service",
    "用户服务": "user-service",
}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    baseline: float
    peak: float
    start_offset_min: int = 1
    end_offset_min: int = 30
    precision: int = 1


@dataclass(frozen=True)
class LogTemplate:
    offset_min: int
    level: str
    logger: str
    message: str


@dataclass(frozen=True)
class ChangeTemplate:
    offset_min: int
    change_id: str
    change_type: str
    summary: str
    diff: str
    risk: str
    operator: str


@dataclass
class Incident:
    service: str
    title: str
    root_cause: str
    anomaly_start: datetime
    anomaly_end: datetime
    metric_specs: list[MetricSpec]
    logs: list[LogTemplate]
    changes: list[ChangeTemplate] = field(default_factory=list)
    primary_action: str = "none"
    action_target: str = ""
    action_diff: str = ""

    @property
    def data_start(self) -> datetime:
        return self.anomaly_start - timedelta(hours=3)

    @property
    def data_end(self) -> datetime:
        return self.anomaly_end + timedelta(hours=2)


def _t(hour: int, minute: int) -> datetime:
    return INCIDENT_DATE.replace(hour=hour, minute=minute)


INCIDENTS: dict[str, Incident] = {
    "payment-service": Incident(
        service="payment-service",
        title="支付服务 P99 延迟突增（连接池配置错误）",
        root_cause="config_change",
        anomaly_start=_t(14, 2),
        anomaly_end=_t(14, 32),
        metric_specs=[
            MetricSpec("latency_p99_ms", "ms", 210, 1780, 0, 30),
            MetricSpec("cpu_utilization_pct", "%", 52, 91, 1, 28),
            MetricSpec("gc_pause_p99_ms", "ms", 42, 612, 0, 28),
            MetricSpec("db_connection_wait_ms", "ms", 3, 2810, 1, 30),
            MetricSpec("request_rate_qps", "qps", 800, 812, 0, 30, 0),
        ],
        changes=[
            ChangeTemplate(
                -42,
                "CHG-20261020-011",
                "config",
                "调整支付超时阈值 3.0s -> 3.5s",
                "- payment.timeout_ms: 3000\n+ payment.timeout_ms: 3500",
                "low",
                "liuyang",
            ),
            ChangeTemplate(
                -10,
                "DEP-20261020-006",
                "deploy",
                "payment-service v2.4.0 常规发布（无 schema 变更）",
                "- image: payment-service:v2.3.8\n+ image: payment-service:v2.4.0",
                "low",
                "release-bot",
            ),
            ChangeTemplate(
                -1,
                "CHG-20261020-014",
                "config",
                "调整数据库连接池上限 50 -> 10",
                "- db.pool.max_size: 50\n+ db.pool.max_size: 10",
                "high",
                "zhangwei",
            ),
        ],
        logs=[
            LogTemplate(-1, "INFO", "config-watcher", "ConfigMap payment-datasource updated: db.pool.max_size 50 -> 10, applied to 12 pods"),
            LogTemplate(1, "WARN", "com.zaxxer.hikari.HikariPool", "HikariPool-1 - Pool stats (total=10, active=10, idle=0, waiting=37)"),
            LogTemplate(2, "WARN", "com.zaxxer.hikari.HikariPool", "HikariPool-1 - Connection is not available, request timed out after 30000ms."),
            LogTemplate(3, "ERROR", "payment.upstream", "upstream inventory-service call timeout: GET /api/v1/stock elapsed=30012ms"),
            LogTemplate(4, "WARN", "jvm.gc", "G1 Young Generation GC pause 612ms (heap 3.1G/4G)"),
            LogTemplate(6, "ERROR", "payment.api", "Request processing failed: DataAccessResourceFailureException: Unable to acquire JDBC Connection"),
            LogTemplate(8, "WARN", "payment.slo", "P99 latency 1780ms exceeds SLO 500ms"),
            LogTemplate(12, "INFO", "resilience4j", "Circuit breaker payment-inventory OPEN (failure rate 68%)"),
            LogTemplate(15, "WARN", "com.zaxxer.hikari.HikariPool", "HikariPool-1 - Connection is not available, request timed out after 30000ms."),
            LogTemplate(20, "INFO", "kube.hpa", "HPA evaluated cpu=91% target=70%, no scaling action (maxReplicas reached)"),
            LogTemplate(-25, "INFO", "payment.api", "GET /healthz 200 (12ms)"),
        ],
        primary_action="rollback",
        action_target="CHG-20261020-014",
        action_diff="- db.pool.max_size: 10\n+ db.pool.max_size: 50",
    ),
    "order-service": Incident(
        service="order-service",
        title="订单服务错误率突增（新版本 OOM）",
        root_cause="bad_deploy",
        anomaly_start=_t(9, 12),
        anomaly_end=_t(9, 42),
        metric_specs=[
            MetricSpec("error_rate_pct", "%", 0.3, 12.6, 0, 30),
            MetricSpec("memory_usage_pct", "%", 72, 99, 0, 28),
            MetricSpec("pod_restarts", "count", 0, 7, 2, 30, 0),
            MetricSpec("latency_p99_ms", "ms", 180, 2400, 1, 30),
            MetricSpec("request_rate_qps", "qps", 420, 418, 0, 30, 0),
        ],
        changes=[
            ChangeTemplate(
                -4,
                "DEP-20261020-003",
                "deploy",
                "order-service v1.9.0 上线优惠券 v2 功能",
                "- image: order-service:v1.8.3\n+ image: order-service:v1.9.0",
                "high",
                "release-bot",
            ),
            ChangeTemplate(
                -32,
                "CHG-20261020-007",
                "config",
                "日志级别 INFO -> DEBUG",
                "- log.level: INFO\n+ log.level: DEBUG",
                "low",
                "wangfang",
            ),
        ],
        logs=[
            LogTemplate(-4, "INFO", "deploy", "Deployment order-service updated image order-service:v1.9.0 (feature coupon-v2)"),
            LogTemplate(2, "ERROR", "order.coupon", "java.lang.NullPointerException: Cannot invoke CouponPolicy.getDiscount() because policy is null"),
            LogTemplate(3, "ERROR", "order.coupon", "Unhandled exception in CouponCalculator.calculate orderId=ORD-88213"),
            LogTemplate(5, "WARN", "jvm.memory", "Heap usage 96%, GC overhead limit approaching"),
            LogTemplate(6, "ERROR", "jvm.memory", "java.lang.OutOfMemoryError: Java heap space"),
            LogTemplate(7, "WARN", "kubelet", "Pod order-service-7d9f terminated with exit code 137 (OOMKilled)"),
            LogTemplate(9, "ERROR", "order.api", "GET /api/v1/orders 500 rate=12.6% exceeds SLO 1%"),
            LogTemplate(12, "INFO", "kube.endpoint", "Readiness probe failed, removing pod from endpoints"),
            LogTemplate(15, "ERROR", "order.coupon", "java.lang.NullPointerException: Cannot invoke CouponPolicy.getDiscount() because policy is null"),
        ],
        primary_action="rollback",
        action_target="DEP-20261020-003",
        action_diff="- image: order-service:v1.9.0\n+ image: order-service:v1.8.3",
    ),
    "search-service": Incident(
        service="search-service",
        title="搜索服务 CPU 限流（营销流量突增）",
        root_cause="traffic_surge",
        anomaly_start=_t(11, 40),
        anomaly_end=_t(12, 10),
        metric_specs=[
            MetricSpec("request_rate_qps", "qps", 1200, 5200, 0, 30, 0),
            MetricSpec("cpu_utilization_pct", "%", 45, 96, 0, 30),
            MetricSpec("cpu_throttled_pct", "%", 0.5, 38, 1, 30),
            MetricSpec("latency_p99_ms", "ms", 150, 1300, 0, 30),
        ],
        changes=[
            ChangeTemplate(
                8,
                "CHG-20261020-021",
                "config",
                "HPA maxReplicas 6 -> 10（事件中缓解措施）",
                "- autoscaling.maxReplicas: 6\n+ autoscaling.maxReplicas: 10",
                "medium",
                "autoscaler",
            )
        ],
        logs=[
            LogTemplate(1, "WARN", "kube.cpu", "CPU throttling detected: 38% of periods throttled"),
            LogTemplate(2, "WARN", "search.queue", "Request queue length 420 exceeds threshold 100"),
            LogTemplate(5, "WARN", "search.slo", "P99 latency 1300ms exceeds SLO 400ms"),
            LogTemplate(8, "INFO", "kube.hpa", "HPA scaled search-service 4 -> 6 replicas"),
            LogTemplate(10, "WARN", "kube.cpu", "CPU throttling still 31% after scale-out"),
            LogTemplate(-8, "INFO", "marketing", "Campaign 'double11-preview' traffic ramp started"),
            LogTemplate(15, "WARN", "search.queue", "Request queue length 260 exceeds threshold 100"),
        ],
        primary_action="scale",
        action_target="search-service",
        action_diff="- replicas: 6\n+ replicas: 10",
    ),
    "inventory-service": Incident(
        service="inventory-service",
        title="库存服务依赖 Redis 故障",
        root_cause="dependency_failure",
        anomaly_start=_t(16, 10),
        anomaly_end=_t(16, 40),
        metric_specs=[
            MetricSpec("error_rate_pct", "%", 0.2, 8.4, 0, 30),
            MetricSpec("dependency_error_rate_pct", "%", 0.1, 31, 0, 30),
            MetricSpec("latency_p99_ms", "ms", 120, 900, 0, 30),
            MetricSpec("request_rate_qps", "qps", 300, 295, 0, 30, 0),
        ],
        logs=[
            LogTemplate(1, "ERROR", "inventory.cache", "Redis connection refused: redis-cart:6379 (ConnectionResetError)"),
            LogTemplate(2, "ERROR", "inventory.lock", "Failed to acquire lock for cart: TimeoutError after 2000ms"),
            LogTemplate(3, "WARN", "resilience4j", "Circuit breaker redis-cart OPEN (failure rate 74%)"),
            LogTemplate(5, "ERROR", "inventory.api", "GET /api/v1/cart 503 dependency_unavailable"),
            LogTemplate(10, "ERROR", "inventory.cache", "connection pool exhausted for redis-cart"),
            LogTemplate(12, "WARN", "inventory.cache", "fallback local cache hit rate 12%"),
            LogTemplate(20, "ERROR", "inventory.cache", "Redis connection refused: redis-cart:6379 (ConnectionResetError)"),
        ],
        primary_action="none",
    ),
    "notification-service": Incident(
        service="notification-service",
        title="通知服务磁盘写满",
        root_cause="resource_exhaustion",
        anomaly_start=_t(18, 5),
        anomaly_end=_t(18, 35),
        metric_specs=[
            MetricSpec("disk_usage_pct", "%", 68, 99, 0, 30),
            MetricSpec("io_wait_pct", "%", 3, 44, 1, 30),
            MetricSpec("latency_p99_ms", "ms", 200, 1600, 0, 30),
            MetricSpec("error_rate_pct", "%", 0.1, 3.2, 2, 30),
        ],
        changes=[
            ChangeTemplate(
                -35,
                "CHG-20261020-031",
                "config",
                "日志轮转 maxFiles 10 -> 30（无关变更）",
                "- log.maxFiles: 10\n+ log.maxFiles: 30",
                "low",
                "ops-bot",
            )
        ],
        logs=[
            LogTemplate(1, "ERROR", "node.disk", "No space left on device: /var/log/app"),
            LogTemplate(2, "ERROR", "notification.audit", "Failed to write audit log: OSError 28 (No space left on device)"),
            LogTemplate(3, "WARN", "node.disk", "disk usage /data 99% (threshold 85%)"),
            LogTemplate(5, "WARN", "node.io", "io wait 44%, average queue depth 18"),
            LogTemplate(7, "ERROR", "log4j2", "Async appender queue full, dropping 1420 events"),
            LogTemplate(10, "WARN", "notification.slo", "P99 latency 1600ms exceeds SLO 600ms"),
        ],
        primary_action="cleanup",
        action_target="/data/logs",
        action_diff="- delete /data/logs/*.gz older than 7d\n- free estimated 42Gi",
    ),
    "user-service": Incident(
        service="user-service",
        title="用户服务 DNS 与网络重传",
        root_cause="network_issue",
        anomaly_start=_t(20, 0),
        anomaly_end=_t(20, 30),
        metric_specs=[
            MetricSpec("dns_error_rate_pct", "%", 0.1, 18, 0, 30),
            MetricSpec("network_retransmit_pct", "%", 0.3, 9.5, 0, 30),
            MetricSpec("latency_p99_ms", "ms", 90, 750, 0, 30),
            MetricSpec("error_rate_pct", "%", 0.1, 2.1, 0, 30),
        ],
        logs=[
            LogTemplate(1, "WARN", "dns.resolver", "DNS resolution slow: user-profile.svc.cluster.local took 4800ms"),
            LogTemplate(2, "ERROR", "dns.resolver", "lookup user-profile on 10.96.0.10:53: no such host"),
            LogTemplate(4, "WARN", "node.network", "TCP retransmit rate 9.5% on eth0"),
            LogTemplate(5, "ERROR", "user.upstream", "upstream user-profile request failed: connection reset by peer"),
            LogTemplate(8, "WARN", "user.slo", "P99 latency 750ms exceeds SLO 300ms"),
            LogTemplate(10, "ERROR", "user.api", "error rate 2.1% exceeds SLO 1%"),
        ],
        primary_action="none",
    ),
}


def get_incident(service: str) -> Incident | None:
    return INCIDENTS.get(service)


def resolve_service(text: str) -> str | None:
    lowered = text.lower()
    # 长别名优先，避免 "user" 抢先匹配 "user-service"
    for alias in sorted(SERVICE_ALIASES, key=len, reverse=True):
        if alias in lowered:
            return SERVICE_ALIASES[alias]
    return None


def _jitter(seed: str, idx: int, scale: float) -> float:
    rng = random.Random(f"{seed}:{idx}")
    return rng.uniform(-scale, scale)


def build_metric_series(incident: Incident, spec: MetricSpec) -> dict:
    start = incident.data_start
    end = incident.data_end
    step = timedelta(minutes=5)
    anomaly_start = incident.anomaly_start + timedelta(minutes=spec.start_offset_min)
    anomaly_end = incident.anomaly_start + timedelta(minutes=spec.end_offset_min)
    points: list[dict] = []
    cursor = start
    idx = 0
    while cursor <= end:
        in_anomaly = anomaly_start <= cursor <= anomaly_end
        if in_anomaly:
            value = spec.peak + _jitter(f"{incident.service}:{spec.name}", idx, max(0.5, abs(spec.peak) * 0.03))
        else:
            value = spec.baseline + _jitter(f"{incident.service}:{spec.name}", idx, max(0.2, abs(spec.baseline) * 0.06))
        if spec.unit == "count":
            value = round(max(0, value))
        else:
            value = round(max(0.0, value), spec.precision)
        points.append({"ts": to_iso(cursor), "value": value})
        cursor += step
        idx += 1

    anomaly = None
    change_pct = (spec.peak - spec.baseline) / spec.baseline * 100 if spec.baseline else 100.0
    # 变化小于 10% 视为噪声，不计入异常证据（例如支付服务 QPS 800 -> 812）
    if spec.peak != spec.baseline and (abs(change_pct) >= 10 or spec.unit == "count"):
        anomaly = {
            "metric": spec.name,
            "unit": spec.unit,
            "start": to_iso(anomaly_start),
            "end": to_iso(anomaly_end),
            "baseline": spec.baseline,
            "peak": spec.peak,
            "change_pct": round(change_pct, 1),
            "direction": "up",
        }
    return {
        "metric": spec.name,
        "unit": spec.unit,
        "service": incident.service,
        "points": points,
        "anomaly": anomaly,
    }


def build_logs(incident: Incident) -> list[dict]:
    rows = []
    for idx, tpl in enumerate(incident.logs):
        ts = incident.anomaly_start + timedelta(minutes=tpl.offset_min)
        rows.append(
            {
                "ts": to_iso(ts),
                "level": tpl.level,
                "logger": tpl.logger,
                "message": tpl.message,
                "trace_id": f"tr-{incident.service[:4]}-{abs(tpl.offset_min):02d}-{idx:02d}",
            }
        )
    rows.sort(key=lambda r: r["ts"])
    return rows


def build_changes(incident: Incident) -> list[dict]:
    rows = []
    for idx, tpl in enumerate(incident.changes):
        ts = incident.anomaly_start + timedelta(minutes=tpl.offset_min)
        rows.append(
            {
                "change_id": tpl.change_id,
                "service": incident.service,
                "ts": to_iso(ts),
                "type": tpl.change_type,
                "summary": tpl.summary,
                "diff": tpl.diff,
                "risk": tpl.risk,
                "operator": tpl.operator,
                "url": f"https://change.internal/{tpl.change_id}",
                "index": idx,
            }
        )
    rows.sort(key=lambda r: r["ts"])
    return rows


INCIDENT_LOGS = {name: build_logs(incident) for name, incident in INCIDENTS.items()}
INCIDENT_CHANGES = {name: build_changes(incident) for name, incident in INCIDENTS.items()}
INCIDENT_METRICS = {
    name: [build_metric_series(incident, spec) for spec in incident.metric_specs]
    for name, incident in INCIDENTS.items()
}