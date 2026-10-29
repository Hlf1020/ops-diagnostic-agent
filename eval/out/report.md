# Ops Diagnostic Agent 评测报告

- 模式：**rules-offline**
- 用例数：24
- **Top-1 根因准确率：100.00%**
- **Top-3 根因命中率：100.00%**
- MRR：1.0000
- Faithfulness（平均）：0.853
- 首要假设被证据支撑比例：100.00%
- 意图准确率：100.00%
- 服务识别准确率：100.00%
- 工具成功率：100.00%（96 次调用）
- P95 诊断延迟：13.81 ms；平均：9.52 ms
- 平均证据数：10.6667
- 需要人工审批的比例：66.67%
- Token / 成本：0 / $0（单次 $0.0）
- 知识库：14 篇

> 离线模式未调用 LLM，token/cost 为 0；人工驳回率需在生产审批日志中统计。

## 诊断用例

| ID | 期望根因 | Top-1 | Top-3 | RR | 置信度 | 支撑度 | 工具 | 延迟(ms) |
|---|---|---|---|---|---|---|---|---|
| lat-01 | config_change | config_change | ✅ | 1.00 | 0.95 | 0.838 | 4/4 | 13.81 |
| lat-02 | config_change | config_change | ✅ | 1.00 | 0.95 | 1.0 | 4/4 | 9.37 |
| lat-03 | config_change | config_change | ✅ | 1.00 | 0.95 | 1.0 | 4/4 | 9.35 |
| lat-04 | config_change | config_change | ✅ | 1.00 | 0.95 | 1.0 | 4/4 | 9.12 |
| lat-05 | config_change | config_change | ✅ | 1.00 | 0.95 | 1.0 | 4/4 | 8.31 |
| err-01 | bad_deploy | bad_deploy | ✅ | 1.00 | 0.95 | 1.0 | 4/4 | 8.35 |
| err-02 | bad_deploy | bad_deploy | ✅ | 1.00 | 0.95 | 0.838 | 4/4 | 8.92 |
| err-03 | bad_deploy | bad_deploy | ✅ | 1.00 | 0.95 | 0.838 | 4/4 | 8.68 |
| err-04 | bad_deploy | bad_deploy | ✅ | 1.00 | 0.95 | 1.0 | 4/4 | 9.04 |
| traffic-01 | traffic_surge | traffic_surge | ✅ | 1.00 | 0.9 | 0.783 | 4/4 | 15.04 |
| traffic-02 | traffic_surge | traffic_surge | ✅ | 1.00 | 0.93 | 1.0 | 4/4 | 8.52 |
| traffic-03 | traffic_surge | traffic_surge | ✅ | 1.00 | 0.93 | 0.783 | 4/4 | 8.2 |
| traffic-04 | traffic_surge | traffic_surge | ✅ | 1.00 | 0.93 | 0.783 | 4/4 | 8.61 |
| dep-01 | dependency_failure | dependency_failure | ✅ | 1.00 | 0.91 | 0.783 | 4/4 | 10.48 |
| dep-02 | dependency_failure | dependency_failure | ✅ | 1.00 | 0.91 | 1.0 | 4/4 | 9.81 |
| dep-03 | dependency_failure | dependency_failure | ✅ | 1.00 | 0.91 | 0.783 | 4/4 | 11.01 |
| dep-04 | dependency_failure | dependency_failure | ✅ | 1.00 | 0.91 | 0.783 | 4/4 | 9.21 |
| disk-01 | resource_exhaustion | resource_exhaustion | ✅ | 1.00 | 0.93 | 0.567 | 4/4 | 8.75 |
| disk-02 | resource_exhaustion | resource_exhaustion | ✅ | 1.00 | 0.93 | 0.675 | 4/4 | 9.41 |
| disk-03 | resource_exhaustion | resource_exhaustion | ✅ | 1.00 | 0.93 | 0.675 | 4/4 | 8.98 |
| net-01 | network_issue | network_issue | ✅ | 1.00 | 0.91 | 1.0 | 4/4 | 9.05 |
| net-02 | network_issue | network_issue | ✅ | 1.00 | 0.8 | 0.783 | 4/4 | 9.11 |
| net-03 | network_issue | network_issue | ✅ | 1.00 | 0.94 | 0.783 | 4/4 | 9.49 |
| net-04 | network_issue | network_issue | ✅ | 1.00 | 0.8 | 0.783 | 4/4 | 7.95 |

## 检索评测

- Hit@3：**100.00%**；Hit@5：100.00%；MRR：0.9583

| ID | 期望文档 | Top-3 命中 | RR | 返回前三 |
|---|---|---|---|---|
| ret-01 | incident-2026-08-payment-pool, runbook-payment-latency, error-codes | ✅ | 1.00 | incident-2026-08-payment-pool, runbook-payment-latency, error-codes |
| ret-02 | runbook-payment-latency, incident-2026-08-payment-pool | ✅ | 1.00 | incident-2026-08-payment-pool, runbook-payment-latency, runbook-change-review |
| ret-03 | incident-order-oom, runbook-error-rate, error-codes | ✅ | 1.00 | incident-order-oom, runbook-error-rate, error-codes |
| ret-04 | incident-search-throttle, runbook-traffic-surge | ✅ | 1.00 | incident-search-throttle, runbook-traffic-surge, runbook-resource-disk |
| ret-05 | runbook-dependency, error-codes | ✅ | 1.00 | runbook-dependency, runbook-network-dns, error-codes |
| ret-06 | runbook-resource-disk, error-codes | ✅ | 1.00 | runbook-resource-disk, error-codes, runbook-network-dns |
| ret-07 | runbook-network-dns, error-codes | ✅ | 1.00 | runbook-network-dns, error-codes, runbook-observability |
| ret-08 | runbook-change-review | ✅ | 1.00 | runbook-change-review, runbook-payment-latency, runbook-rollback |
| ret-09 | runbook-rollback | ✅ | 1.00 | runbook-rollback, runbook-change-review, runbook-resource-disk |
| ret-10 | runbook-gc, runbook-payment-latency | ✅ | 0.50 | incident-2026-08-payment-pool, runbook-gc, runbook-payment-latency |
| ret-11 | runbook-observability | ✅ | 1.00 | runbook-observability, runbook-error-rate, runbook-rollback |
| ret-12 | incident-order-oom, runbook-error-rate | ✅ | 1.00 | incident-order-oom, runbook-error-rate, error-codes |
