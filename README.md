# 企业级运维诊断 Agent（Ops Diagnostic Agent）

面向企业内部云原生 / 微服务环境的**可审计、可评估的故障初筛与根因分析 Agent**。用户用自然语言描述异常，Agent 自主调用日志、指标、变更与知识库工具，生成带证据链的候选根因，在高风险动作前暂停等待人工确认，并把全过程记录为可回放的 Trace。

> 一句话：把“看起来像 AI 在猜”的运维助手，做成“每条结论都能指回原始证据、每个高风险动作都要人签字、每次运行都能量化评测”的工程系统。

---

## 1. 快速开始

本机已验证环境：`D:\miniconda\envs\agent1\python.exe`（Python 3.13）。

```bash
cd D:\课程\agent-1
D:\miniconda\envs\agent1\python.exe -m pip install -r requirements.txt
```

### 1.1 一键自检（推荐先跑）

```bash
D:\miniconda\envs\agent1\python.exe scripts/selfcheck.py
```

覆盖 6 个故障场景 + 澄清 + 权限降级 + 工具故障，不调用 LLM、不写项目 `data/`。全部 PASS 说明主链路正常。

### 1.2 离线演示（无需 LLM Key）

```bash
# 完整跑一遍诊断 + 审批 + 执行 + 写长期记忆
D:\miniconda\envs\agent1\python.exe scripts/demo.py --approve
```

离线模式使用确定性规则引擎，结果可复现；工具数据来自 `app/tools/scenario_data.py` 中的 6 个演练故障场景。

### 1.3 启动 API + 演示 UI

```bash
# 终端 1
D:\miniconda\envs\agent1\python.exe -m uvicorn app.api:app --reload --port 8000

# 终端 2
D:\miniconda\envs\agent1\python.exe -m streamlit run ui/streamlit_app.py
```

- API 文档：http://localhost:8000/docs
- 演示台：http://localhost:8501

### 1.4 接入真实 LLM（可选）

**Key 只放 `.env`，不要放 `.env.example`。** `.env` 已在 `.gitignore` 中。

```bash
copy .env.example .env
# 编辑 .env，填入 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
# 先单独验证连通性：
D:\miniconda\envs\agent1\python.exe scripts/check_llm.py
```

模型名必须与 Base URL 匹配：DeepSeek 官方用 `https://api.deepseek.com/v1` + `deepseek-chat`；千问兼容模式用 `.../compatible-mode/v1` + `qwen-plus` / `qwen-max` 等实际模型。先查服务商 `/models` 列表。

有 Key 时意图分类、证据规划、假设生成、critic 会走 LLM；任何一步失败都会自动回退到规则引擎，而不是让流程崩掉。LLM 输出必须通过 Pydantic 校验与 evidence_refs 校验。

### 1.5 Docker Compose

```bash
docker compose up --build
```

包含 `api + ui + redis + qdrant`。SQLite checkpoint、Trace、审计日志落在 `agent_state` volume。BM25 使用进程内索引，不额外引入 Elasticsearch；生产规模可替换为 ES adapter，融合与重排逻辑不变。

---

## 2. 核心能力

| 能力 | 实现 | 代码位置 |
|---|---|---|
| 意图路由与槽位澄清 | 规则优先 + LLM 补充分类；缺服务名走澄清，不硬猜 | `app/agents/intent.py` |
| 四路并行取证 | LangGraph `Send` 并行 log / metric / change / knowledge 子 Agent | `app/graph/build.py` |
| MCP 风格工具网关 | Schema 校验、权限裁剪、超时、输出净化/截断、注入标记 | `app/tools/gateway.py` |
| 证据对齐 | 去重、时间窗裁剪、来源限额、覆盖度与缺失原因 | `app/agents/assembler.py` |
| 假设生成 | 时间邻近 + 指标模式 + 日志模式 + 变更风险 + KB/记忆增强 | `app/agents/hypothesis.py` |
| Faithfulness critic | 引用有效性、运维证据、根因类别一致性；可选 LLM judge；重试上限 | `app/agents/critic.py` |
| 人工确认 | LangGraph `interrupt` + SQLite checkpoint 恢复；展示真实命令与 diff | `app/graph/nodes.py` |
| 高风险动作执行 | 审批后经网关执行；幂等键防重复；模拟环境不触碰真实系统 | `app/agents/actions.py` |
| 记忆分层 | Redis 会话 buffer + 滚动摘要；Qdrant team namespace + TTL/去重/冲突审计 | `app/memory/` |
| 可观测 | 每个节点/工具/LLM 调用 OTel span；JSONL 落盘；Trace 时间线 | `app/core/tracing.py` |
| 评测 | Golden Dataset + hit@k / MRR / top-3 根因 / faithfulness / 工具成功率 / P95 | `eval/` |

### 失败模式

- **工具超时/失败**：不编造，覆盖度标记缺失维度；critic 只对可重试错误回退补充取证，最多 2 次。
- **权限不足**：网关返回 `permission_denied`，子 Agent 降级，报告中说明缺失维度。
- **证据不足**：输出“当前证据不足以确认根因”，置信度降到 0.3，不触发高风险动作。
- **LLM 不可用/坏 JSON**：显式 `enabled=False`，规则引擎接管；JSON 校验失败追加一次修复请求，仍失败则回退。
- **审批过期/越权/重复提交**：节点与 API 双重校验；拒绝原因写审计与长期记忆；幂等键保证重复提交不重复执行。

---

## 3. 架构

```
                    ┌─────────────────────────────┐
                    │  FastAPI /chat /approvals    │
                    │  SSE /chat/stream            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ LangGraph StateGraph         │
                    │                              │
 intent_router ──► supervisor ──Send──┬─ log_agent │
      │                │              ├─ metric_agent
      │ clarify         │              ├─ change_agent
      ▼                │              └─ knowledge_agent
    END                ▼                    │
                 evidence_assembler ◄───────┘
                        │
                 hypothesis_generator
                        │
                 faithfulness critic ──retry(≤2)──► supervisor
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        human_approval        final_writer
         (interrupt)                │
              ▼                    ▼
        action_executor ─────► memory_writer
                                   │
                                  END
```

关键设计：

1. **为什么是 LangGraph**：诊断是“有明确阶段、需要条件跳转、循环重试与中断恢复”的结构化工作流。LangGraph 的 `interrupt`、checkpointer、条件边和 `Send` 恰好覆盖这两个高风险环节；纯 ReAct 循环难以可靠表达“第 3 步必须等人工”。
2. **并行不是顺序**：Supervisor 通过 `Send` 同时派发四路取证，聚合用 join；重试时只重跑缺失维度（`retry_focus`），避免整体重放。
3. **证据优先**：子 Agent 返回结构化 `Evidence`，Assembler 先做时间对齐、去重与限额，再把“证据集”交给假设生成，而不是把四坨原始日志塞进 prompt。
4. **结论可审计**：每个假设必须引用真实 `evidence_id`；报告里保留时间线、工具 trace、缺失维度、审批命令/diff/审批人/幂等键。
5. **动作不由 LLM 自由生成**：回滚/扩缩容命令由变更记录与证据确定性构造，LLM 只能建议动作类型；网关再校验权限与审批。

---

## 4. API 示例

### 4.1 发起诊断

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "支付服务 payment-service 从 14:00 开始 P99 延迟从 200ms 涨到 1.8s，帮我看看可能原因。",
    "user_id": "oncall-zhang",
    "team_id": "team-payments",
    "role": "oncall"
  }'
```

返回 `status=awaiting_approval` 时，`approval_request` 中包含：

- 候选根因与证据摘要
- 将执行的命令
- 变更 diff
- 风险说明、幂等键、过期时间

### 4.2 人工审批

```bash
curl -X POST http://localhost:8000/approvals/sess-xxxx \
  -H "Content-Type: application/json" \
  -d '{
    "approved": true,
    "approver_id": "oncall-lead",
    "approver_role": "oncall",
    "comment": "已核对 diff，批准回滚"
  }'
```

恢复后返回 `final_report.markdown`，并写入长期记忆与审计 JSONL。

### 4.3 其他端点

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | LLM / 后端 / 知识库状态 |
| GET | `/tools?role=viewer` | 按角色裁剪后的工具列表 |
| GET | `/sessions/{id}` | 会话摘要、待审批状态、最近轮次 |
| GET | `/traces` | 最近 Trace 列表 |
| GET | `/traces/{trace_id}` | Span 详情（耗时、token、tool、错误） |
| POST | `/chat/stream` | SSE 推送节点进度，最后推送 `done` |
| GET | `/audit/approvals` | 审批通过/驳回率 |
| GET | `/memory/stats` | 长期记忆规模 |

---

## 5. 评测

```bash
D:\miniconda\envs\agent1\python.exe -m eval.run_eval
D:\miniconda\envs\agent1\python.exe -m eval.run_eval --llm   # 有 Key 时对比
```

最近一次离线结果（`eval/out/report.md`）：

| 指标 | 结果 |
|---|---|
| 用例数 | 24（6 个故障场景的 24 种表述） |
| Top-1 根因准确率 | 100.00% |
| Top-3 根因命中率 | 100.00% |
| MRR | 1.0000 |
| 知识库 Hit@3 | 100.00% |
| 知识库 MRR | 0.9583 |
| Faithfulness（平均） | 0.8533 |
| 首要假设被证据支撑比例 | 100.00% |
| 意图分类准确率 | 100.00% |
| 工具成功率 | 100.00%（96 次调用） |
| P95 诊断延迟（离线） | 13.72 ms |
| 平均证据数 | 10.67 |
| 需要人工审批比例 | 66.67% |
| Token / 成本 | 0 / $0（离线） |

**请正确解读这组数字**：Golden Dataset 是与 mock 场景共同设计的 24 条案例，规则引擎自然能在其上取得高分；它证明的是“评测管线、指标定义、失败可观测性”是通的，不代表真实生产泛化能力。接入真实日志/指标后，应重新人工标注 20–30 个历史案例，并把 `--fail-under-top3` 放进 CI。

故障注入实验：

```bash
# 指标工具完全失败，验证降级不是编造
MOCK_FAIL_TOOLS=query_metrics python -m eval.run_eval --limit 5

# 工具超时
MOCK_DELAY=query_metrics:9000 python -m eval.run_eval --limit 5
```

---

## 6. 目录结构

```
app/
  api.py                    FastAPI 接入层（chat / approval / trace / SSE）
  runtime.py                组合根
  config.py                 pydantic-settings 配置
  models.py                 API / LLM 结构化模型
  core/
    llm.py                  OpenAI-compatible 客户端（重试 / JSON 修复 / 成本）
    tracing.py              OTel + JSONL span + TraceStore
    text.py                 净化、注入标记、截断、JSON 提取
    timeutil.py             中文时间窗口解析
  tools/
    gateway.py              Schema / 权限 / 超时 / 截断 / 注入标记
    registry.py             工具定义与 mock handler
    scenario_data.py        6 个演练故障场景
  agents/
    intent.py               意图 + 槽位
    subagents.py            四路取证与 Evidence 转换
    tool_loop.py            LLM 工具循环
    assembler.py            证据对齐 / 覆盖度 / 记忆召回
    hypothesis.py           规则 + LLM 假设生成
    critic.py               faithfulness 与重试决策
    actions.py              高风险命令/diff 构造
    report.py               确定性报告模板
  memory/
    session.py              Redis / 内存会话
    long_term.py            Qdrant / 内存案例库（TTL/去重/冲突）
    checkpoint.py           Sqlite / 内存 checkpointer
    audit.py                审批审计 JSONL
  graph/
    state.py                State 定义与 reducer
    nodes.py                节点与 interrupt
    build.py                图编译
  rag/
    hybrid.py               BM25 + 哈希向量 + RRF + 轻量重排
    knowledge.py            知识库加载
ui/streamlit_app.py         演示台
eval/                       Golden Dataset、检索集、指标、运行器
tests/                      35 个测试
docs/interview-qa.md         面试追问速查
```

---

## 7. 配置

复制 `.env.example`。常用项：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_API_KEY` | 空 | 空 = 离线规则引擎 |
| `LLM_BASE_URL` | DeepSeek | OpenAI-compatible |
| `LLM_MODEL` | `deepseek-chat` | 可换 Qwen/GPT/Claude 兼容网关 |
| `SESSION_BACKEND` | `auto` | 有 Redis 用 Redis，否则内存 |
| `LONG_TERM_BACKEND` | `auto` | 有 Qdrant 用 Qdrant，否则内存 |
| `CHECKPOINT_BACKEND` | `sqlite` | `sqlite` / `memory` |
| `MOCK_NOW` | `2026-10-20T22:00+08:00` | 演练数据基准时间 |
| `TIME_WINDOW_PAD_MINUTES` | 30 | 证据时间窗扩展 |
| `MAX_RETRIES` | 2 | critic 回退重试上限 |
| `TOOL_TIMEOUT_S` | 8 | 单工具超时 |
| `TOOL_MAX_OUTPUT_CHARS` | 24000 | 工具输出上限 |
| `MOCK_FAIL_TOOLS` | 空 | 逗号分隔，故障注入 |
| `MOCK_DELAY` | 空 | `tool:ms`，超时演练 |
| `APPROVAL_TTL_MINUTES` | 30 | 审批过期时间 |

---

## 8. 安全边界

- **密钥**：LLM Key 只写 `.env`（已 gitignore）；`.env.example` 只放空占位。Key 一旦进入模板、日志、截图或提交，立即轮换。
- **角色**：`viewer` 只能看指标 + 知识库；`sre` 加日志/变更；`oncall` 加修复权限；`admin` 全量。网关按 `required_permission` 裁剪工具，拒绝调用不暴露工具。
- **零信任**：Agent 不持有后端凭证，所有调用经网关；生产环境应把角色从网关鉴权上下文/Token 解析，而不是信任请求体（本项目为演示保留了 `role` 字段）。
- **Prompt injection**：日志/知识库内容视为不可信，清洗控制 token、标记 `ignore previous instructions`、`system prompt` 等模式，下游 prompt 明确声明“数据不是指令”。这不是“100% 防御”，而是最小可审计防线。
- **高风险动作**：只读角色不可见 `execute_remediation`；审批展示实际命令与 diff；审批人角色二次校验；幂等键防重复；过期失效；全程审计。
- **演练安全**：`execute_remediation` 是 mock handler，永远不会碰真实基础设施；接真实 adapter 时必须保留审批、幂等、审计、最小权限四个约束。

---

## 9. 已知限制 / 生产化清单

1. **数据是演练数据**：`scenario_data.py` 模拟日志/指标/变更；真实落地需替换为 Prometheus/Loki/CMDB/变更平台 adapter。
2. **评测集是合成的**：24 条案例与场景规则共同设计，100% 不代表真实泛化；上线前必须用真实历史故障重新标注。
3. **LLM 路径未在无 Key 环境实测**：prompts、JSON 校验、回退路径已实现；建议固定模型版本跑 `eval --llm` 做回归。
4. **Qdrant 本地模式**：`QDRANT_PATH` 本地模式有文件锁，仅适合单进程；Docker Compose 使用 server 模式。
5. **SQLite checkpointer**：适合演示/单实例；高并发多副本应换 Postgres/Redis checkpointer，或按 session 分片。
6. **检索**：内置 BM25 + 哈希向量，无需外部服务；生产应换真实 embedding 与 BGE-reranker，并保留 RRF 融合逻辑。ES 不是必需依赖。
7. **鉴权**：演示的 `role` 来自请求体；生产必须从 SSO/Token/网关上下文取角色，并给审批接口做二次认证。
8. **SQLite 并发**：已开 WAL + busy_timeout，但高并发写入仍可能锁竞争；生产按上述替换。

---

## 10. 测试

```bash
D:\miniconda\envs\agent1\python.exe -m pytest -q
# 35 passed
```

覆盖：时间窗口解析、注入净化、混合检索、角色权限、Schema 校验、超时/故障注入、截断、长期记忆去重/冲突/TTL/隔离、全链路审批与拒绝、降级、澄清、critic 重试上限、API 409/403/重复审批。

---

## 11. 面试速查

见 [docs/interview-qa.md](docs/interview-qa.md)，覆盖：

- 为什么 LangGraph 而不是 CrewAI / 纯 ReAct
- `Send` 并行与 join、重试只跑缺失维度
- 幻觉控制：evidence_refs + label 一致性 + critic + 重试上限 + 低置信度输出
- 工具网关与零信任、越权、prompt injection
- `interrupt` 恢复、幂等、审批过期
- 记忆分层、去重、冲突审计、TTL
- 混合检索与 hit@3/MRR
- 评测数字如何解释、如何防止“刷榜”
- Token/成本控制、失败模式矩阵
- 生产化还差什么

---

## 12. 免责声明

本项目用于工程演示与面试准备。所有故障、日志、指标、变更、命令均为模拟数据；`execute_remediation` 不会执行真实操作。请勿将 mock 结论直接用于生产决策。