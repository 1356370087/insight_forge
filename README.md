# InsightForge

> 企业市场与竞争情报多 Agent 研究平台

InsightForge 是一个面向深度研究任务的可配置多 Agent 运行时。它使用手写的 `QueryEngine/query` 双层 Agent Loop，将用户问题转化为研究计划，调度多个专职 Researcher 搜索与核验证据，并最终生成带来源的结构化研究报告。

项目同时提供 Next.js 研究控制台、FastAPI 服务和 Python 运行时，覆盖持久化恢复、人工审批、任务协调、质量门禁、长期记忆、安全治理与可观测性，适合构建市场扫描、竞争情报、技术调研、政策追踪和战略决策支持系统。

- 主仓库：[Gitee](https://gitee.com/zeng-haozhe/open_deep_research)
- 镜像仓库：[GitHub](https://github.com/1356370087/open_deep_research)
- 许可证：[MIT](LICENSE)

当前仓库名和 Python 导入命名空间仍为 `open_deep_research`；InsightForge 是项目对外名称，本次品牌调整不改变现有代码接口。

## 核心能力

- 手写 Agent Runtime：不依赖 LangGraph 图执行器，由 `QueryEngine` 和通用 `query()` 循环管理状态、工具调用、重试、终止与上下文恢复。
- 多 Agent 研究：Lead Agent 负责澄清与编排，Supervisor 负责拆解与调度，Researcher 在独立上下文中完成单主题研究。
- 同步与异步调度：支持同步 `ConductResearch` 并行调用，也支持文件状态、Mailbox、Teammate Pool 和检查点驱动的异步研究任务。
- 可追溯 Web 证据：在 `enforced` 模式下执行 Search → Top-K Fetch → Extract → Evidence，最终引用仅允许来自已抓取、可验证的证据。
- 多模型与多工具：支持 OpenAI、Anthropic、Google、DeepSeek、Groq 等模型接入，支持 Tavily、原生 Web Search、MCP 和可选 Playwright MCP。
- 覆盖与来源契约：从用户消息编译稳定的需求清单（Coverage Contract）和来源准入范围（Source Contract），支持显式 URL 白/黑名单与"仅官方来源"约束，范围外证据不会进入报告。
- 人机协作：支持研究计划审批、报告大纲审批、修改、取消、中途方向反馈和面向具体任务的证据追问。
- 质量门禁：结合确定性规则与独立 Judge 模型评估工具结果和研究交接，按需求维护覆盖账本，支持多档质量严格度、评估输入预算及失败恢复。
- 研究控制台：内置 Next.js 前端，提供研究任务创建、SSE 实时进度、HITL 审批、Subagent 执行详情、本地账号登录和管理后台。
- 报告编排：支持综合报告、执行摘要、决策简报、FAQ、对比矩阵、优劣分析和文献综述等报告类型。
- 可恢复运行：研究摘要、状态增量、公开事件、任务工件和稳定检查点持久化到本地文件，可显式恢复中断任务。
- 安全与治理：包含自有 IAM/RBAC（本地账号、角色权限、审计）、工具权限、参数校验、域名审批、SSRF 防护、外部内容隔离和可选 Docker 沙箱。
- 可观测性：内置 SQLite Trace Store，可选接入 Langfuse、Prometheus/Grafana 和 Helicone。
- 长期记忆：可选接入 Mem0，支持用户偏好、领域画像、项目记忆、研究洞察、反思和软遗忘。

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- [uv](https://docs.astral.sh/uv/)
- 至少一个可用的模型 API Key
- 使用 Tavily 搜索时需要 `TAVILY_API_KEY`
- 启用完整 IAM 认证时需要 PostgreSQL；仅本地开发旁路或 Docker 演示模式不需要
- 运行研究控制台前端需要 Node.js 与 [pnpm](https://pnpm.io/)
- Docker 仅在容器化部署或启用 Researcher 沙箱时需要

### 安装

```bash
git clone https://gitee.com/zeng-haozhe/open_deep_research.git
cd open_deep_research
uv sync
```

复制环境变量模板：

```bash
# Linux / macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

根据所选模型和搜索服务填写 `.env`。下面是一个使用 OpenAI 与 Tavily 的最小示例：

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

SUMMARIZATION_MODEL=openai:gpt-4.1-mini
RESEARCH_MODEL=openai:gpt-4.1
COMPRESSION_MODEL=openai:gpt-4.1
FINAL_REPORT_MODEL=openai:gpt-4.1

SEARCH_API=tavily
WEB_PIPELINE_MODE=enforced
```

服务启动时会加载项目根目录下的 `.env`。配置优先级为：

1. 环境变量，例如 `RESEARCH_MODEL`
2. `RunnableConfig.configurable` 或 HTTP 请求中的 `configurable`
3. `Configuration` 的代码默认值

运行开始后，影响恢复一致性的关键配置会被冻结并写入运行清单。恢复任务不会因为外部环境变量变化而静默改变质量策略或核心执行语义。

### 启动本地服务

后端默认启用自有 IAM 认证：需要配置 `IAM_DATABASE_URL`（PostgreSQL）、执行数据库迁移，并通过 CLI 创建首位管理员（见[安全与沙箱](#安全与沙箱)）。仅在可信的本地开发环境中，可以启用显式认证旁路：

```env
APP_ENV=development
LOCAL_DEV_AUTH_BYPASS=true
```

旁路仅在 `APP_ENV=development` 时生效，所有请求会获得一个合成的 researcher/developer 身份，不包含 IAM 管理权限。不要在共享环境或生产部署中启用该选项。

启动服务：

```bash
uv run uvicorn open_deep_research.server:app --reload --host 127.0.0.1 --port 2024
```

启动后可访问：

- API：`http://127.0.0.1:2024`
- OpenAPI 文档：`http://127.0.0.1:2024/docs`
- 内置观测页面：`http://127.0.0.1:2024/observability/ui`
- Prometheus 指标：`http://127.0.0.1:2024/metrics`

### 启动研究控制台前端

仓库内置基于 Next.js 的研究控制台，提供任务创建、实时进度、HITL 审批、Subagent 执行详情抽屉、账号设置与管理后台：

```bash
cd frontend
cp .env.example .env.local   # Windows PowerShell: Copy-Item .env.example .env.local
pnpm install
pnpm dev
```

访问 `http://localhost:3000`。浏览器只连接 Next.js BFF（`/api/research`、`/api/auth`、`/api/iam`）；Access/Refresh Token 保存在 HttpOnly Cookie 中，由服务端代理注入 `Authorization` 头，浏览器不接触 JWT。

前端回源地址等变量见 `frontend/.env.example`：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RESEARCH_API_ORIGIN` | `http://127.0.0.1:2024` | BFF 服务端回源地址 |
| `NEXT_PUBLIC_RESEARCH_API_BASE` | `/api/research` | 浏览器端 API 前缀 |
| `NEXT_PUBLIC_LOCAL_DEV_AUTH_BYPASS` | `false` | 前端本地开发鉴权旁路，需与后端旁路同时开启 |

### Docker Compose 一键部署

```bash
docker compose up --build
```

该编排启动三个服务：FastAPI API（容器内 2024）、Next.js 前端（容器内 3000）和 Nginx 代理（宿主机 `http://localhost:8080`）。默认以 `LOCAL_DEV_AUTH_BYPASS=true` 的本地演示模式运行，无需 PostgreSQL；如需完整认证，将 compose 中的旁路开关改为 `"false"` 并提供 `IAM_DATABASE_URL`。

### 创建后台研究任务

```bash
curl -X POST "http://127.0.0.1:2024/runs" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "分析 2026 年中国企业级 AI Agent 市场的竞争格局、主要厂商、进入壁垒与未来两年趋势，并引用一手来源。"
      }
    ],
    "configurable": {
      "allow_clarification": false,
      "report_type": "decision_brief"
    },
    "metadata": {}
  }'
```

响应包含 `run_id`、当前状态和事件流地址：

```json
{
  "run_id": "0f6c...",
  "status": "running",
  "events_url": "/runs/0f6c.../events",
  "last_event_id": 1
}
```

查询状态与最终结果：

```bash
curl "http://127.0.0.1:2024/runs/<run_id>"
```

### 使用 SSE 流式运行

直接创建任务并持续接收公开事件：

```bash
curl -N -X POST "http://127.0.0.1:2024/runs/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "比较三家主要云厂商的企业生成式 AI 平台能力、定价思路与生态策略。"
      }
    ],
    "configurable": {
      "allow_clarification": false
    }
  }'
```

已有任务可以从指定事件序号继续订阅：

```bash
curl -N "http://127.0.0.1:2024/runs/<run_id>/events" \
  -H "Last-Event-ID: 42"
```

公开事件经过字段白名单、URL 规范化和敏感信息清理，只暴露阶段、任务进度、来源摘要、审批请求和最终状态；内部恢复 Journal 不会直接暴露给客户端。

生产环境请求需要增加认证头。Access Token 通过 `POST /auth/login` 获取：

```bash
curl -X POST "http://127.0.0.1:2024/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "..."}'

# 后续请求携带返回的 access token
-H "Authorization: Bearer <access-token>"
```

### Python 调用

```python
import asyncio

from open_deep_research.agents.query_engine import QueryEngine


async def main() -> None:
    config = {
        "configurable": {
            "allow_clarification": False,
            "search_api": "tavily",
            "web_pipeline_mode": "enforced",
            "report_type": "default",
        },
        "metadata": {
            "deployment_surface": "python",
        },
    }
    engine = QueryEngine(config)
    state = await engine.submit_message(
        [
            {
                "role": "user",
                "content": "研究目标市场、竞争者定位和可验证的增长驱动因素。",
            }
        ],
        config,
    )
    print(state["final_report"])


asyncio.run(main())
```

`QueryEngine` 负责完整研究流程；`ResearcherQueryEngine.run_topic()` 可用于在干净上下文中单独执行一个聚焦研究主题。

## 系统架构

```mermaid
flowchart TD
    Client["研究控制台 / HTTP / SSE / Python 客户端"] --> API["FastAPI 服务"]
    API --> Lead["QueryEngine / Lead Agent"]

    Lead --> Prepare["消息压缩 · 记忆召回 · 澄清 · Research Brief"]
    Prepare --> Plan["研究计划与可选 HITL 审批"]
    Plan --> Supervisor["Supervisor Query Loop"]

    Supervisor --> Sync["同步 ConductResearch"]
    Supervisor --> Async["异步任务池 / File Mailbox"]
    Sync --> Researcher["ResearcherQueryEngine"]
    Async --> Researcher

    Researcher --> Tools["Governed Tools"]
    Tools --> Search["Tavily / 原生搜索 / MCP / Skills"]
    Search --> Evidence["Search → Top-K Fetch → Evidence"]
    Evidence --> Researcher

    Researcher --> Compression["研究压缩与交接工件"]
    Compression --> Quality["质量评估与证据准入"]
    Quality --> Supervisor

    Supervisor --> Outline["报告大纲与可选 HITL 审批"]
    Outline --> Report["Report Orchestrator"]
    Report --> Memory["记忆提取与维护"]
    Memory --> Result["报告、结构化产物与公开事件"]

    Runtime["预算 · 超时 · 取消 · Lease · Checkpoint"] -.-> Lead
    Runtime -.-> Supervisor
    Runtime -.-> Researcher
    Security["认证 · 权限 · 域名治理 · 内容安全 · Sandbox"] -.-> Tools
    Observe["SQLite · Langfuse · Prometheus"] -.-> Lead
    Observe -.-> Supervisor
    Observe -.-> Researcher
```

### 外层 QueryEngine

主流程由 [`QueryEngine`](src/open_deep_research/agents/query_engine.py) 驱动：

```text
summarize_messages
  → memory_recall
  → clarify_with_user
  → write_research_brief
  → plan_approval
  → supervisor loop
  → outline_approval
  → final_report_generation
  → memory_extract_and_write
```

对外公开的六个阶段为 `preparing`、`planning`、`researching`、`synthesizing`、`writing` 和 `finalizing`。内部检查点记录下一稳定阶段，因此服务中断后可以通过 `/runs/{run_id}/resume` 显式恢复。

### 通用 query 循环

[`agents/query.py`](src/open_deep_research/agents/query.py) 提供通用模型—工具循环，Lead、Supervisor 和 Researcher 共用同一套运行语义：

- 模型调用、结构化工具定义和工具批次执行
- 工具并发、批次大小和超时控制
- Before-turn、Tool-results 和 Stop hooks
- 上下文裁剪与工具结果截断
- 预算、取消、重试和模型传输错误分类
- 统一的状态更新与终止原因

### Supervisor 与 Researcher

Supervisor 负责理解 `research_brief`、反思研究缺口、拆分任务和决定何时停止。Researcher 在隔离的消息窗口中执行单主题搜索、抓取、证据抽取与压缩，避免多个研究主题相互污染上下文。

两种调度方式由 `enable_async_research` 切换：

| 模式 | 主要工具 | 行为 |
|---|---|---|
| 同步，默认 | `ConductResearch` | Supervisor 一次发起多个研究单元，通过 `asyncio.gather` 并行等待结果 |
| 异步 | `StartResearchTask`、`CheckResearchTask`、`WaitForResearchUpdates` 等 | Teammate Pool 独立执行任务，状态、Mailbox、工件和检查点写入文件系统 |

Supervisor 只接收压缩结果和 SHA 校验后的研究工件。启用质量评估后，未通过证据准入或交接质量门禁的结果不会直接进入最终报告。

### 模块导航

| 模块 | 主要职责 |
|---|---|
| [`agents`](src/open_deep_research/agents) | QueryEngine、通用 query 循环、Lead/Supervisor/Researcher 节点与提示词协议 |
| [`web`](src/open_deep_research/web) | 候选来源规范化、排序、抓取、HTML/PDF 提取、分块、证据抽取与缺口分析 |
| [`tools`](src/open_deep_research/tools) | 按工具目录组织的协议声明、统一 registry、动态提示词、MCP 子包、权限治理、参数校验、重试和错误分类 |
| [`tasks`](src/open_deep_research/tasks) | 异步任务、Teammate Pool、文件状态、Mailbox、Lease、恢复、事件与域名审批 |
| [`report`](src/open_deep_research/report) | 报告 Profile、一次性/分节组装、引用恢复、覆盖率检查和多格式渲染 |
| [`memory`](src/open_deep_research/memory) | Mem0 Store、记忆候选策略、检索排序、画像、反思、衰减和维护 |
| [`evaluation`](src/open_deep_research/evaluation) | Judge 工厂、统一指标、执行合规、评估快照和 Benchmark 导出 |
| [`security`](src/open_deep_research/security) | HTTP 输入边界、外部内容保护、SSRF 与网络目标校验 |
| [`sandbox`](src/open_deep_research/sandbox) | Docker Researcher 工作区、资源限制、网络策略和清理 |
| [`observability`](src/open_deep_research/observability) | TraceRecorder、SQLite Span Store、Langfuse Bridge 和 Prometheus 指标 |

工具系统以 `tools/registry.py` 作为唯一装配入口。每个工具目录通过 `definition.py` 声明协议对象，并由 `prompt.py` 提供与工具生命周期一致的模型指导；Researcher 和 Supervisor 的 `<Available Tools>` 均从最终通过启用条件与权限裁剪的工具集动态生成。MCP 装载、OAuth 与内置浏览器能力集中在 `tools/mcp/`，Supervisor 工具通过冻结的 `SupervisorToolDeps` 注入运行依赖。`max_tool_description_chars` 控制投影给模型的单工具描述预算。

## 关键机制

### Web 证据流水线

`web_pipeline_mode` 控制 Web 研究的迁移与强制级别：

| 模式 | 行为 |
|---|---|
| `legacy` | 保留 Tavily、OpenAI 或 Anthropic 的提供商特定搜索输出 |
| `shadow` | 继续返回旧输出，同时抽样执行候选规范化、去重、排序和 Top-K 指标记录 |
| `enforced`，默认 | 暴露 `web_research` 与 `fetch_url`，最终引用只允许来自已抓取并进入 Evidence Registry 的来源 |

`enforced` 模式的数据流：

```text
SearchRequest
  → 多来源候选发现
  → URL 规范化与去重
  → 权威性和相关性排序
  → Top-K 选择
  → 域名策略与抓取预算检查
  → Local HTTP / Playwright / Tavily Extract / Firecrawl
  → HTML 或 PDF 提取
  → 分块与证据句抽取
  → Evidence Registry
  → 缺口分析与后续检索
```

默认抓取顺序为本地 HTTP、可选 Playwright MCP、Tavily Extract、Firecrawl Scrape。没有对应凭据时会跳过远程提取器。`respect_robots_txt=true` 时本地抓取遵守 robots.txt。

### 人机协作

设置 `enable_human_in_loop=true` 后，可以启用：

- 研究计划审批：`hitl_require_plan_approval`
- 报告大纲审批：`hitl_require_outline_approval`
- 计划修改次数限制：`hitl_max_plan_revisions`
- 中途反馈模式：`safe_points` 或 `task_queue`

处理待审批动作：

```bash
curl -X POST \
  "http://127.0.0.1:2024/runs/<run_id>/human-actions/<action_id>" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "revise",
    "message": "增加对价格体系、渠道策略和客户迁移成本的研究。"
  }'
```

`action` 支持 `approve`、`revise` 和 `cancel`。

发送中途方向或证据追问：

```bash
curl -X POST "http://127.0.0.1:2024/runs/<run_id>/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "direction",
    "message": "优先引用公司财报、监管文件和官方产品文档。"
  }'
```

`type=evidence_question` 时还可以携带 `task_id`、`source_url` 或 `claim_text`，把反馈定向到具体研究任务或证据。

### 质量门禁

设置 `quality_evaluation_enabled=true` 后，运行时会使用独立 Judge 模型评估 Researcher 工具结果和提交给 Supervisor 的研究交接。

**覆盖契约与覆盖账本**：运行开始时从用户消息编译覆盖契约（Coverage Contract），研究需求被切分为最多 20 条清单项，每条获得稳定可复现的需求 ID（`COV-NN-<hash>`）。Supervisor 委派任务时会把需求子集分配给对应 Researcher，质量门禁只对归属需求做硬覆盖检查；未被认领的需求会被兜底轮询分配，避免需求在调度中丢失。覆盖账本（Coverage Ledger）按需求 ID 跨任务合并状态（`supported`/`partial`/`unsupported`，只单调提升），并记录支撑证据与任务来源，供报告合成阶段消费。

**来源契约**：用户消息中的来源约束会被编译为来源准入范围（Source Contract）：

- 显式 URL 白名单：例如"只使用以下 URL/链接 …"，范围外来源一律拒绝
- 显式 URL 黑名单：例如"不得使用 / 排除 …"，命中即拒绝
- 仅官方来源：只有命中内置官方源 Profile（如 PostgreSQL 官方文档与仓库）或显式白名单的证据可准入，无法验证归属的来源标记为不可信

受限契约下证据准入 fail-closed；显式 URL 白名单模式下最少来源数放宽为 1，多样性留到运行合并层评估；压缩文本中出现范围外 URL 会触发硬拒。

`quality_evaluation_rigor` 支持：

- `very_relaxed`
- `relaxed`
- `balanced`，默认
- `strict`
- `very_strict`

严格度只调整语义评分阈值。以下规则属于不可补偿的硬门禁，不会被较高的平均分覆盖：

- 外部证据安全检查
- 工具执行和用户约束合规性
- 研究工件 SHA 完整性
- 已抓取证据准入与来源契约范围校验
- `quality_evaluation_min_sources` 最少来源要求

**评估输入预算**：Judge 输入受 `quality_evaluation_max_input_chars`（默认 30000 字符）硬预算约束。超长语义字段按身份保护规则截断并加 `input_truncated` 标记；证据注册表按引用优先级做有界投影，被压缩研究显式引用的证据最先保留。

每个运行会冻结质量严格度、策略版本、Judge 模型和评估纪元。旧配置 `QUALITY_EVALUATION_MIN_SCORE=1..5` 仅用于兼容迁移，并分别映射到五档严格度；新部署应使用 `QUALITY_EVALUATION_RIGOR`。

Judge 传输或解析失败是否放行由 `quality_evaluation_fail_open` 控制，但确定性证据硬门禁始终有效：内层（工具批次）评估失败时按策略放行或停止研究支出；外层（交接门禁）评估失败时，v4 策略会拒绝该份自由文本交接，但允许确定性校验通过的运行继续。若完整交接被拒绝，运行时会尝试从 SHA 校验且已准入的证据中恢复部分报告；不存在合格证据时，任务以 `insufficient_evidence` 失败。

### 持久化、恢复与任务租约

`query_session_persistence_enabled=true` 时，每次运行都在 `.runs/<run_id>` 下保存稳定状态。Lead 通过文件 Lease 和 fencing token 保证同一运行同一时间只有一个有效所有者，心跳丢失会触发取消，防止旧进程继续写入。

异步研究默认使用文件后端：

- `TaskSnapshot` 保存任务状态和工件引用
- `FileMailbox` 传递任务进度与控制消息
- `TeammatePool` 管理固定 Researcher worker
- `CheckpointManager` 恢复未完成或已进入压缩阶段的任务
- 研究工件读取时校验 SHA-256

显式恢复：

```bash
curl -X POST "http://127.0.0.1:2024/runs/<run_id>/resume" \
  -H "Content-Type: application/json" \
  -d '{
    "configurable": {},
    "metadata": {}
  }'
```

已完成、已取消、仍被其他 worker 持有或清单不兼容的运行会拒绝恢复。

取消任务：

```bash
curl -X POST "http://127.0.0.1:2024/runs/<run_id>/cancel"
```

### 报告类型与输出格式

`report_type` 支持：

| 值 | 说明 | 默认组装方式 |
|---|---|---|
| `default` | 综合研究报告 | 单次综合 |
| `executive_summary` | 执行摘要 | 单次综合 |
| `decision_brief` | 决策简报 | 单次综合 |
| `faq` | FAQ | 单次综合 |
| `comparison_matrix` | 对比矩阵 | 分节组装 |
| `pros_cons` | 优劣分析 | 分节组装 |
| `literature_review` | 文献综述 | 分节组装 |

`output_format` 支持：

- `markdown`
- `structured_json`
- `slides`
- `one_pager`

无论选择哪种格式，`final_report` 始终保留规范化后的 Markdown。非 Markdown 产物通过结果中的 `artifacts` 返回。当前 `slides` 是结构化幻灯片数据，不是 `.pptx` 文件。

`reference_style` 支持编号引用 `numbered` 和类 BibTeX 引用 `bibtex_like`。在 `enforced` Web 模式或质量评估模式下，报告中的 URL 会经过 Evidence Allowlist 过滤。

### 长期记忆

`enable_memory=true` 时，运行开始前会召回与用户和项目相关的记忆，报告生成后可自动抽取新记忆。

基础模式支持：

- 用户研究偏好
- 领域画像
- 项目记忆

`memory_advanced_enabled=true` 后额外支持：

- 经过证据校验的研究洞察
- 重要性、相关性与时间衰减联合排序
- 访问强化、冲突判断和软遗忘
- 周期性反思与研究画像更新

可使用 Mem0 Platform 或 OSS 后端。评估脚本默认关闭跨运行记忆，避免评测污染。

### 安全与沙箱

HTTP 服务使用自有 IAM 与 RBAC 子系统认证（位于 `src/security/rbac`），不依赖外部身份提供商：

- 本地账号体系：Argon2 密码哈希、邮箱验证、注册审批、密码重置与会话管理
- EdDSA（Ed25519）签名的 Access/Refresh 双 Token：独立密钥与 kid，Refresh Token 一次性轮换并检测重用，`authz_version` 支持权限变更后即时吊销旧令牌
- 角色与权限：4 个系统角色（`viewer`、`researcher`、`developer`、`admin`）加自定义角色，基于封闭的权限目录（研究域 + IAM 域）构建权限矩阵；长连接 SSE 会周期性重新授权
- 内置登录、注册、邮件与重置限流，以及 IAM 审计事件
- 认证身份写入运行 owner 元数据，查询、恢复、事件订阅、控制和可观测接口都会校验运行所有权

启用完整认证需要 PostgreSQL 和数据库迁移，并创建首位管理员：

```bash
# 配置 IAM_DATABASE_URL 后执行迁移
uv run alembic upgrade head

# 创建首位管理员（幂等，授予 admin + researcher 角色）
uv run python -m security.cli bootstrap-admin \
  --email admin@example.com --password '...'
```

主要安全边界包括：

- 客户端不能伪造 `system` 或 `tool` 消息
- HTTP 请求不能覆盖 MCP、沙箱、工具白名单、运行目录、模型端点白名单等管理员配置
- 工具按 Agent 角色、用户角色、来源和副作用分类授权
- 工具参数在执行前按 JSON Schema 和管理员约束校验
- 外部内容被标记为不可信证据，并检查提示注入模式
- HTTP 抓取校验 DNS、连接对端和私有地址，降低 SSRF 风险
- 严格网络模式批量收集目标域名并等待显式批准

V7 沙箱通过 `SANDBOX_ENABLED=true` 启用，并强制要求
`ENABLE_ASYNC_RESEARCH=true`。API 不持有 Docker Socket：独立 Controller
管理固定 digest 的 Worker 容器，Gateway 负责模型/搜索/MCP 凭据、预算 RPC
和受控出网。Worker 使用只读 RootFS、非 root 用户、CPU/内存/PID/tmpfs
限制及每任务内部网络；任何依赖或策略不可用都会 fail-closed。

管理员通过 `config/sandbox-policy.toml` 配置默认的
`research-gateway-only` 或可选 `developer-workspace` Profile。旧
`enable_docker_sandbox`、`sandbox_network_mode` 与其他平铺 `sandbox_*`
字段已删除，迁移与验收规范见
[`docs/07-Docker沙箱隔离修复SPEC.md`](docs/07-Docker沙箱隔离修复SPEC.md)。
当前默认策略未把任何角色映射到 `developer-workspace`；在 Linux/WSL2
的 bwrap+socat 与短生命周期 command task 验收通过前不得启用。沙箱内
Browser MCP 同样保持关闭，直到管理员提供并验收独立 Browser Gateway
镜像 digest。

首次启用或 Worker 源码变化后，先构建镜像并把本机内容 ID 固定到策略，
再启动沙箱服务：

```bash
docker compose --profile sandbox-build build sandbox-worker-image
uv run python -m open_deep_research.sandbox.pin_image \
  --image insightforge-sandbox-worker:local \
  --policy config/sandbox-policy.toml
docker compose --profile sandbox up --build -d
uv run python -m open_deep_research.sandbox.doctor
```

`pin_image` 只接受 `sha256:` image ID 并原子改写、重新校验 TOML；未固定、
镜像不存在或 Controller/Gateway/资源预算不满足时，API readiness 会拒绝流量。

### 可观测性

`TraceRecorder` 是统一的运行、Agent、模型和工具埋点边界。默认启用本地 SQLite：

```text
.runs/traces.sqlite3
```

内置接口：

- `/observability/runs`
- `/observability/runs/{run_id}`
- `/observability/runs/{run_id}/spans`
- `/observability/runs/{run_id}/usage`
- `/observability/runs/{run_id}/metrics`
- `/observability/ui`

可选接入 Langfuse：

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=production
```

可选启用 Prometheus：

```env
PROMETHEUS_ENABLED=true
PROMETHEUS_METRICS_PATH=/metrics
```

指标覆盖运行、Agent、模型与工具延迟，Token、缓存、重试、429、吞吐、估算成本、工具成功率、搜索零来源、任务队列健康、证据数量、报告规模、引用密度和质量评分。高基数字段如 run ID、trace ID、task ID 和 user ID 不会作为 Prometheus Label。

Trace Payload 默认为 `preview`，常见凭据和 Bearer Token 默认脱敏。启用 Langfuse LangChain Callback 时，由 Callback 记录 Langfuse generation，`TraceRecorder` 仍负责 SQLite 和 Prometheus，避免重复 generation。

## HTTP API

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET` | `/capabilities` | 查询运行能力与配置投影，供前端渲染 |
| `POST` | `/auth/login` 等 `/auth/*` | 本地账号登录、注册、邮箱验证、Token 刷新、注销、会话与密码管理 |
| `GET`/`PATCH` 等 | `/admin/*` | IAM 管理接口：用户审批、角色与权限目录、审计事件（需 admin 权限） |
| `POST` | `/runs` | 创建后台研究任务 |
| `POST` | `/runs/stream` | 创建任务并通过 SSE 返回持久化公开事件 |
| `GET` | `/runs/{run_id}` | 获取当前状态、进度投影和最终结果 |
| `GET` | `/runs/{run_id}/events` | 回放并持续订阅公开事件，支持 `Last-Event-ID` |
| `POST` | `/runs/{run_id}/resume` | 从稳定检查点显式恢复可恢复任务 |
| `POST` | `/runs/{run_id}/cancel` | 取消活动或持久化任务 |
| `POST` | `/runs/{run_id}/human-actions/{action_id}` | 审批、修改或取消 HITL 动作 |
| `POST` | `/runs/{run_id}/feedback` | 提交中途方向或证据追问 |
| `GET` | `/runs/{run_id}/tasks/{task_id}/activity` | 查询子代理任务的活动事件（安全投影） |
| `GET` | `/runs/{run_id}/tasks/{task_id}/activity/stream` | 以 SSE 订阅子代理任务活动 |
| `GET` | `/observability/runs` | 查询当前用户的观测运行列表 |
| `GET` | `/observability/runs/{run_id}` | 查询单次运行观测摘要 |
| `GET` | `/observability/runs/{run_id}/spans` | 查询 Span 树 |
| `GET` | `/observability/runs/{run_id}/usage` | 查询 Token 和成本聚合 |
| `GET` | `/observability/runs/{run_id}/metrics` | 查询重试、限流、缓存和工具指标 |
| `GET` | `/observability/ui` | 服务端渲染的轻量观测页面 |
| `GET` | `/metrics` | Prometheus 指标端点，路径可配置 |

管理员拥有的安全配置必须通过环境变量或服务端 `RunnableConfig` 设置，不能由 `/runs` 请求覆盖。

## 常用配置

完整字段、校验规则和 UI 元数据见 [`configuration.py`](src/open_deep_research/configuration.py)，环境变量示例见 [`.env.example`](.env.example)。

下表默认值来自 `Configuration`。复制 `.env.example` 后，模板中的显式环境变量会覆盖代码默认值。

### 模型与研究

| 配置字段 / 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `summarization_model` / `SUMMARIZATION_MODEL` | `openai:gpt-4.1-mini` | 搜索结果和公开发现摘要 |
| `research_model` / `RESEARCH_MODEL` | `openai:gpt-4.1` | Researcher 模型 |
| `compression_model` / `COMPRESSION_MODEL` | `openai:gpt-4.1` | 研究压缩模型 |
| `final_report_model` / `FINAL_REPORT_MODEL` | `openai:gpt-4.1` | 报告写作模型 |
| `model_fallbacks` / `MODEL_FALLBACKS` | `{}` | 按七个模型角色配置候选链；默认关闭，仅在限流、瞬态错误或模型不可用时切换 |
| `search_api` / `SEARCH_API` | `tavily` | `tavily`、`openai`、`anthropic` 或 `none` |
| `max_concurrent_research_units` | `5` | 同步研究单元最大并发 |
| `max_researcher_iterations` | `6` | Supervisor 最大研究迭代 |
| `max_react_tool_calls` | `10` | 单个 Researcher 最大工具循环 |
| `allow_clarification` | `true` | 是否允许研究前向用户澄清 |

所选 Researcher 模型必须与搜索方式兼容，并支持 Tool Calling；参与结构化计划、摘要或质量评估的模型还需要支持相应的 Structured Output。

模型标识、provider 推断、API key/base URL 和兼容参数统一由 `model_resolution.py` 解析。`model_fallbacks` 可配置 `supervisor`、`researcher`、`summarization`、`message_summary`、`compression`、`final_report`、`quality_evaluation`；候选链在运行开始时冻结，跨 provider 切换前会清理专有消息元数据。切换会记录到 Trace，并以 `query.model_fallback` 公共事件暴露 `{turn, from_model, to_model, reason}`。

需要按角色隔离凭据时，可使用 `SUPERVISOR_API_KEY`、`RESEARCHER_API_KEY`、`SUMMARIZATION_API_KEY`、`MESSAGE_SUMMARY_API_KEY`、`COMPRESSION_API_KEY`、`FINAL_REPORT_API_KEY` 和 `QUALITY_EVALUATION_API_KEY` 覆盖对应 provider 的通用 Key。启用 `GET_API_KEYS_FROM_CONFIG=true` 后，这些同名字段应放入 `configurable.apiKeys`，不会回退读取进程环境变量。

### Web、预算与执行

| 配置字段 | 默认值 | 说明 |
|---|---:|---|
| `web_pipeline_mode` | `enforced` | Web 证据流水线模式 |
| `fetch_top_k` | `5` | 每轮选择抓取的候选数 |
| `web_min_source_authority` | `0.65` | 候选来源最低权威度 |
| `max_fetches_per_researcher` | `12` | 单 Researcher 抓取预算 |
| `max_fetches_per_run` | `40` | 单次运行全局抓取预算 |
| `fetch_global_concurrency` | `4` | 全局抓取并发 |
| `max_concurrent_tool_calls` | `8` | 工具调用并发 |
| `model_call_timeout_seconds` | `180` | 模型调用超时 |
| `research_tool_call_timeout_seconds` | `300` | 搜索、抓取、排序和证据抽取工具超时 |
| `run_deadline_seconds` | `null` | 可选整次运行墙钟截止时间 |

还可以设置模型调用数、工具调用数、输入/输出 Token 和微美元成本预算。任一预算或截止时间耗尽后，Completion Policy 会根据已接受证据决定生成部分报告或终止。

### 协作、持久化与质量

| 配置字段 | 默认值 | 说明 |
|---|---:|---|
| `enable_human_in_loop` | `false` | 开启计划、大纲审批和中途反馈 |
| `enable_async_research` | `false` | 开启异步研究任务工具 |
| `task_state_backend` | `file` | 生产文件状态或测试内存状态 |
| `query_session_persistence_enabled` | `true` | 持久化 Query 会话与恢复工件 |
| `runs_dir` | `.runs` | 运行、事件、任务和观测数据目录 |
| `quality_evaluation_enabled` | `false` | 开启运行时 Judge |
| `quality_evaluation_model` | `openai:qwen3.7-plus` | Judge 评估模型 |
| `quality_evaluation_rigor` | `balanced` | Judge 语义严格度 |
| `quality_evaluation_min_sources` | `2` | 质量门禁最少来源 |
| `quality_evaluation_fail_open` | `true` | Judge 传输失败是否放行 |
| `quality_evaluation_max_input_chars` | `30000` | Judge 评估输入字符预算 |
| `quality_gap_recovery_max_attempts` | `1` | 质量缺口恢复追加轮数 |
| `task_timeout_seconds` | `600` | 研究子任务基础超时，质量门禁在其上叠加宽限 |
| `enable_memory` | `false` | 开启 Mem0 长期记忆 |

### 报告与可观测性

| 配置字段 | 默认值 | 说明 |
|---|---:|---|
| `report_type` | `default` | 报告产品类型 |
| `output_format` | Profile 默认值 | Markdown、JSON、Slides 或 One-pager |
| `reference_style` | Profile 默认值 | 编号引用或类 BibTeX |
| `observability_enabled` | `true` | 可观测性总开关 |
| `sqlite_observability_enabled` | `true` | 本地 SQLite Trace Store |
| `trace_payload_mode` | `preview` | `none`、`preview` 或 `full` |
| `trace_redaction_enabled` | `true` | Trace 凭据脱敏 |
| `langfuse_enabled` | `false` | 镜像 Trace 到 Langfuse |
| `prometheus_enabled` | `false` | 输出 Prometheus 指标 |

### 身份与访问控制（IAM）

IAM 配置通过环境变量设置，完整清单见 [`.env.example`](.env.example) 的 IAM 配置段：

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `APP_ENV` | `development` | 运行环境；`production` 下强制显式签名密钥与 SMTP |
| `IAM_DATABASE_URL` | 空 | IAM PostgreSQL 连接串（asyncpg），关闭旁路时必填 |
| `LOCAL_DEV_AUTH_BYPASS` | `false` | 本地开发认证旁路，仅 `APP_ENV=development` 生效 |
| `IAM_JWT_ACCESS_SIGNING_KEY` / `IAM_JWT_REFRESH_SIGNING_KEY` | 空 | Ed25519 签名私钥（PEM）；开发环境可自动生成临时密钥 |
| `IAM_TOKEN_DIGEST_SECRET` | 空 | Refresh Token 摘要 HMAC 密钥，生产必填 |
| `IAM_ACCESS_TOKEN_TTL` | `900` | Access Token 有效期（秒） |
| `IAM_REFRESH_IDLE_TTL` / `IAM_SESSION_ABSOLUTE_TTL` | `2592000` / `7776000` | 会话空闲与绝对有效期（秒） |
| `IAM_MAIL_BACKEND` | `console` | `console` 或 `smtp`，生产强制 `smtp` |
| `IAM_OPEN_REGISTRATION` | `false` | 是否开放注册；关闭时新用户需管理员审批 |
| `IAM_LOGIN_RATE_LIMIT` 等 | 见 `.env.example` | 登录、注册、邮件与密码重置限流 |

## MCP、Browser MCP 与 Skills

`mcp_config` 可配置一个外部 MCP Server 的 URL、允许工具列表和认证要求。支持通过 OAuth Token Exchange（RFC 8693）以服务端管理的 Subject Token 换取 MCP Access Token，并缓存 Token。

`browser_mcp_enabled=true` 时可加载 Playwright MCP。若未提供显式配置，运行时使用默认的 stdio Playwright MCP 启动方式。Browser MCP 主要用于普通 HTTP 抓取不足时的页面渲染与交互探索。

内置 Skills 当前提供医疗、法律和金融领域的研究与报告上下文包：

```json
{
  "configurable": {
    "skills": ["finance", "legal"]
  }
}
```

MCP、Browser MCP、工具权限、端点白名单和沙箱配置属于管理员安全边界，HTTP 租户请求不能自行覆盖。

## 运行产物

默认运行目录为 `.runs/<run_id>`，主要内容包括：

```text
.runs/<run_id>/
├── public_events.jsonl
├── events.jsonl
├── coordination/
│   ├── leader_lease.json
│   └── ...
└── context/
    ├── manifest.json
    ├── session_memory.jsonl
    ├── research_brief.md
    ├── approved_plan.md
    ├── report_outline.md
    ├── final_report.md
    ├── findings_manifest.json
    ├── artifacts/
    │   └── research_tasks/
    └── ...
```

- `public_events.jsonl`：唯一对 SSE 客户端公开的持久化、脱敏事件源
- `events.jsonl`：内部诊断与研究事件
- `context/manifest.json`：运行状态、下一稳定阶段、配置指纹、所有者和最终工件引用
- `context/session_memory.jsonl`：Query 状态增量和恢复 Journal
- `context/artifacts/research_tasks`：SHA 校验的 Researcher 交接工件
- `context/final_report.md`：最终规范化 Markdown 报告
- `coordination`：Leader Lease、Mailbox、任务状态和并发协调数据

`.runs` 包含研究内容、运行状态和诊断数据，已加入 `.gitignore`。生产环境应对该目录设置访问控制、备份与保留策略。

## 开发与测试

代码检查：

```bash
uv run ruff check
```

类型检查：

```bash
uv run mypy src/open_deep_research
```

当前代码库仍有待收敛的第三方类型桩和静态类型债务；该命令用于执行完整诊断，不代表当前分支已经达到零错误。

运行全部测试：

```bash
uv run pytest
```

测试覆盖 Query Runtime、并行研究、持久化恢复、Lease、Mailbox、HITL、工具治理、Web 证据、提示注入防护、报告 Profile、质量门禁、长期记忆、可观测性和认证。

前端测试（单元 + E2E）：

```bash
cd frontend
pnpm test        # Vitest 单元测试
pnpm test:e2e    # Playwright E2E，覆盖桌面/平板/移动视口
```

## 研究评估

### 本地评估

本地评估不依赖 LangSmith，默认运行一条内置研究问题并使用 Judge 模型评分：

```bash
uv run python tests/run_local_evaluate.py
```

运行最多五条内置问题：

```bash
uv run python tests/run_local_evaluate.py --question-limit 5
```

运行自定义问题：

```bash
uv run python tests/run_local_evaluate.py \
  --question "评估目标行业未来三年的市场规模、竞争结构和主要风险。" \
  --question-title "目标行业研究"
```

结果写入 `tests/local_eval_results/<timestamp>`，包含逐题 JSON、Markdown 报告、聚合评分和评估摘要。脚本还支持 `--resume`、`--rescore-json`、`--refresh-quality-json` 和 `--quality-rigor`。

### LangSmith 与 Deep Research Bench

[`tests/run_evaluate.py`](tests/run_evaluate.py) 使用 LangSmith Dataset 执行研究与 Judge 评估。运行前需要在脚本中确认 Dataset、模型、搜索参数和实验名称，并配置 `LANGSMITH_API_KEY`：

```bash
uv run python tests/run_evaluate.py
```

严格导出完整 Benchmark JSONL：

```bash
uv run python tests/extract_langsmith_data.py \
  --project-name "实验名称" \
  --model-name "模型名称" \
  --dataset-name "deep_research_bench"
```

导出器会核对参考数据集、根运行数量、样本 ID 和最终报告完整性，验证全部记录后才写入 `tests/expt_results`。

深度研究和 LLM-as-Judge 都会产生模型、搜索与抓取费用。运行完整数据集前，请先用少量问题验证模型配置、并发限制、Token 预算和预期成本。

## 运维模型与后台维护

当前生产部署推荐单个 Uvicorn worker。运行记录、API 固定窗口限流、SSE
连接计数和后台保留清扫都包含进程内状态；多 worker 会分别计算这些状态。
SQLite Trace Store 已配置 WAL、5 秒 busy timeout 和 NORMAL synchronous，能够
降低并发写锁冲突，但多 worker 下 Trace 写入仍属于 best-effort，不应把它当作
强一致审计数据库。

启用高级记忆后，可通过系统定时器每天运行一次维护，或者使用持锁循环模式：

```bash
uv run python -m open_deep_research.memory.maintenance daily --loop --interval-hours 24
```

循环在整个生命周期持有 `.runs/memory-maintenance.lock`，第二个维护进程会立即
退出；实际用户维护还会与 API 记忆写入共享 tenant/user 级锁，API 写入默认最多等待
5 秒以吸收短暂竞争。项目级 Mem0 Decay 使用独立的
`.runs/memory-configure-decay.lock`，因此无需停止 daily loop 即可显式变更，例如
`uv run python -m open_deep_research.memory.maintenance configure-decay --enabled`，
不会由运行级配置在请求路径中自动翻转。`docker-compose.yaml` 中提供了默认关闭的
`memory-maintenance` 服务示例。

## 沙箱密钥威胁模型

V7 Worker 不接收真实 Provider、Search 或 MCP 凭据。环境模式凭据仅存在于
Gateway 服务；OAP 的 per-run `apiKeys` 通过内部认证控制面注册到 Gateway
内存 Vault，既不写入 Run manifest，也不进入 payload、日志、结果、制品或
`docker inspect`。Worker 仅持有绑定 run/task/fence/profile/policy digest 且有
期限的 capability token。

## 许可证

本项目采用 [MIT License](LICENSE)。
