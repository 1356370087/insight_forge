# AGENTS.md

## 项目概述

InsightForge（仓库名与 Python 导入命名空间仍为 `open_deep_research`）是一个可配置的、完全开源的深度研究（Deep Research）多 Agent 平台，使用手写 QueryEngine/query 双层 Agent Loop 构建。它支持多模型提供商、多种搜索工具和 MCP（Model Context Protocol）服务器，实现自动化研究并生成带来源的结构化研究报告。在 [Deep Research Bench](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard) 排行榜上曾获得 #6 排名。

仓库包含三个主要部分：

- `src/open_deep_research/`：Python 研究运行时与 FastAPI 服务
- `src/security/`：自有身份与 RBAC 子系统（IAM，已完全移除 Supabase）
- `frontend/`：Next.js 研究控制台（包管理器为 pnpm）

## 常用命令

```bash
# 安装依赖
uv sync

# 启动 FastAPI 开发服务器
uv run uvicorn open_deep_research.server:app --reload --host 127.0.0.1 --port 2024

# 代码检查
uv run ruff check

# 类型检查（当前仍有待收敛的类型债务，不代表零错误）
uv run mypy src/open_deep_research

# 运行全部后端测试
uv run pytest

# 本地评估（不依赖 LangSmith，默认运行一条内置问题）
uv run python tests/run_local_evaluate.py

# LangSmith 评估（需要在 tests/run_evaluate.py 中配置模型和参数）
uv run python tests/run_evaluate.py

# 从 LangSmith 提取评估结果用于提交 Deep Research Bench
uv run python tests/extract_langsmith_data.py --project-name "实验名称" --model-name "模型名称" --dataset-name "deep_research_bench"
```

前端（`frontend/` 目录）：

```bash
cd frontend
pnpm install
pnpm dev          # http://localhost:3000，需先启动后端并复制 .env.example 为 .env.local
pnpm test         # Vitest 单元测试
pnpm test:e2e     # Playwright E2E
```

IAM（启用完整认证时需要 PostgreSQL）：

```bash
uv run alembic upgrade head      # 建表并 seed 权限目录与系统角色
uv run python -m security.cli bootstrap-admin --email admin@example.com --password '...'
```

容器化部署：`docker compose up --build` 启动 api + frontend + nginx 代理（宿主机 `http://localhost:8080`），默认为免 IAM 的本地演示模式。

环境配置：复制 `.env.example` 为 `.env`，填入所需的 API 密钥（`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`TAVILY_API_KEY` 等）。样例中还包含 IAM/JWT/SMTP/限流等自有身份配置段。

## 核心架构

整个系统使用手写运行时，入口为 `src/open_deep_research/agents/query_engine.py` 中的 `QueryEngine` 和 `src/open_deep_research/agents/deep_researcher.py` 中导出的 `deep_researcher`。工作流分为四个阶段：

### 1. 外层 QueryEngine 流程

```
QueryEngine.submit_message → summarize_messages → memory_recall → clarify_with_user → write_research_brief → plan_approval（可选）→ supervisor loop → outline_approval（可选）→ final_report_generation → memory_extract_and_write
```

- **clarify_with_user**：判断用户问题是否需要澄清。如果 `allow_clarification=false` 则跳过。澄清采用暂停-回答-继续式 HITL，暂停点会持久化到检查点。
- **write_research_brief**：将用户消息转化为结构化的研究摘要（`research_brief`），并编译覆盖契约（Coverage Contract，含稳定需求 ID）。
- **plan_approval / outline_approval**：`enable_human_in_loop=true` 时分别挂起等待研究计划和报告大纲审批，支持 approve/revise/cancel。
- **research_supervisor**：一个手写 supervisor loop，负责将研究任务分解并委派给子研究员并行执行。
- **final_report_generation**：委托给 report 产品系统（`open_deep_research.report.build_report`），`default` 类型复现原单次综合；单次综合仍有 3 次重试，token 超限时渐进截断输入；终态写作失败会让整个运行记录为失败而不是写入错误字符串。

### 2. Supervisor 手写循环（研究调度器）

```
supervisor → supervisor_tools → supervisor (循环) 或 → END
```

- **supervisor**：LLM 节点，绑定的工具：
  - `ConductResearch`：将研究主题委派给 Researcher runtime（同步并行路径）
  - `StartResearchTask`、`CheckResearchTask`、`WaitForResearchUpdates`、`ReadResearchArtifact`：异步任务路径，由 `enable_async_research=true` 开启，Teammate Pool 独立执行，状态与工件写入文件系统
  - `ResearchComplete`：研究完成信号
  - `think_tool`：用于战略规划和反思
- **supervisor_tools**：执行工具调用。`ConductResearch` 调用会**并行**执行多个 Researcher runtime（并发数由 `max_concurrent_research_units` 控制，默认 5）。退出条件：调用 ResearchComplete、超过 `max_researcher_iterations`（默认 6）或无工具调用。委派调用会携带 `requirement_ids`（该任务硬性负责的覆盖需求子集），未认领的需求会兜底轮询分配。

### 3. Researcher runtime（单主题研究员）

```
START → researcher → researcher_tools → researcher (循环) 或 → assess_research_results → compress_research → END
```

- **researcher**：LLM 节点，绑定了搜索工具 + MCP 工具 + `think_tool`。`web_pipeline_mode=enforced`（默认）时搜索工具为 `web_research` 与 `fetch_url`（Search → Top-K Fetch → Extract → Evidence 流水线）。
- **researcher_tools**：并行执行所有工具调用，返回搜索结果。退出条件：超过 `max_react_tool_calls`（默认 10）、调用 ResearchComplete、或无工具调用。
- **assess_research_results**：`quality_evaluation_enabled=true` 时评估工具结果批次。
- **compress_research**：将所有研究发现压缩为结构化摘要，保留关键信息和引用来源，并执行证据登记与来源契约校验；遇到 token 超限会移除较早的消息。

### 4. 状态管理（state.py）

状态定义使用 TypedDict + 自定义 reducer：

| 状态类 | 用途 | 关键字段 |
|--------|------|---------|
| `AgentInputState` | 主图输入 | `messages` |
| `AgentState` | 主图状态 | `messages`、`supervisor_messages`、`research_brief`、`raw_notes`、`notes`、`final_report`，以及 `human_feedback`、`coverage_contract`、`coverage_ledger`、`evidence_registry`、`handoff_assessments`、`report_artifacts` 等 |
| `SupervisorState` | Supervisor 子图 | `supervisor_messages`、`research_brief`、`coverage_contract`、`coverage_ledger`、`research_iterations`、`raw_notes`、`handoff_assessments` 等 |
| `ResearcherState` | Researcher runtime | `researcher_messages`、`tool_call_iterations`、`research_topic`、`requirement_ids`、`compressed_research`、`evidence_registry`、`raw_notes` 等 |

`override_reducer`：当值包含 `{"type": "override", "value": ...}` 时替换整个字段，否则使用 `operator.add` 追加。

### 5. 配置系统（configuration.py）

`Configuration` 是 Pydantic BaseModel，所有字段可通过以下方式配置（优先级从高到低）：
1. 环境变量（字段名大写，如 `RESEARCH_MODEL`）
2. `RunnableConfig` 中的 `configurable` 字典
3. 代码默认值

关键配置项：
- **模型**：`summarization_model`（摘要）、`research_model`（研究）、`compression_model`（压缩）、`final_report_model`（最终报告）、`quality_evaluation_model`（Judge），格式为 `provider:model_name`（如 `openai:gpt-4.1`）
- **搜索 API**：`search_api` 枚举（`tavily`/`openai`/`anthropic`/`none`）
- **Web 证据管线**：`web_pipeline_mode`（`legacy`/`shadow`/`enforced`，默认 `enforced`）
- **并发控制**：`max_concurrent_research_units`（默认 5）、`max_researcher_iterations`（默认 6）、`max_react_tool_calls`（默认 10）
- **质量门禁**：`quality_evaluation_enabled`、`quality_evaluation_rigor`（五档）、`quality_evaluation_min_sources`、`quality_evaluation_max_input_chars`（Judge 输入预算，默认 30000）
- **MCP**：`mcp_config`（URL + 工具列表 + 是否需要认证）、`mcp_prompt`（额外指令）

运行开始后，影响恢复一致性的关键配置会被冻结并写入运行清单。

### 6. 工具系统（tools/）

工具代码位于 `src/open_deep_research/tools/`（`adapters.py` 统一 Tool 协议、`governance.py` 权限治理与 MCP 装配、`token_store.py` Token 缓存、`utils.py` 工具实现）：

- **搜索工具**：`get_search_tool()` 根据 `SearchAPI` 枚举返回不同的搜索工具实现（Tavily 结构化工具、OpenAI `web_search_preview`、Anthropic 原生 Web Search）
- **MCP 工具**：`load_mcp_tools()` 通过 `MultiServerMCPClient` 加载外部 MCP 服务器工具，支持 OAuth Token Exchange（RFC 8693，服务端管理的 Subject Token → MCP access token）
- **think_tool**：反思工具，用于在研究步骤之间进行战略分析
- **tavily_search**：Tavily 搜索工具，包含并行搜索、去重、LLM 摘要三个步骤。摘要预算 120 秒（含重试），失败时隔离外部内容
- **token 限制检测**：`is_token_limit_exceeded()` 根据模型提供商（OpenAI/Anthropic/Google）检测不同的 token 超限错误模式
- **模型解析层**：`model_resolution.py` 统一 provider 推断、API key/base URL、兼容参数、模型配置和惰性模板；`configuration.py` 与 `tools/utils.py` 中的旧入口仅保留兼容 shim
- **模型回退底层**：`model_fallback.py` 与 `model_errors.py` 位于 `agents`/`tools` 之下，统一负责候选链、错误分类、跨 provider 消息清洗和 `query.model_fallback` 公共事件；禁止从这两个低层模块顶层反向导入 `agents` 或 `tools`
- **MODEL_TOKEN_LIMITS**：硬编码的模型 token 限制表，用于计算截断阈值；查找采用精确键优先、再按键长度降序的最长子串匹配。注意：此表需要手动维护

### 7. 质量与证据（quality.py / evidence.py / quality_contract.py）

- **覆盖契约**：从用户消息编译需求清单，每条需求有稳定 ID（`COV-NN-<hash>`）；门禁只对任务归属需求做硬覆盖检查
- **来源契约**：用户消息中的显式 URL 白/黑名单、"仅官方来源"约束会被编译为来源准入范围（`SourceScope`）；受限契约下证据准入 fail-closed
- **覆盖账本**：按需求 ID 跨任务合并 `supported`/`partial`/`unsupported` 状态（单调提升）
- **质量门禁**：`evaluate_tool_results`（内层，工具批次）与 `evaluate_subagent_handoff`（外层，交接）结合确定性硬门禁与 Judge 语义评分；Judge 输入受字符预算约束

### 8. 评估系统（tests/）

LangSmith 评估使用 `tests/run_evaluate.py`，本地评估使用 `tests/run_local_evaluate.py`，两者共用 `tests/evaluators.py` 的核心评估器：
- 10 个评估器：`eval_overall_quality`、`eval_relevance`、`eval_structure`、`eval_correctness`、`eval_evidence_integrity`、`eval_groundedness`、`eval_completeness`、`eval_citation_accuracy`、`eval_tool_efficiency`、`eval_execution_compliance`
- 评估结果通过 `tests/extract_langsmith_data.py` 导出为 JSONL，提交至 Deep Research Bench
- 评估固定使用 Tavily 搜索以保持一致性

### 9. 安全认证（src/security/rbac/）

FastAPI 部署时的认证与授权（Supabase 已完全移除；`src/security/auth.py` 仅保留兼容别名）：
- 本地账号体系：Argon2 密码哈希、邮箱验证、注册审批、密码重置、会话管理
- JWT：EdDSA（Ed25519）签名的 Access/Refresh 双 Token，独立密钥与 kid，Refresh 一次性轮换并检测重用，`authz_version` 支持即时吊销
- RBAC：4 个系统角色（`viewer`/`researcher`/`developer`/`admin`）+ 自定义角色，基于封闭权限目录构建权限矩阵；入口为 `require_permissions()` / `get_current_principal`
- 运行与任务级所有权通过 `require_run_owner()` / `require_task_owner()` 校验
- `LOCAL_DEV_AUTH_BYPASS=true` 且 `APP_ENV=development` 时返回合成 researcher/developer 身份，跳过 JWT/DB
- 数据库迁移位于 `src/security/rbac/migrations/`；CLI 入口为 `python -m security.cli`

### 10. 前端（frontend/）

- Next.js（App Router）+ React + TypeScript，包管理器为 pnpm，端口 3000
- 浏览器只连接 Next.js BFF（`/api/research`、`/api/auth`、`/api/iam` 路由处理器）；Token 保存在 HttpOnly Cookie，由服务端代理注入 `Authorization` 头
- 主要页面：研究创建（`/research/new`）、运行工作区（SSE 实时进度 + HITL 审批 + Subagent 活动抽屉）、设置、登录/注册、管理后台（`/admin`）

## 开发注意事项

- 每次执行完E2E验证后，需要关闭验证时启动的前端与后端工作进程，否则可能导致端口占用或资源泄漏
- `configurable_model` 由 `model_resolution.get_configurable_model_template()` 提供进程级惰性单例，每次调用时通过 `.with_config()` 传入具体模型配置
- Researcher 由 `ResearcherQueryEngine` 以干净上下文窗口运行，在 supervisor_tools 中通过 `asyncio.gather` 并行调用
- API 密钥获取：统一由 `model_resolution.resolve_api_key()` 处理；`tools/utils.py:get_api_key_for_model()` 是兼容 shim。`GET_API_KEYS_FROM_CONFIG` 决定从环境变量还是 `RunnableConfig` 读取（OAP 部署时需要设为 `true`）
- 七个模型角色支持同名角色级 API Key 覆盖：`SUPERVISOR_API_KEY`、`RESEARCHER_API_KEY`、`SUMMARIZATION_API_KEY`、`MESSAGE_SUMMARY_API_KEY`、`COMPRESSION_API_KEY`、`FINAL_REPORT_API_KEY`、`QUALITY_EVALUATION_API_KEY`
- 模型 fallback：`model_fallbacks` 支持 `supervisor`、`researcher`、`summarization`、`message_summary`、`compression`、`final_report`、`quality_evaluation` 七个角色，仅对限流、瞬态错误和模型不可用切换
- Token 超限处理：压缩阶段通过 `remove_up_to_last_ai_message()` 移除最近的消息；最终报告阶段（`report/assembly.py`）通过渐进截断重试（最多 3 次）
- 添加新模型时，需要在 `MODEL_TOKEN_LIMITS` 字典（`tools/utils.py`）中注册其 token 限制
- Tavily 搜索的摘要模型独立于研究模型，由 `summarization_model` 配置
- 评估脚本 `run_evaluate.py` 中的模型和参数是硬编码的，每次运行前需要手动调整
- ruff 配置使用 Google 风格的 docstring 规范（`convention = "google"`），测试文件忽略 D 和 UP 规则
- `.runs` 目录包含运行数据与 Trace Store，已加入 `.gitignore`，不要提交
- 前端统一使用 pnpm（不要用 npm/yarn 生成锁文件）；`NEXT_PUBLIC_*` 变量在构建期烧入客户端代码
