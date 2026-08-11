# AGENTS.md

## 项目概述

Open Deep Research 是一个可配置的、完全开源的深度研究（Deep Research）智能体，使用手写 QueryEngine/query 双层 Agent Loop 构建。它支持多模型提供商、多种搜索工具和 MCP（Model Context Protocol）服务器，实现自动化研究并生成综合报告。在 [Deep Research Bench](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard) 排行榜上曾获得 #6 排名。

## 常用命令

```bash
# 安装依赖
uv sync

# 启动 FastAPI 开发服务器
uvicorn open_deep_research.server:app --reload --host 127.0.0.1 --port 2024

# 代码检查
ruff check

# 类型检查
mypy

# 运行评估（需要在 tests/run_evaluate.py 中配置模型和参数）
python tests/run_evaluate.py

# 从 LangSmith 提取评估结果用于提交 Deep Research Bench
python tests/extract_langsmith_data.py --project-name "实验名称" --model-name "模型名称" --dataset-name "deep_research_bench"
```

环境配置：复制 `.env.example` 为 `.env`，填入所需的 API 密钥（`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`TAVILY_API_KEY` 等）。

## 核心架构

整个系统使用手写运行时，入口为 `src/open_deep_research/query_engine.py` 中的 `QueryEngine` 和 `src/open_deep_research/deep_researcher.py` 中导出的 `deep_researcher`。工作流分为四个阶段：

### 1. 外层 QueryEngine 流程

```
QueryEngine.submit_message → summarize_messages → memory_recall → clarify_with_user → write_research_brief → supervisor loop → final_report_generation → memory_extract_and_write
```

- **clarify_with_user**：判断用户问题是否需要澄清。如果 `allow_clarification=false` 则跳过，直接进入研究阶段。
- **write_research_brief**：将用户消息转化为结构化的研究摘要（`research_brief`），初始化 supervisor 的上下文。
- **research_supervisor**：一个手写 supervisor loop，负责将研究任务分解并委派给子研究员并行执行。
- **final_report_generation**：汇总所有研究发现，生成最终报告。有 3 次重试机制，每次重试会将输入截断 10% 以应对 token 超限。

### 2. Supervisor 手写循环（研究调度器）

```
supervisor → supervisor_tools → supervisor (循环) 或 → END
```

- **supervisor**：LLM 节点，绑定了三个工具：
  - `ConductResearch`：将研究主题委派给 Researcher runtime
  - `ResearchComplete`：研究完成信号
  - `think_tool`：用于战略规划和反思（必须先于 ConductResearch 调用）
- **supervisor_tools**：执行工具调用。`ConductResearch` 调用会**并行**执行多个 Researcher runtime（并发数由 `max_concurrent_research_units` 控制，默认 5）。退出条件：调用 ResearchComplete、超过 `max_researcher_iterations`（默认 6）或无工具调用。

### 3. Researcher runtime（单主题研究员）

```
START → researcher → researcher_tools → researcher (循环) 或 → compress_research → END
```

- **researcher**：LLM 节点，绑定了搜索工具（Tavily/OpenAI/Anthropic 原生搜索）+ MCP 工具 + `think_tool`。
- **researcher_tools**：并行执行所有工具调用，返回搜索结果。退出条件：超过 `max_react_tool_calls`（默认 10）、调用 ResearchComplete、或无工具调用（包括原生网络搜索已被使用的情况）。
- **compress_research**：将所有研究发现压缩为结构化摘要，保留所有关键信息和引用来源。有 3 次重试，遇到 token 超限会移除较早的消息。

### 4. 状态管理（state.py）

状态定义使用 TypedDict + 自定义 reducer：

| 状态类 | 用途 | 关键字段 |
|--------|------|---------|
| `AgentInputState` | 主图输入 | `messages` |
| `AgentState` | 主图状态 | `messages`, `supervisor_messages`, `research_brief`, `raw_notes`, 
otes`, `final_report` |
| `SupervisorState` | Supervisor 子图 | `supervisor_messages`, `research_brief`, 
otes`, `research_iterations`, `raw_notes` |
| `ResearcherState` | Researcher runtime | `researcher_messages`, `tool_call_iterations`, `research_topic`, `compressed_research`, `raw_notes` |

`override_reducer`：当值包含 `{"type": "override", "value": ...}` 时替换整个字段，否则使用 `operator.add` 追加。

### 5. 配置系统（configuration.py）

`Configuration` 是 Pydantic BaseModel，所有字段可通过以下方式配置（优先级从高到低）：
1. 环境变量（字段名大写，如 `RESEARCH_MODEL`）
2. `RunnableConfig` 中的 `configurable` 字典
3. 代码默认值

关键配置项：
- **模型**：`summarization_model`（摘要）、`research_model`（研究）、`compression_model`（压缩）、`final_report_model`（最终报告），格式为 `provider:model_name`（如 `openai:gpt-4.1`）
- **搜索 API**：`search_api` 枚举（`tavily`/`openai`/`anthropic`/
one`）
- **并发控制**：`max_concurrent_research_units`（默认 5）、`max_researcher_iterations`（默认 6）、`max_react_tool_calls`（默认 10）
- **MCP**：`mcp_config`（URL + 工具列表 + 是否需要认证）、`mcp_prompt`（额外指令）

### 6. 工具系统（utils.py）

- **搜索工具**：`get_search_tool()` 根据 `SearchAPI` 枚举返回不同的搜索工具实现（Tavily 结构化工具、OpenAI `web_search_preview`、Anthropic `web_search_20250305`）
- **MCP 工具**：`load_mcp_tools()` 通过 `MultiServerMCPClient` 加载外部 MCP 服务器工具，支持 OAuth token 交换认证（Supabase JWT → MCP access token）
- **think_tool**：反思工具，用于在研究步骤之间进行战略分析
- **tavily_search**：Tavily 搜索工具，包含并行搜索、去重、LLM 摘要三个步骤。摘要超时 60 秒，失败时返回原始内容
- **token 限制检测**：`is_token_limit_exceeded()` 根据模型提供商（OpenAI/Anthropic/Google）检测不同的 token 超限错误模式
- **MODEL_TOKEN_LIMITS**：硬编码的模型 token 限制表，用于计算截断阈值。注意：此表需要手动维护

### 7. 评估系统（tests/）

评估通过 LangSmith 平台运行，使用 `tests/run_evaluate.py`：
- 6 个评估器（evaluators.py）：`eval_overall_quality`、`eval_relevance`、`eval_structure`、`eval_correctness`、`eval_groundedness`、`eval_completeness`
- 使用 GPT-4.1 作为评估模型（`eval_model`）
- 评估结果通过 `tests/extract_langsmith_data.py` 导出为 JSONL，提交至 Deep Research Bench
- 评估固定使用 Tavily 搜索以保持一致性

### 8. 安全认证（src/security/auth.py）

FastAPI 部署时的认证 dependency：
- 使用 Supabase JWT 进行用户认证
- 通过 `@auth.on.*` 装饰器控制线程（threads）和助手（assistants）的 CRUD 权限
- `on_thread_create`：设置 owner 元数据
- `on_thread_read/delete/update/search`：过滤只允许 owner 访问
- Store 访问通过 namespace 第一段（user identity）进行授权

## 开发注意事项

- 每次执行完E2E验证后，需要关闭验证时启动的前端与后端工作进程，否则可能导致端口占用或资源泄漏
- `configurable_model` 在模块顶层通过 `init_chat_model(configurable_fields=...)` 创建，每次调用时通过 `.with_config()` 传入具体模型配置
- Researcher 由 `ResearcherQueryEngine` 以干净上下文窗口运行，在 supervisor_tools 中通过 `asyncio.gather` 并行调用
- API 密钥获取：`get_api_key_for_model()` 根据 `GET_API_KEYS_FROM_CONFIG` 环境变量决定从环境变量还是 `RunnableConfig` 中读取（OAP 部署时需要设为 `true`）
- Token 超限处理：压缩阶段通过 `remove_up_to_last_ai_message()` 移除最近的消息；最终报告阶段通过字符截断（`model_token_limit * 4` 作为初始值）并逐步缩小
- 添加新模型时，需要在 `MODEL_TOKEN_LIMITS` 字典中注册其 token 限制
- Tavily 搜索的摘要模型独立于研究模型，由 `summarization_model` 配置
- 评估脚本 `run_evaluate.py` 中的模型和参数是硬编码的，每次运行前需要手动调整
- ruff 配置使用 Google 风格的 docstring 规范（`convention = "google"`），测试文件忽略 D 和 UP 规则

