# Open Deep Research 多智能体深度研究系统

## 项目介绍

面向复杂、开放式问题构建的可配置多智能体深度研究系统。系统采用 Orchestrator-Workers 架构，由主 Agent 负责需求澄清、研究规划、任务拆解与报告汇总，多个 SubAgent 在独立上下文中并行完成资料检索、证据分析和结果压缩，最终生成带可追溯引用的结构化研究报告。项目兼容 OpenAI、Anthropic、DeepSeek 等多模型提供商，支持 Tavily、模型原生搜索及 MCP 工具扩展，并围绕异步协作、断点恢复、沙箱安全、上下文与长期记忆、可观测性和 LLM-as-Judge 评估完成工程化建设；项目曾在 Deep Research Bench 排行榜取得第 6 名。

## 技术栈

- **核心开发：** Python 3.10+、asyncio、FastAPI、Uvicorn、Pydantic、LangChain Core
- **大模型与智能体：** OpenAI、Anthropic、DeepSeek、Google Gemini/Vertex AI、结构化输出、Function Calling、手写 Agent Loop
- **搜索与工具生态：** Tavily、OpenAI/Anthropic 原生搜索、MCP、HTTPX、BeautifulSoup、PyMuPDF
- **任务协作与持久化：** JSON/JSONL、SQLite、Portalocker 文件锁、原子写入、任务租约与 Checkpoint
- **安全与认证：** Docker Sandbox、Supabase JWT、域名白名单、SSRF 防护、Prompt Injection 内容防火墙
- **上下文与记忆：** Mem0 Platform/OSS、会话摘要压缩、运行日志回放、分租户记忆检索
- **可观测性与评估：** Langfuse、Prometheus、Grafana、LangSmith、Helicone、LLM-as-Judge、Deep Research Bench
- **工程质量：** Pytest、Ruff、Mypy

## 项目职责描述

- **Multi-Agent 双层运行时：** 设计外层 `QueryEngine` 与可复用内层 `query` 循环。外层串联消息压缩、长期记忆召回、需求澄清、研究摘要生成、Supervisor 调度、报告生成和记忆写入；内层通过 Hook 化的 Model-Tool Loop 统一承载 Supervisor 与 Researcher 的多轮推理、工具执行、停止条件和状态更新，降低不同 Agent 运行逻辑的重复实现。

- **同步与异步 SubAgent 调度：** 在同步模式下支持 Supervisor 单轮发起多个 `ConductResearch`，通过 `asyncio.gather` 并行执行独立研究任务，并使用并发上限避免资源失控；在异步模式下提供任务创建、查询、列表、更新、取消和等待工具，由可复用 Teammate Pool 后台执行 SubAgent，使主 Agent 可在子任务运行期间继续规划、接收用户反馈或处理其他研究方向。

- **可靠协作与中断恢复：** 实现基于 JSON 文件邮箱和 Portalocker 文件锁的 Lead-SubAgent 通信机制，采用原子写入、`fsync`、消息去重、优先级、Claim/ACK、超时租约、失败重投和 Dead Letter 保证跨进程消息可靠性；通过 Leader Lease 与心跳机制防止同一研究任务被多个 Orchestrator 同时接管，并利用任务 Checkpoint、会话 Journal 和结果 Artifact 恢复未完成任务，跳过已完成阶段与重复搜索，支持任务中断、动态指令更新和显式 Resume。

- **渐进式搜索提示词优化：** 针对 Agent 容易生成过长、约束过多查询的问题，设计“先广泛探索、再基于证据逐步收窄”的搜索策略：首轮使用 1～3 个短而宽泛的查询建立信息全景，识别关键术语、核心实体、权威来源与争议点；每轮搜索后通过反思工具评估结果质量，再仅围绕真实证据缺口增加时间、地域、实体或来源类型等必要约束，弱结果场景主动放宽查询，提升搜索召回率和研究覆盖度。

- **研究压缩与质量门禁：** 每个 Researcher 使用独立上下文完成检索，并将结果压缩为保留事实、来源、时间、冲突与不确定性的结构化摘要；完整工具记录、原始笔记、候选来源、抓取文档和证据注册表持久化为带 SHA-256 校验的研究 Artifact，主 Agent 默认只接收摘要，必要时可按 Section 和 Offset 限量回读原始证据。增加独立评估模型对工具结果与 SubAgent Handoff 进行质量判断，低质量结果不进入 Supervisor 共享上下文，而是返回具体缺口和后续查询建议，减少错误信息在多 Agent 间传播。

- **Docker 沙箱与网络权限流水线：** 为异步 Researcher 实现可配置 Docker 沙箱，隔离每个任务的输入、输出、临时文件与日志目录，并通过只读 RootFS、非 Root 用户、删除 Linux Capabilities、`no-new-privileges`、CPU/内存/PID 配额、执行超时和清理策略限制运行权限；设计 `no-network`、`allow-search-only`、`allowlist-domain`、`open-network` 四级网络策略，结合静态域名白名单、运行时域名审批、连接目标校验和 SSRF 防护控制外部访问。

- **Prompt Injection 与信任边界防护：** 将网页、搜索和 MCP 返回内容统一视为不可信数据，清理 Script、Iframe、Form 等主动标记，检测指令覆盖、角色冒充、工具诱导、凭据窃取和分隔符逃逸等注入模式，并将可用事实封装为带来源、哈希、截断与隔离状态的 Evidence Envelope；同时禁止客户端伪造 System/Tool 消息、身份字段和管理员安全配置，对长期记忆与最终 Markdown 再次执行注入过滤和危险协议清理，形成输入、工具结果、记忆、模型上下文和报告输出的分层防护。

- **上下文、计划与长期记忆管理：** 基于模型上下文上限和 Token 比例触发会话压缩，完整保留 System Prompt、研究摘要及最近消息窗口，并保证 AI Tool Call 与 Tool Result 成对保留；将研究摘要、批准后的研究计划、报告大纲、任务结果和最终报告写入运行目录，通过 Journal 重放恢复主流程。以 Mem0 Platform v3 为高级记忆主路径，通过 `user_id + project_id + app_id` 和独立 v2 命名空间实现租户隔离与旧记忆兼容，仅从用户明确表达及通过多来源质量门禁的证据中写入可信观察，并将重要性与可信度分离；结合相关性、重要性和时间衰减进行可解释重排与访问强化，构建“显著问题生成—相关观察检索—高层洞察写回”的两阶段反思、canonical 研究画像、冲突消解、Platform Decay 和软归档闭环，同时以功能开关、Fail-Open 降级及单实例每日维护命令保障平滑上线。支持研究计划与报告大纲的人工审批、修订和运行中反馈下发。

- **统一工具治理与预算控制：** 抽象统一 Tool 协议和注册中心，为工具标注来源、读写副作用、鉴权状态与可重试属性；在模型绑定前按 Supervisor/Researcher 角色、工具白名单、来源策略和用户角色黑名单过滤，在执行阶段依次完成工具存在性检查、Pydantic Schema 校验、参数边界校验、敏感操作审批、网络出口检查和调用执行。对网络超时、429、503 等瞬时故障执行指数退避重试，并返回稳定的结构化错误；通过 Agent 轮次、工具调用次数、并发量、搜索/抓取次数、模型输出 Token 和上下文压缩阈值实施多层预算控制。

- **结构化 Web 证据与报告生成：** 补充“候选发现—标准化去重—语义重排—Top-K 抓取—HTML/PDF 解析—证据抽取”的 Web Pipeline，将搜索候选与可引用证据分离，只允许成功抓取的文档进入最终引用，提升报告可追溯性。构建注册表驱动的报告产品层，支持一次性生成与分章节生成策略，可输出 Markdown、结构化 JSON、Slides 和 One-Pager，并提供执行摘要、决策简报、FAQ、对比矩阵、优缺点分析和文献综述等报告类型，以及引用去重和编号/BibTeX-like 格式渲染。

- **可观测性与运维：** 以自建 `TraceRecorder` 统一记录 Run、Agent、LLM 和 Tool 多级 Span，并将链路、脱敏后的输入输出预览、Token 用量、缓存命中、估算成本、质量评分和异常信息写入本地 SQLite，同时可镜像到 Langfuse；通过 Prometheus 暴露低基数 Counter、Gauge 和 Histogram，覆盖运行/模型/工具耗时、Token 与成本、失败和重试次数、限流、空结果、搜索零来源、任务排队与重分配、证据数量、引用密度、来源覆盖率和报告长度等指标，支持 Grafana 看板与告警分析。

- **系统评估与质量闭环：** 打通本地评估与 LangSmith 数据集评估流程，能够批量运行深度研究任务、持久化报告与中间证据，并采用 LLM-as-Judge 从整体质量、相关性、结构与连贯性、正确性、完整性、Groundedness、引用准确性、来源权威性和工具效率等维度评分；设计非补偿式质量门禁，对证据、引用和来源等关键指标设置最低阈值，避免综合平均分掩盖严重短板，并支持基于已持久化证据单独重新评分，无需重复执行高成本研究流程。

## 项目职责描述（精简版）

- **Multi-Agent 双层运行时：** 设计外层 `QueryEngine` 与可复用内层 `query` 循环，统一编排需求澄清、研究规划、Supervisor 调度、工具执行、报告生成及记忆写入。

- **同步与异步 SubAgent 调度：** 同步模式通过 `asyncio.gather` 并行执行多个研究任务；异步模式基于 Teammate Pool 提供任务创建、查询、更新、取消与等待能力，使主 Agent 与 SubAgent 并行工作。

- **可靠协作与中断恢复：** 基于 JSON 文件邮箱、Portalocker 文件锁和 Claim/ACK 租约实现可靠通信，并结合 Leader Lease、Checkpoint 与会话 Journal 支持进度追踪、任务中断和断点恢复。

- **渐进式搜索提示词优化：** 设计“先宽泛探索、再基于证据逐步收窄”的搜索策略，通过搜索后反思、弱结果主动放宽和按真实缺口增加约束，提高搜索召回率与研究覆盖度。

- **研究压缩与质量门禁：** SubAgent 返回保留来源与不确定性的压缩摘要，完整研究记录持久化为带哈希校验的 Artifact；使用独立评估模型拦截低质量结果并给出补充研究建议。

- **Docker 沙箱与网络权限流水线：** 为异步 Researcher 提供资源受限的 Docker 隔离环境，并通过多级网络策略、域名白名单、运行时审批和 SSRF 防护控制外部访问。

- **Prompt Injection 与信任边界防护：** 将网页、搜索和 MCP 内容封装为不可信 Evidence Envelope，检测并隔离指令覆盖、角色冒充和凭据窃取等注入，同时保护客户端输入、记忆和报告输出。

- **上下文、计划与长期记忆管理：** 实现保留研究摘要和工具调用边界的 Token 感知上下文压缩，持久化研究计划与报告大纲；基于 Mem0 v2 分租户记忆构建可信观察、三信号检索、访问强化、周期反思、canonical 画像及 Decay/软遗忘闭环，并通过功能开关和 Fail-Open 降级兼容旧记忆与 OSS，结合 HITL 完成计划审批和运行中反馈。

- **统一工具治理与预算控制：** 建立工具注册、角色权限、参数校验、敏感操作审批、出口检查和失败重试流水线，并从 Agent 轮次、并发量、工具调用、抓取次数和 Token 等维度控制预算。

- **结构化 Web 证据与报告生成：** 构建候选发现、去重重排、Top-K 抓取和证据抽取流水线，仅允许已抓取证据进入引用；支持多种报告类型、生成策略、输出格式和引用样式。

- **可观测性与运维：** 使用自建 `TraceRecorder`、Langfuse 和 Prometheus 采集多级 Span、Token、成本、延迟、失败、重试、任务队列及报告质量指标，为 Grafana 看板和告警提供数据。

- **系统评估与质量闭环：** 打通本地与 LangSmith 评估流程，使用 LLM-as-Judge 从报告质量、完整性、Groundedness、引用和工具效率等维度评分，并通过关键指标门禁形成质量闭环。
