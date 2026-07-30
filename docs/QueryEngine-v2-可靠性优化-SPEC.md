# Open Deep Research QueryEngine v2 可靠性优化 SPEC

状态：已实现

版本：1.0

适用运行配置版本：`RUN_CONFIG_SCHEMA_VERSION = 3`

适用 RunContext journal 版本：`schema_version = 2`

适用 Query 状态版本：`QueryLoopState.schema_version = 1`

## 1. 背景与结论

Open Deep Research 原有 QueryEngine 已具备外层阶段恢复、journal/checkpoint、预算、取消、HITL、工具协议闭合、证据持久化与质量门控。其主要可靠性缺口位于 Supervisor 与 Researcher 共用的内层 model/tool loop：循环控制状态没有完整持久化，模型响应与工具结果之间缺少 write-ahead 边界，prompt-too-long、输出截断和模型 fallback 也没有统一、可审计且有硬上限的恢复协议。

QueryEngine v2 将 Claude Code QueryEngine 的关键设计思想映射到当前项目，但保留 Open Deep Research 的领域边界：

- 引入显式、冻结、可序列化的内层循环状态。
- 每次跨越模型、工具、stop hook 或恢复边界前先提交 checkpoint。
- 将继续原因与终止原因分开建模。
- 上下文编译按完整工具协议块进行，避免裁出孤立 tool result。
- 所有恢复路径都有持久化计数和硬上限。
- 保留研究质量门控，不引入通用 token-budget 收尾提示。
- 暂不引入 token 级模型流式和流式工具执行。

journal 提供 at-least-once 保障。系统不承诺外部副作用 exactly-once；危险操作依靠幂等键、禁止自动重放或人工核对控制风险。

## 2. 目标

QueryEngine v2 必须满足：

1. Supervisor 和 Researcher 使用同一套内层状态与恢复协议。
2. 崩溃发生在已支持的 checkpoint 边界时，恢复结果与无崩溃运行一致。
3. 已持久化的模型响应不重复请求。
4. 已提交的工具结果不重复执行、不重复回注。
5. prompt-too-long、输出截断、模型 fallback 均有明确状态、事件和尝试上限。
6. 截断输出不能被当作成功结果。
7. 工具调用与工具结果在投影、压缩、恢复和 fallback 后保持协议有效。
8. 旧 journal 可读取，但不允许按新协议恢复。

## 3. 非目标

本版本不包含：

- token 级模型流式。
- `StreamingToolExecutor`。
- 流式 fallback tombstone。
- token-budget continuation nudge。
- journal v1 到 v2 的 checkpoint 迁移。
- 外部副作用 exactly-once。

## 4. 状态模型

核心类型位于 `src/open_deep_research/agents/query_state.py`。

### 4.1 QueryLoopState

`QueryLoopState` 是冻结 dataclass，字段包括：

- `schema_version`
- `state_key`
- `role`
- `phase`
- `messages`
- `turn`
- `revision`
- `transition_reason`
- `context_recovery`
- `output_recovery`
- `model_route`
- `pending_tool_batch`
- `stop_hook_active`
- `terminal`

`state_key` 的规范值：

- Supervisor：`supervisor`
- Researcher：`researcher:{task_id}`
- 无 task id 的独立 Researcher：`researcher:standalone`

状态只能通过纯函数 `advance(state, action)` 推进。`advance` 使用 `dataclasses.replace` 创建新对象，增加 revision，并拒绝非法 phase 转换、无 terminal outcome 的终态以及携带 terminal outcome 的非终态。

### 4.2 Phase

状态阶段：

- `PREPARING`
- `CALLING_MODEL`
- `EXECUTING_TOOLS`
- `STOP_GOVERNANCE`
- `TERMINAL`

合法转换由 `_LEGAL_PHASE_TRANSITIONS` 明确定义。`TERMINAL` 不允许再转换。

### 4.3 继续原因

- `NEXT_TURN`
- `EXTERNAL_UPDATE`
- `STOP_HOOK_BLOCKING`
- `CONTEXT_REPROJECT_RETRY`
- `REACTIVE_COMPACT_RETRY`
- `OUTPUT_TOKEN_ESCALATE`
- `OUTPUT_CONTINUATION`
- `MODEL_FALLBACK`

### 4.4 终止原因

- `COMPLETED`
- `MAX_TURNS`
- `CANCELLED`
- `BUDGET_EXHAUSTED`
- `DEADLINE_EXCEEDED`
- `MODEL_TIMEOUT`
- `PROMPT_TOO_LONG`
- `OUTPUT_RECOVERY_EXHAUSTED`
- `MODEL_ERROR`
- `TOOL_PROTOCOL_VIOLATION`
- `HOOK_STOPPED`

## 5. 持久化协议

### 5.1 QueryCheckpointSink

`QueryCheckpointSink` 是内层循环唯一依赖的持久化端口：

```python
class QueryCheckpointSink(Protocol):
    async def save(self, state: QueryLoopState) -> None: ...
```

实现包括：

- `RunContextQueryCheckpointSink`：写入运行 journal。
- `CallbackQueryCheckpointSink`：把 Researcher 状态交给异步任务 checkpoint。
- `InMemoryQueryCheckpointSink`：测试和嵌入使用。

checkpoint 是强制 await 的。保存失败时不得越过对应边界继续执行。

### 5.2 journal v2

RunContext 新增 `query_state` record，payload 包含：

- `state_key`
- `revision`
- `transition_reason`
- `state`

replay 按 `state_key` 投影 revision 最新的状态。大消息和已提交工具结果继续使用原有 artifact spillover 编码，避免把大 payload 直接塞入 JSONL。

### 5.3 模型边界

模型成功返回后，QueryEngine 先把 AIMessage 和 `PendingToolBatch` 写入 `EXECUTING_TOOLS` 状态，再开始执行工具。因此恢复到该状态时直接执行待处理工具，不重复调用模型。

无工具响应先提交 `STOP_GOVERNANCE`，再运行 stop hooks。stop hook 注入的消息、updates 对应的内层消息状态会在下一轮前提交。

### 5.4 工具边界

模型工具批次先写入：

```text
PendingToolBatch
  batch_id
  tool_calls
  committed_tool_call_ids
  committed_results
  result_refs
```

默认工具执行器在每个工具返回后立即增加 `committed_tool_call_ids` 和 `committed_results`，并保存新的 `EXECUTING_TOOLS` revision。恢复时：

- 已提交结果直接复用。
- 只读工具可按原 operation id 重放。
- 声明 `supports_idempotency=True` 的工具可按原 operation id 重放。
- 非只读且不支持幂等的未提交工具不自动重放，终止为人工核对路径，并产生 `query.replay_confirmation_required`。

Supervisor 的领域工具批次仍由其批次 hook 管理；其异步 Researcher 任务自身有 task checkpoint 和稳定 task id。该路径仍是 at-least-once，不宣称外部副作用 exactly-once。

### 5.5 Researcher checkpoint

`ResearcherCheckpoint` schema 升至 3，并新增 `query_state`。异步任务 executor 注入 `_query_checkpoint_callback`，使 `ResearcherQueryEngine.ainvoke()` 在每个内层边界保存真实 Query 状态。

恢复时优先使用 task checkpoint 中的 `query_state_snapshot`，而不是从 Researcher 初始输入重新运行。只有受运行 fence 保护的正式恢复流程才会从共享 RunContext journal 加载 Researcher 状态，避免独立测试或新任务误读旧 run 状态。

## 6. ContextCompiler

`ContextCompiler` 位于 `src/open_deep_research/agents/context_compiler.py`，Supervisor 与 Researcher 共用。

### 6.1 预算

真实输入预算：

```text
context_window
- system_prompt_tokens
- tool_schema_tokens
- reserved_output_tokens
- safety_margin_tokens
= available_input_tokens
```

安全余量：

```text
max(2048, context_window * 5%)
```

模型窗口解析优先级：

1. `model_context_window_overrides`
2. 内置模型能力表
3. `unknown_model_context_window_tokens`，默认 32768

未知模型不再静默假设为 200K。

### 6.2 工具协议块

投影单位不是单条 message，而是协议块：

- 普通消息单独成块。
- 包含 tool calls 的 AIMessage 与其连续 ToolMessage 组成一个不可拆分块。

裁剪只选择完整块。投影结束后再次运行 `validate_tool_transcript`。受保护上下文包括前导系统消息与第一条非工具目标消息，最近的完整工具回合按预算从尾部保留。

### 6.3 prompt-too-long 恢复

真实 provider 错误被分类为 `PROMPT_TOO_LONG` 后：

1. 第一次将投影目标乘以 0.8。
2. 第二次执行一次 reactive compact；外部摘要器失败时使用确定性压缩。
3. 后续继续把目标乘以 0.8。
4. 尝试次数由 `context_recovery_max_attempts` 限制，默认 3。
5. 恢复耗尽以 `PROMPT_TOO_LONG` 终止。

如果受保护上下文本身超过模型 envelope，模型调用前的最终复核直接终止，不运行 stop hooks。

## 7. 输出截断恢复

provider 无关分类位于 `model_recovery.py`。识别：

- `PROMPT_TOO_LONG`
- `OUTPUT_TRUNCATED`
- `TRANSIENT`
- `RATE_LIMITED`
- `MODEL_UNAVAILABLE`
- `AUTH`
- `INVALID_REQUEST`
- `INVALID_MEDIA`
- `CANCELLED`
- `UNKNOWN`

当无工具调用的响应以 `length`、`max_tokens`、`max_output_tokens` 或 `model_length` 结束：

1. 若能力表或配置表明模型可提供更大输出，先静默提高一次 max tokens，并丢弃第一次不完整片段。
2. 仍截断时，把片段和内部续写提示写入 Query 状态。
3. 最多续写 `output_continuation_max_attempts` 次，默认 3。
4. 完成后把所有片段合并成一个规范 AIMessage。
5. 内部片段和续写提示从规范历史中移除。
6. 恢复耗尽以 `OUTPUT_RECOVERY_EXHAUSTED` 终止。

`invoke_with_output_recovery()` 把相同协议复用于 Researcher 压缩、单次最终报告以及分节报告的文本 writer。压缩模型恢复失败时仍可进入确定性摘要 fallback；最终报告恢复失败会向外传播，使运行失败，而不是返回半截报告。

## 8. 模型 fallback

fallback 默认关闭；只有 `model_fallbacks` 配置了候选链时启用。

允许 fallback：

- `MODEL_UNAVAILABLE`
- `RATE_LIMITED`
- `TRANSIENT`
- 模型 timeout 在 transport retry 用尽后被归一化为不可用

禁止 fallback：

- `AUTH`
- `INVALID_REQUEST`
- `PROMPT_TOO_LONG`
- `INVALID_MEDIA`
- `CANCELLED`
- 预算或 deadline 终止

跨模型回放前移除 provider 绑定的 reasoning、thinking、signature 和 cache 元数据，同时保留标准消息内容、工具调用和工具结果。

## 9. 工具并发、幂等与 operation id

工具契约新增：

- `concurrency_safe: bool = False`
- `supports_idempotency: bool = False`
- `ToolContext.operation_id`
- `ToolContext.attempt`

operation id：

```text
{run_id}:{state_key}:{turn}:{tool_call_id}
```

调度规则：

- 只有显式 `concurrency_safe=True` 的连续工具组并行执行。
- 其他工具严格按模型调用顺序串行执行。
- 结果始终按原模型调用顺序回注。
- LangChain 只读 adapter 为保持旧行为，默认声明可并发；项目原生工具默认保守串行。

## 10. Stop governance

`StopHookResult` 支持显式：

- `StopAction.CONTINUE`
- `StopAction.COMPLETE`
- `StopAction.HALT`

旧 `should_continue` 字段继续兼容，并通过 `resolved_action` 转换。

以下响应不运行 stop hooks：

- 输出截断
- 模型错误
- prompt-too-long 恢复耗尽
- 未闭合或无效工具协议

领域质量门控仍是 Supervisor 和 Researcher 是否完成的核心判断。

## 11. 配置与版本

新增冻结配置：

- `output_token_escalation_enabled: bool = true`
- `output_continuation_max_attempts: int = 3`
- `model_fallbacks: dict[str, list[str]] = {}`
- `model_context_window_overrides: dict[str, int] = {}`
- `model_max_output_tokens_overrides: dict[str, int] = {}`
- `unknown_model_context_window_tokens: int = 32768`

版本策略：

- 新运行写 `RUN_CONFIG_SCHEMA_VERSION = 3`。
- RunContext 写 journal schema 2。
- QueryLoopState 写 schema 1。
- journal v1 的状态、结果和 artifact 仍可读取。
- journal v1 调用恢复接口返回 HTTP 409，detail 为 `run_schema_not_resumable`。
- 不迁移旧 checkpoint。

REST/SSE 外部协议保持兼容。新增内部 `query.state_changed`，只向调用方提供 revision、phase、reason、turn 等恢复诊断；完整快照不进入公共 SSE payload。

## 12. 可观测性

新增或扩展的内部事件：

- `query.state_changed`
- `query.model_recovery`
- `query.model_fallback`
- `query.replay_confirmation_required`

`query.state_changed` 包含：

- `state_key`
- `revision`
- `phase`
- `reason`
- `turn`

公开事件继续使用既有快照和脱敏逻辑，不暴露完整消息、checkpoint 或 provider 私有元数据。

## 13. 验收测试

新增 `tests/test_query_v2_recovery.py`，覆盖：

- reducer 合法推进、非法终态、不可变性和序列化往返。
- ContextCompiler 完整 envelope 预算与工具协议块保留。
- 未知模型 32768 保守默认。
- prompt-too-long 重投影、reactive compact、恢复耗尽和 stop hook 跳过。
- 输出上限提升、续写、规范合并和恢复耗尽。
- 模型响应 checkpoint 后崩溃恢复不重复调用模型。
- 单个工具结果提交后崩溃恢复不重复执行工具。
- Researcher 从 `EXECUTING_TOOLS` task checkpoint 恢复并忽略 stale 初始输入。
- fallback 白名单和 provider 元数据清理。
- 非并发安全工具严格串行。
- journal v2 最新 Query 状态 replay。
- journal v1 可读但不可恢复。

`tests/test_query_persistence.py` 增加旧 schema 恢复 API 返回特定 409 的测试。

本次验证结果：

- 原 QueryEngine 基线：61/61 通过。
- Query、Researcher、Supervisor、持久化、质量门控和报告综合回归：117/117 通过。
- `ruff check`：通过。
- v2 核心模块隔离 `mypy --follow-imports=skip`：通过，0 错误。
- 全项目 `mypy src/open_deep_research`：未通过，存在 324 个改造前已存在的类型错误，主要位于旧 Configuration Pydantic Field 元数据、MCP 类型、历史工具适配和旧 TypedDict 调用。该基线债务不由本 SPEC 的功能改造引入。

## 14. 仍需后续处理

以下事项不阻塞 QueryEngine v2 上线，但应作为后续工程任务：

1. 清理全项目 mypy 基线，使 CI 能真正以全仓 0 错误作为门禁。
2. 为 Supervisor 自定义 `tool_batch_hook` 增加逐结果提交接口；当前默认工具执行器已逐结果提交，Supervisor 领域批次仍依赖稳定 task id 与 Researcher task checkpoint。
3. 把危险工具重放的人工核对事件扩展为可恢复的通用 HITL action；当前实现安全地禁止自动重放并终止等待人工处理。
4. 增加进程级 kill/restart 的端到端测试，补充现有确定性 checkpoint sink 崩溃注入。
5. 在后续独立版本评估 token 流式与流式工具执行，不能与本版本的恢复状态机隐式耦合。
