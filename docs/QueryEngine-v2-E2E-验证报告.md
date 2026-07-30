# QueryEngine v2 端到端验证报告

## 1. 验证范围

本轮验证于 2026-07-30 使用项目真实模型、Tavily 搜索、质量门控、本地 Judge、RunContext journal v2 和 QueryLoopState v1 完成。

深度研究流程严格限制为三次，未启动第四次运行。

重点检查：

- Supervisor 与 Researcher 的 QueryLoopState 阶段转换和 revision 连续性。
- 模型响应、PendingToolBatch、工具结果、stop governance 和 terminal 边界的持久化。
- 并行 Researcher 的部分失败隔离。
- 工具协议闭合、质量门控、证据恢复和最终报告。
- 外层模型超时是否遵守运行配置。
- 降级报告是否存在无界输出。

## 2. 三次真实运行结果

| 次数 | run_id | 结果 | 研究耗时 | Judge 综合分 | 关键结论 |
|---:|:---|:---|---:|---:|:---|
| 1 | `34daf612-c5b6-4cb7-a37c-d5e63e12ec66` | failed | 635.9 秒 | 0 | Supervisor 批次在 600 秒被整体取消，两个已完成 artifact 未提交 |
| 2 | `a14b4cc9-f478-4331-bfc2-431e7ed4022b` | partial | 647.5 秒 | 0.515 | v2 状态机和工具协议正常；三个 handoff 均被质量门控拒绝，生成 111,203 字符证据恢复报告 |
| 3 | `5e985ed4-f368-4c5c-b098-a50de200c894` | partial | 249.0 秒 | 0.494 | 模型超时和报告上限修复生效；恢复报告降至 19,198 字符并仅展开 40 条证据 |

运行产物：

- `tests/local_eval_results/query-v2-e2e-run1/01_custom-research.json`
- `tests/local_eval_results/query-v2-e2e-run2/01_custom-research.json`
- `tests/local_eval_results/query-v2-e2e-run3/01_custom-research.json`
- `.runs/{run_id}/context/session_memory.jsonl`
- `.runs/{run_id}/public_events.jsonl`
- `.runs/{run_id}/context/artifacts/research_tasks/*.json`

## 3. 已确认符合设计的链路

第 2 次运行中，Supervisor 和三个 Researcher 的 QueryLoopState revision 均从 0 连续递增，无跳号或倒退：

| state_key | revision | turn | terminal |
|:---|:---|:---|:---|
| `supervisor` | `0..12` | `0..4` | `completed` |
| `researcher:call_00_*` | `0..16` | `0..4` | `completed` |
| `researcher:call_01_*` | `0..22` | `0..4` | `completed` |
| `researcher:call_02_*` | `0..22` | `0..4` | `completed` |

第 3 次运行中，Supervisor revision 为 `0..9` 并以 `completed` 结束；Researcher revision 为 `0..14`，在配置的三轮限制下以 `max_turns` 结束。两者都属于合法终止。

三次运行中未自然触发 prompt-too-long、输出截断续写或模型 fallback，对应状态计数均为 0。相关恢复路径仍由确定性故障注入测试覆盖。

已确认：

- 每个 Query state snapshot 都携带 schema version、state key、phase、turn、revision 和 transition reason。
- PendingToolBatch 在工具执行前持久化。
- Researcher 工具结果提交后 revision 单调推进。
- tool-call 与 tool-result 协议在第 2、3 次运行中闭合。
- 安全内容检测产生的 prompt injection 事件不会直接污染共享证据。
- 质量门控失败不会伪装成正常研究成功，而是以 `partial / quality_gate_recovery` 返回。

## 4. 已发现并修复的问题

### 4.1 Supervisor 批次总超时丢弃部分成功结果

第 1 次运行中，三个同步 Researcher 被一个 `task_timeout_seconds=600` 的外层 batch timeout 整体包裹。

两个 Researcher 已完成并分别写出约 0.87 MB 和 1.06 MB artifact；第三个 Researcher 到达第八轮压缩时仍未返回。外层 timeout 取消整个 hook，`query()` 得到 `tool_protocol_violation`，已完成的两个结果也未进入 Supervisor。

修复：

- `task_timeout_seconds` 恢复为单个 Researcher 的超时语义。
- 每个同步 ConductResearch 独立使用 `asyncio.wait_for`。
- 单个超时任务生成闭合的 `ToolErrorType.timeout` ToolMessage。
- 同批其他成功结果继续进入 handoff 评估和 Supervisor update。
- Supervisor batch timeout 增加有界的质量评估与摘要时间，不再与单任务超时发生同刻竞争。

回归测试：

- `test_parallel_sync_handoff_timeout_preserves_completed_results`
- `test_parallel_sync_handoffs_do_not_merge_raw_context_into_supervisor`
- `test_supervisor_research_batch_uses_task_timeout_not_hook_timeout`

### 4.2 外层结构化模型调用未应用 model_call_timeout

第 2 次运行的 `write_research_brief` 耗时约 203 秒，超过 `model_call_timeout_seconds=180`。

原因是 Query 内层模型调用有显式 wait_for，但统一的 `invoke_model_with_retry_observability()` 没有对每次 provider 请求施加超时。研究简报、澄清和其他外层结构化调用因此不受该配置约束。

修复：

- 统一模型调用器对每次 `_ainvoke_model` 使用 `asyncio.wait_for`。
- TimeoutError 继续进入现有错误分类、重试计数和有界退避。
- 第 3 次运行的研究简报约 15 秒完成。

回归测试：

- `test_llm_retry_loop_applies_configured_attempt_timeout`

### 4.3 证据恢复报告无界膨胀

第 2 次运行在 handoff 全部被拒绝后，将 238 条结构化证据全部展开，生成 111,203 字符的恢复报告。Judge 对相关性、结构和完整性均给出 0.2。

修复：

- 最多展开 40 条去重证据。
- 限制 claim、excerpt、locator、source title、gap 和 rejection reason 长度。
- 最多展开 50 个 artifact ref。
- 明确提示其余证据仍保留在 SHA-256 校验后的研究工件中。

第 3 次运行验证：

- 报告长度为 19,198 字符。
- 恰好展开 40 条证据。
- 存在省略提示。
- citation accuracy 为 1.0，source authority 为 0.99。

回归测试：

- `test_evidence_recovery_report_has_a_hard_output_bound`
- `test_rejected_handoff_with_safe_evidence_produces_bounded_partial_report`

## 5. 尚未收口的问题

### 5.1 严格质量门控下无法生成正常用户报告

第 2、3 次运行均完成真实搜索、artifact 持久化和合法 Query terminal，但所有 handoff 被拒绝。

第 3 次的拒绝原因是：虽然 relevance=4、source_quality=5、evidence_coverage=3、corroboration=4，但未取得 Checkpoint 对象的完整字段级结构、Managed Values 是否持久化和明确版本号。

当前行为在安全语义上正确：被拒绝的压缩报告不能直接当成研究结论。但用户体验仍不完整，最终只能得到证据恢复报告，Judge 综合分约 0.49。

建议后续：

- 将质量门控的 required coverage 与用户原始需求绑定，避免 evaluator 自动扩展出用户未要求的字段级证明责任。
- 在 Researcher 到达 max_turns 前，把 `result_assessment.missing_information` 注入最后一次定向检索。
- 对已经达到高 source quality、且仅缺少“负面事实证明”的 handoff，区分 `accepted_with_caveats` 和完全 rejected。
- 允许最终报告生成器只基于安全通过的结构化 evidence records 生成明确标注不确定性的部分回答，而不是原始证据清单。

### 5.2 Public event 的 task completed 语义重复

同步 Researcher 会先发布一次 `research.task.completed` 且 `admission_status=pending`，质量评估后再发布一次 `research.task.completed` 且状态为 accepted 或 rejected。

第 2 次运行有三个任务，却产生六条 completed 事件。虽然 dedupe key 不同，但客户端若仅按 event type 计数，可能误判任务数量。

建议将前一事件改为 `research.task.progress` 或新增 `research.task.awaiting_admission`，只在最终 admission 决策后发布 completed。

### 5.3 真实崩溃恢复仍由故障注入验证

本轮真实运行未主动杀死进程，因此没有在真实 provider 流程中验证从 PendingToolBatch 中途重启。模型提交后、工具批次前、单工具提交后和 stop hook 后恢复，当前由确定性 FakeModel/FakeTool 故障注入测试覆盖。

## 6. 最终回归结果

- QueryEngine v2 与相邻模块回归：`170 passed`。
- Ruff 全量检查：通过。
- mypy 修改文件检查：通过，4 个源文件无类型错误。
- `git diff --check`：通过，仅提示仓库现有的 LF/CRLF 转换策略。
- 源码与测试中未发现遗留的 `[DEBUG-` 临时标记。

## 7. 最终判断

QueryEngine v2 的显式状态、revision、模型与工具边界 checkpoint、合法终止和协议闭合设计在真实端到端运行中表现符合预期。

第 1 次运行发现的批次原子超时是严重可靠性缺陷，现已通过逐 Researcher 超时隔离修复。外层模型调用超时缺口和恢复报告无界膨胀也已修复并由第 3 次真实运行验证。

当前主要剩余风险不在 v2 状态机本身，而在“严格 handoff 质量门控到最终用户答案”的产品语义：系统能够安全、可恢复地结束，但仍可能无法产出正常研究报告。

## 8. 质量门禁 v4 独立验证补充

质量门禁 v4 完成后又执行了一组独立、最多三次的验证序列。该序列的第 3 次运行是本轮最终一次完整深度研究，之后只进行 Artifact 重放和确定性故障注入，没有启动第 4 次研究。

### 8.1 第 3 次正式运行

- run_id：`db5f3f7e-16a7-4923-a261-a2bb0ecc66f8`
- 题目：仅基于 LangGraph 官方文档、官方 API reference 和官方 GitHub 仓库，解释 super-step checkpoint、interrupt/resume、重执行、幂等性，并区分官方保证与工程推断。
- 运行结果：`partial`
- 终止原因：`quality_gate_recovery`
- 研究耗时：945.3 秒
- 质量评估模型：`anthropic:glm-5.2`
- Judge 综合分：0.457085
- handoff：3 个均被拒绝，`evaluator_error` 均为空
- 最终报告：19,412 字符，展开 40 条 SHA-verified evidence；最终报告中的 5 个唯一 URL 均属于 LangGraph 官方域名。

该运行使用的是随后修复前已加载的代码，因此它的 journal 是缺陷证据，不应被描述为修复后的成功 E2E。

### 8.2 该运行暴露并已修复的问题

1. Researcher 的最后一个普通回合如果只有 `think_tool`，当前 update 不含 `result_assessment`，会丢失上一回合的 retry assessment，导致额外定向补证轮未触发。修复后会回退读取持久化状态中的上一份 assessment，并通过 checkpoint 保证恢复次数只消费一次。
2. official-only handoff 投影错误地清空 `compressed_research`，Judge 因此把实际存在的结构、guarantee/inference 标签和 checklist 误判为空。修复后 Judge 可读取候选压缩交付的结构，但事实支持仍只能引用 source-scoped eligible evidence。
3. `[quarantined external content]` 占位文本曾可能以 `security_status=accepted` 进入 eligible evidence。修复后 claim 或 supporting excerpt 命中该标记即 fail-closed。
4. evidence-limited writer 没有把 `EvidenceSynthesisDraft` JSON Schema 发送给模型，真实调用发生 Pydantic `ValidationError`。修复后调用与 token 预算估算共用同一份完整 Schema prompt，未知 evidence ID、越权 URL 和无证据主张仍然 fail-closed。
5. Supervisor 的并行 `ConductResearch`/`ReadResearchArtifact` 只在整个批次结束后聚合提交。修复后每个工具结果独立 checkpoint；在 1/3、2/3 结果提交后崩溃，恢复时不会重调模型、不会重复已提交工具、结果仍按原调用顺序回注。

### 8.3 GLM-5.2 Artifact 正式重放

修复后对第 3 次运行的既有 Artifact 做了只读重放，没有恢复运行或发起新检索：

- task1 的 `compressed_research` 为 8,064 字符，Judge 明确读取，未再出现 empty 类误判。该任务仍因 COV-01 仅部分覆盖和两项具体 unsupported claim 被正确拒绝。
- task2 的 `compressed_research` 为 9,169 字符，同样未出现 empty 类误判。候选交付中的 `forum.langchain.com` URL 被确定性 source-scope 检查拒绝。
- 原 evidence registry 中的 6 条隔离占位副本全部被过滤，只保留同 evidence ID 的 2 条正常副本；Judge 未读取隔离内容。

这证明原先由 handoff 投影丢失造成的误拒绝已经消除。重放后的拒绝来自真实用户 requirement、unsupported claim 或越界来源，而不是 Supervisor advisory 扩张。

### 8.4 最终回归与剩余限制

- 全量 pytest：`738 passed, 3 skipped`
- 联合 QueryEngine v2 / quality-gate v4 回归：`146 passed`
- Ruff：通过
- `git diff --check`：通过
- 本次新增模块以及 QueryEngine/Researcher 大型运行时在 `--follow-imports=skip` 的目标 mypy 检查下通过
- 全仓库 mypy 仍受既有 Pydantic Field 元数据、第三方缺少类型声明和旧消息/工具联合类型债务影响，尚未达到全量通过

由于三次完整研究上限已经用尽，修复后的结论来自 GLM Artifact 重放、逐边界故障注入和全量回归，不宣称已完成第 4 次真实 provider E2E。另一个低风险剩余项是：官方来源约束目前在 evidence admission 和报告阶段严格执行，但候选发现阶段仍可能检索第三方页面，造成预算浪费或候选压缩交付被后续门禁拒绝；最终报告安全边界未被突破。
