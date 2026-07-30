# Open Deep Research 质量门禁 v4：覆盖约束与安全部分综合 SPEC

## 1. 文档状态

- 状态：已实现，待全量回归验收
- 质量策略版本：`quality-gate-v4`
- Run Config Schema：4
- QueryLoopState Schema：2
- Research Task Artifact Schema：2
- RunContext Journal Envelope：2（保持不变）

本文档描述质量门禁 v4 的验收合同、三态准入、定向补证、覆盖账本和安全部分综合。其目标是在不降低安全与证据完整性要求的前提下，阻止 Supervisor 生成的建议性任务描述漂移为用户未提出的硬性验收条件。

## 2. 问题复现与根因

真实第 3 次 LangGraph 研究运行完成了搜索、证据登记和 Artifact 持久化，最终评估分为：

- relevance：4
- source quality：5
- evidence coverage：3
- corroboration：4

交付仍被拒绝，主要缺口包括 Checkpoint 对象逐字段结构、Managed Values 是否持久化、精确 LangGraph 版本号，以及逐字段 guarantee/inference 标注。

其中部分内容来自 Supervisor 生成的 `research_topic`，而不是用户原始问题。旧实现直接把 `research_topic` 当作完整验收合同，并使用二元 `accepted` 判断；因此任何建议性缺口都可能拒绝整份交付。最后一个 Researcher 回合产生的新缺口也没有补证机会。当全部 handoff 被拒绝时，系统只能输出确定性证据清单。

根因可归纳为四项：

1. 用户需求与 Supervisor 建议没有数据边界。
2. Judge 同时承担事实评分与最终 policy 决策，缺少确定性 reducer。
3. handoff 只有接受和拒绝两种状态，无法表达“用户必答项已满足但存在软性限制”。
4. 拒绝路径缺少受限的结构化综合能力。

## 3. 设计目标与非目标

### 3.1 设计目标

1. 只有原始 HumanMessage 能产生硬性 coverage requirement。
2. Research brief、Supervisor 规划和 research topic 只能产生 advisory dimension。
3. 每个 Researcher 只对显式分配给自己的 requirement IDs 负责。
4. Judge 提供评分和候选缺口，最终准入由确定性 policy reducer 计算。
5. 对满足用户要求但仍有软性缺口的非高风险任务，允许 `accepted_with_caveats`。
6. 正常工具回合耗尽后，最多执行配置允许的一次定向补证。
7. 全部 handoff 拒绝但仍有 SHA 校验后的 eligible evidence 时，生成安全的结构化部分报告。
8. v3 运行、QueryLoopState v1 和 Research Task Artifact v1 保持可读，并继续沿用旧语义。

### 3.2 明确不放宽

- Prompt injection、安全状态、Artifact SHA-256 和最小来源数仍是硬门禁。
- unsupported claim 永远不能通过 caveat 准入。
- 用户明确 requirement 为 partial 或 unsupported 时必须拒绝。
- 被拒绝的 compressed research、raw notes 和 Researcher 自由文本不得进入安全部分综合。
- 高风险研究不能使用 caveat 准入。
- evaluator、grounding Judge、协议或截断错误继续 fail closed。
- REST 路径和 SSE event type 保持不变。

## 4. 用户覆盖合同

### 4.1 数据模型

```python
class CoverageRequirement(BaseModel):
    requirement_id: str
    text: str
    source_message_index: int
    source_start: int
    source_end: int


class ResearchCoverageContract(BaseModel):
    schema_version: int = 1
    original_query_sha256: str
    requirements: tuple[CoverageRequirement, ...]
    advisory_dimensions: tuple[str, ...] = ()
```

模型为冻结 Pydantic 对象，可 JSON 序列化。`original_query_sha256` 对规范化后的原始 HumanMessage 计算摘要，用于检测合同与输入是否匹配。每个 requirement 保留消息索引及字符范围，使验收责任能够追溯到用户原文。

### 4.2 构建规则

- 合同只读取原始 HumanMessage。
- 编号列表、项目符号和独立句子按确定性规则提取。
- requirement ID 按源顺序稳定生成。
- Research brief 和 `research_topic` 可以写入 `advisory_dimensions`，但不能修改 `requirements`。
- 合同写入主状态、Supervisor 状态、Researcher 状态和 schema 2 任务 Artifact。
- 运行上下文额外持久化 `context/coverage_contract.json`。

### 4.3 工具边界

`ConductResearch` 和异步 `StartResearchTask` 均携带 `requirement_ids`。在 quality-gate-v4 运行中：

- requirement ID 必须存在于当前 coverage contract。
- 未知 ID 返回 `unknown_coverage_requirement_ids` 工具参数错误。
- Researcher 的评估范围为 owned requirement IDs。
- v3 和 Artifact v1 不存在 coverage contract 时继续使用旧二元语义，不进行静默升级。

## 5. 三态准入模型

### 5.1 AdmissionStatus

```python
AdmissionStatus = Literal[
    "accepted",
    "accepted_with_caveats",
    "rejected",
]
```

兼容字段 `accepted` 的映射如下：

| admission_status | accepted |
|---|---:|
| accepted | true |
| accepted_with_caveats | true |
| rejected | false |

### 5.2 RequirementCoverage

```python
class RequirementCoverage(BaseModel):
    requirement_id: str
    status: Literal["supported", "partial", "unsupported"]
    evidence_ids: tuple[str, ...]
    explanation: str
```

Judge 必须为 owned requirement 生成 requirement-to-evidence 映射。`supported` requirement 至少绑定一个存在于 Artifact evidence registry 的 evidence ID。未知 requirement ID 和不存在的 evidence ID 作为协议错误处理。

### 5.3 确定性 Policy Reducer

最终准入由 `resolve_handoff_admission()` 计算。Judge 的 requested status 仅作为建议，不能越过以下硬拒绝规则：

- 安全、Artifact 完整性、来源数量或确定性检查失败。
- 存在 unsupported claim。
- 任一 owned requirement 为 partial、unsupported 或缺失。
- 用户明确的来源、时间、格式或范围约束未满足。
- 任一评分低于 rigor dimension floor，或平均分低于 average floor。
- fail-closed evaluator/协议失败。
- 存在影响用户必答项的未解决冲突。
- 高风险任务请求 caveat 准入。

`accepted_with_caveats` 仅在全部 owned requirement 为 supported、硬检查和评分通过、无 unsupported claim 且任务非高风险时成立。不能关联到用户 requirement 的缺口被降为 caveat 或 follow-up suggestion。

## 6. 高风险识别

配置：

```python
quality_risk_mode: Literal["auto", "high", "standard"] = "auto"
quality_caveat_admission_enabled: bool = True
```

`auto` 使用版本化中英文规则识别医疗、法律和金融建议，并可结合运行中启用的 medical、legal、finance skill。识别结果只保存风险类别、规则 ID 和等级，不额外保存敏感文本。

- `high`：强制高风险，只允许完全接受或拒绝。
- `standard`：强制标准风险，但其他硬门禁不变。
- `auto`：命中规则时关闭 caveat 准入。

风险识别只能收紧策略，不能放宽质量或安全要求。

## 7. Coverage Ledger 与完成治理

Supervisor 维护：

```python
coverage_ledger: dict[str, {
    "status": "supported" | "partial" | "unsupported",
    "evidence_ids": list[str],
    "task_ids": list[str],
    "caveats": list[str],
}]
```

账本由 handoff assessment 的 requirement coverage 确定性合并，并随 Supervisor checkpoint 持久化。状态优先级为 supported、partial、unsupported；证据、任务和 caveat 去重保存。

完成治理检查整个 coverage contract：

- 所有 requirement supported：允许完整完成。
- 存在未覆盖 requirement 且仍可研究：继续调度。
- 达到预算、期限或轮次上限但有 eligible evidence：`complete_partial`。
- 即使某个 Artifact 已重新评估通过，只要全局 ledger 仍有用户 requirement 未覆盖，也必须保持 `complete_partial`，不得误标为完整成功。

## 8. 定向补证恢复轮

配置：

```python
quality_gap_recovery_max_attempts: int = 1
```

QueryLoopState schema 2 新增：

```python
class QualityRecoveryState:
    attempts: int
    active: bool
    target_requirement_ids: tuple[str, ...]
    triggering_assessment_revision: int | None
```

触发条件必须全部成立：

1. 正常 Researcher 工具回合达到 `max_react_tool_calls`。
2. owned requirement 尚未被支持。
3. result assessment 包含具体 missing information 和 suggested queries。
4. 恢复次数未达到上限。
5. 未取消，且 QueryEngine 的预算与期限守卫仍允许下一次调用。

恢复消息只包含 requirement IDs、缺口、建议查询和已有 evidence 摘要，不重复完整上下文。恢复状态在下一次模型调用前通过 Query checkpoint sink 提交。崩溃恢复后复用相同状态和 operation ID，不重复消费恢复次数或重复提交已完成工具结果。恢复回合结束后 `active=false`，随后必须重新评估并进入压缩。

## 9. 报告路径

### 9.1 accepted

通过现有正常报告生成链路。`quality_gate.status=passed`。

### 9.2 accepted_with_caveats

通过正常报告生成链路，报告编排器强制检查并补充“限制与不确定性”章节。`quality_gate.status=passed_with_caveats`。最终引用仍受 evidence URL allowlist 约束。

### 9.3 Evidence-limited partial synthesis

全部 handoff 被拒绝但存在 SHA-256 校验后的 eligible evidence 时，调用受限 writer。writer 只能读取：

- coverage contract；
- eligible evidence records；
- coverage ledger；
- caveats；
- uncovered requirement IDs；
- 证据 URL 白名单。

禁止传入 rejected compressed research、raw notes 和 Researcher 自由文本。

writer 输出：

```python
class EvidenceBoundClaim(BaseModel):
    text: str
    evidence_ids: list[str]
    qualification: str | None


class EvidenceSynthesisDraft(BaseModel):
    title: str
    summary: str
    summary_evidence_ids: list[str]
    sections: list[EvidenceSynthesisSection]
    unresolved_requirements: list[str]
```

提交前执行四项验证：

1. 所有 evidence ID 存在且 eligible。
2. 所有 requirement ID 存在于 coverage contract。
3. 所有 URL 属于 evidence allowlist。
4. 每个事实主张绑定至少一个 evidence ID。

随后由独立 grounding Judge 检查主张是否被证据支持。writer 或 Judge 不可用、输出截断、未知 ID、越权 URL、无证据主张或 grounding 失败时，一律回退现有确定性证据恢复报告。运行状态仍为 `partial`。

### 9.4 无 eligible evidence

保持失败，不生成研究结论。

## 10. 持久化与崩溃恢复

质量 v4 状态跨以下边界持久化：

- coverage contract 建立后；
- ConductResearch 或 StartResearchTask 分派后；
- Researcher 正常回合评估后；
- 定向补证激活且 attempts 增加后；
- 每个工具结果提交后；
- handoff assessment 和 coverage ledger 合并后；
- stop governance 和终态提交前。

QueryLoopState v1 读取时，`quality_recovery` 默认：

```json
{
  "attempts": 0,
  "active": false,
  "target_requirement_ids": [],
  "triggering_assessment_revision": null
}
```

任务 Artifact schema 2 保存 coverage contract、owned requirement IDs 和风险结果。Artifact v1 仍可 SHA 校验、读取和重新评估，但没有 coverage contract 时必须使用 v3 二元准入。

## 11. 配置冻结与兼容性

新增冻结字段：

- `quality_risk_mode`
- `quality_caveat_admission_enabled`
- `quality_gap_recovery_max_attempts`

schema 4 将上述字段纳入 run fingerprint。恢复运行不能更改风险模式、caveat 策略或补证次数。

项目保留 `RUN_CONFIG_FROZEN_FIELDS_V3`。schema 3 运行恢复时：

- 不要求提供 v4 新字段；
- 保留原 `quality_policy_version`；
- 不启用 caveat 准入；
- 不在恢复期间把 v3 assessment 静默升级为 v4。

外部 REST 路径和 SSE event type 不变。`research.task.completed.payload` 可新增 `admission_status=accepted_with_caveats`，`quality_gate.status` 可新增 `passed_with_caveats`。对外只暴露 requirement ID、caveat 数量、恢复次数和最终状态。

## 12. 安全不变量

1. Artifact 必须在当前恢复流程中按预期 SHA-256 重新读取和校验。
2. eligible evidence 必须通过安全状态过滤。
3. rejected compressed research 和 raw notes 不得进入 evidence-limited writer。
4. 任一事实主张必须绑定 eligible evidence。
5. 报告 URL 必须属于 evidence allowlist。
6. unsupported claim 永远硬拒绝。
7. 高风险任务不能 accepted with caveats。
8. evaluator 或 grounding Judge 失败时 fail closed。
9. 定向补证次数具有硬上限。
10. 未覆盖用户 requirement 的运行不能标记为完整成功。

## 13. 实现映射

| 能力 | 主要实现 |
|---|---|
| Coverage contract、风险与 reducer | `src/open_deep_research/quality_contract.py` |
| Handoff 模型与 evaluator | `src/open_deep_research/quality.py` |
| QueryLoopState schema 2 | `src/open_deep_research/agents/query_state.py` |
| 补证 checkpoint 状态转换 | `src/open_deep_research/agents/query.py` |
| Supervisor/Researcher 状态传播 | `src/open_deep_research/agents/deep_researcher.py` |
| 终止恢复与 evidence-limited 路由 | `src/open_deep_research/agents/query_engine.py` |
| Coverage-aware 完成治理 | `src/open_deep_research/completion.py` |
| 安全部分综合 | `src/open_deep_research/report/evidence_synthesis.py` |
| caveat 报告编排 | `src/open_deep_research/report/orchestrator.py` |
| 异步任务状态传播 | `src/open_deep_research/tasks/` |
| 配置冻结与版本 | `src/open_deep_research/configuration.py` |

## 14. 测试与验收矩阵

| 场景 | 验收点 |
|---|---|
| 第 3 次运行 replay | Supervisor 的精确版本号和旧域名不成为硬 requirement |
| 用户明确要求版本 | 缺失版本必须 rejected |
| owned requirement | Researcher 不因其他任务 requirement 被拒绝 |
| 未知 requirement ID | 工具参数校验失败并要求重试 |
| soft gap | 标准风险任务可 accepted with caveats |
| unsupported claim | 永远 rejected |
| 高风险关键词 | 关闭 caveat 准入 |
| 正常回合耗尽 | 最多增加配置允许的一次补证 |
| 补证崩溃恢复 | 不重复消费次数、模型调用或已提交工具结果 |
| coverage ledger 恢复 | Supervisor resume 后内容一致 |
| caveat 报告 | 正常 writer 输出包含限制章节 |
| evidence-limited 输入 | 不含 rejected summary 和 raw notes |
| 非法 writer 输出 | 未知 evidence ID、越权 URL、无证据主张均回退 |
| v3/v1 兼容 | 冻结配置、QueryLoopState 和 Artifact 沿用旧语义 |
| 全局未覆盖 | 已接受部分 Artifact 时仍返回 partial |

确定性 replay fixture 位于 `tests/fixtures/quality_gate_v4_run3_replay.json`，保留第 3 次运行的用户原始需求、Supervisor 扩展任务和 v3 缺口，不依赖本机 `.runs` 目录。

最终验收命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
```

验收成功标准：

- 所有现有与新增测试通过。
- ruff 与 mypy 无错误。
- 第 3 次运行 replay 不再因 Supervisor 新增要求误拒绝。
- 未覆盖用户需求时返回正常结构的 partial report。
- 所有引用来自 eligible evidence allowlist。
- 不存在 rejected 自由文本进入报告、unsupported claim 被 caveat 接受或补证无限执行的路径。
