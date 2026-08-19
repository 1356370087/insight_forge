# Mem0 高级长期记忆运维说明

## 启用方式

高级记忆默认关闭。Platform v3 推荐配置如下：

```dotenv
ENABLE_MEMORY=true
MEM0_PROVIDER=platform
MEM0_API_KEY=...
MEMORY_APP_ID=open_deep_research
MEM0_MEMORY_PROJECT_ID=your-project
MEMORY_AGENT_ID=lead_researcher
MEMORY_ADVANCED_ENABLED=true
```

启用后，新记录写入 `${MEMORY_APP_ID}.v2` 命名空间。默认只召回 v2 记录，避免不可更正的 legacy 旧偏好与新记录并列注入。迁移期确需双读时可临时设置 `MEMORY_LEGACY_RECALL_ENABLED=true`，完成迁移后应关闭。

Memory Decay 是 Platform project 级设置，不会再由单个请求或 run 自动翻转。部署管理员需显式执行一次：

```bash
python -m open_deep_research.memory.maintenance configure-decay --enabled
```

需要关闭时使用 `--disabled`。该命令使用独立的 `${RUNS_DIR}/memory-configure-decay.lock`，不会被长期持有 `memory-maintenance.lock` 的 daily loop 阻塞。`MEMORY_DECAY_ENABLED` 仅表示部署期望值，不会在请求路径写入 Platform。

## 信任边界

可写入的 observation 只有以下来源：

- 用户明确表达的长期偏好、事实和项目约束。
- `evidence_registry` 中由两个不同来源域名支持，或由一个权威来源支持的高置信度 claim。

网页正文、工具原始输出、整份报告、未经验证的结论、凭据和指令形文本不会写入。预提取记录统一使用 `infer=False`，避免 Mem0 再次改写。

画像只覆盖研究协作偏好、专业领域、重复主题和项目背景。健康、政治、宗教、种族、心理状态等敏感推断会被提示约束和应用侧过滤共同移除。

## 每日维护

运行所有 Platform 用户：

```bash
python -m open_deep_research.memory.maintenance daily
```

运行指定用户，OSS 必须使用此方式：

```bash
python -m open_deep_research.memory.maintenance daily --user-id USER_ID
```

预览而不写入：

```bash
python -m open_deep_research.memory.maintenance daily --user-id USER_ID --dry-run
```

命令使用 `${RUNS_DIR}/memory-maintenance.lock` 保证 CLI 单实例运行，并与 API 写入共享 tenant/user 级文件锁。API 写入默认有界等待 5 秒，可通过 `MEMORY_MUTATION_LOCK_TIMEOUT_SECONDS` 调整；维护任务仍采用非阻塞争锁。重复执行时，已处理标记、反思指纹和画像来源指纹会阻止重复计算与写入。完整维护默认不在 run 终态关键路径执行；如确需 run-end 后台维护，可显式设置 `MEMORY_RUN_END_MAINTENANCE_ENABLED=true`。召回发现已有 observation 但没有 active canonical profile 时会记录 `memory.profile_missing`，用于发现漏配 daily 维护的部署。

## 生命周期

反思由以下任一条件触发：新增未反思观察达到 5 条、重要性总和达到 25，或每日维护发现超过 24 小时的未反思观察。流程先生成最多 3 个问题，再针对每个问题只检索 active v2 observation，最后写入带问题、作用范围、置信度和来源 ID 的 reflection。

canonical profile 由每日维护更新。每个用户、项目和 app 边界内按 `profile_version` 和更新时间确定性选择一个 active profile；其他活跃画像会取消 canonical 标记并归档。

未决冲突记录 fail-closed，不参与召回、反思或画像生成。新的矛盾会加入目标已有的 open conflict group；同时命中多个组时会合并完整组并记录来源组，避免成员被动离组后被误判为已裁决。只有后续明确的 `SUPERSEDE`/`TEMPORAL_CHANGE` 才会成组关闭冲突并保留审计字段，单成员 open 组不会由维护任务自动“洗白”。软遗忘不再永久豁免冲突记录，只归档超过四个类别半衰期未访问、重要性不高于 3、访问次数不高于 1 且不是画像的记录。归档和被替代记录保留审计历史，默认召回排除。

## 能力降级

Platform v3 支持项目 Decay、typed options、时间参考查询和全用户每日维护。OSS 保留基础读写、应用侧评分、时间参考查询和指定用户维护，但不支持项目 Decay 或全局用户枚举。OSS 元数据更新会重新写入 embedding，因此召回访问强化默认跳过；全量列表达到适配器硬上限时会显式失败，避免在截断视图上执行生命周期写入。
