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

启用后，新记录写入 `${MEMORY_APP_ID}.v2` 命名空间。旧命名空间只读，不迁移也不修改；召回时旧记录按中性重要性和新鲜性与 v2 结果合并。

Platform 项目会同步开启 Memory Decay。关闭 `MEMORY_ADVANCED_ENABLED` 后，系统停止读取 v2 命名空间，并尝试关闭项目 Decay，从而恢复旧召回路径。配置失败不会阻断研究流程，会记录降级指标并回退到基础记忆。

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

命令使用 `${RUNS_DIR}/memory-maintenance.lock` 保证单实例运行，适合由 cron 或 Kubernetes CronJob 每日调用。重复执行同一日窗口时，已反思的来源 ID、反思指纹、画像来源指纹和每日尝试窗口会阻止重复写入。

## 生命周期

反思由以下任一条件触发：新增未反思观察达到 5 条、重要性总和达到 25，或每日维护发现超过 24 小时的未反思观察。流程先生成最多 3 个问题，再针对每个问题只检索 active v2 observation，最后写入带问题、作用范围、置信度和来源 ID 的 reflection。

canonical profile 在 run 结束检查和每日维护中更新。每个用户、项目和 app 边界内只保留一个 active profile；其他活跃画像会归档。

软遗忘仅归档同时满足以下条件的记录：超过四个类别半衰期未访问、重要性不高于 3、访问次数不高于 1、不是画像且没有未决冲突。归档和被替代记录保留审计历史，默认召回排除。

## 能力降级

Platform v3 支持项目 Decay、typed options、时间参考查询和全用户每日维护。OSS 保留基础读写、应用侧评分、指定用户的 run 结束反思和维护，但不承诺 Platform 时间推理、Decay 或全局用户枚举。
