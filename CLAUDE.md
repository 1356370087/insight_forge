AGENTS.md

## 开发注意事项

- 运行 Claude Code、Codex 等编程类 Agent 时，尽量避免同时启动多个 subagent；优先使用单个 subagent 或串行执行，以免触发并发限制，导致 HTTP 429 错误或频繁重试