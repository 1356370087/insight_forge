# Open Deep Research 前端展示模块 SPEC

## 1. 目标与边界

在仓库内以独立 `frontend/` 提供面向内部演示并可继续产品化的研究工作台。前端只消费 FastAPI 的稳定公开契约，不引入 LangGraph SDK，不移植候选项目后端，不在浏览器管理模型 API Key、MCP、沙箱、白名单或管理员安全配置。

技术栈为 Next.js 16 App Router、React 19、TypeScript strict、pnpm、Tailwind CSS 4、shadcn/ui 风格基础组件、assistant-ui、Zustand、TanStack Query、`@microsoft/fetch-event-source`、React Hook Form、Zod、next-intl、Supabase SSR、react-markdown、remark-gfm、rehype-sanitize、Vitest、Testing Library、MSW 和 Playwright。

## 2. 页面与视觉

路由包含 `/login`、`/research/new`、`/research/[runId]` 和 `/settings`。桌面使用 280px 左栏、自适应中栏和 360px 右栏；小于 1280px 时右栏进入抽屉，小于 768px 时左栏也进入抽屉。

视觉采用深色工业研究指挥舱：近黑石墨背景、分层灰蓝表面、低对比网格与点阵、IBM Plex Sans 与 IBM Plex Mono，中文回退 Noto Sans SC。青绿色表示活动，琥珀色表示人工等待，红色表示失败，灰绿色表示完成。状态同时使用图标、文字和颜色，并尊重 `prefers-reduced-motion`。

## 3. 核心交互

创建任务采用 `POST /runs` 后导航并订阅 `GET /runs/{runId}/events`。工作区展示六阶段轨道、按 wave 分组的任务、实时来源、按任务更新的 Findings、运行摘要和警告。

`approval.required` 在原位显示安全 Markdown，支持批准、修改和取消；`clarification.required` 在同一 run 内暂停并回答，回答后直接继续。运行中支持取消、全局方向反馈和证据追问。完成后支持过程/报告切换、引用安全跳转、Markdown 下载、打印/PDF 和 artifact 展示。

## 4. 后端公开契约

`GET /capabilities` 返回事件 schema 版本、功能开关、显式 `FRONTEND_EDITABLE_CONFIG_KEYS` 白名单、JSON Schema、默认值和 UI 元数据，不暴露任何密钥或管理员字段。

`GET /runs?limit=&cursor=&status=` 从持久化 `RunManifest` 返回当前用户的历史，按 `created_at` 倒序并使用不透明 cursor。manifest 增加 `title`、`query_preview`、`idempotency_key` 和 `pending_human_action`，旧数据缺失标题时回退 run ID。

`POST /runs` 接受可选 title 和 `Idempotency-Key`。`GET /runs/{runId}` 保留旧字段并增加 title、时间、稳定 output 和可恢复 pending action。human action 支持 `approve | revise | answer | cancel`，并严格匹配 clarification 或 approval 类型。

公开事件 schema 为 v2，新增 `clarification.required`、`clarification.resolved`，并扩展 `approval.required` 的 `content_markdown` 与 `allowed_actions`。公开 projection 包含 status、pending action、任务投影和去重来源。客户端继续接受 v1，未知事件只进入诊断日志。

## 5. 同 run clarification

当 `clarify_with_user` 需要澄清时，QueryEngine 创建持久化 `pending_human_action`，状态进入 `awaiting_clarification`，发布 `clarification.required` 并停在可恢复 checkpoint。收到 answer 后把 AI 问题和 Human 回答保留在消息状态，发布 `clarification.resolved`，继续 `write_research_brief`。cancel 进入标准 cancelled 终态。

计划与大纲审批也把 pending action 写入 manifest，浏览器刷新后从 snapshot 恢复操作卡。

## 6. 前端状态与流恢复

`ResearchRunStore` 包含 runId、title、status、connectionState、currentStage、stageProgress、plan、wavesById、tasksById、sourcesById、findingsByTaskId、pendingHumanAction、report、artifacts、qualityGate、warnings、lastEventId、isHydrated 和 isReconnecting。

事件 reducer 是纯函数：小于等于 lastEventId 的事件忽略；任务、来源和 Findings 按稳定 ID upsert；terminal 事件停止重连并回源一次最终 snapshot；刷新先加载 snapshot，再从服务器 last_event_id 订阅。SSE 支持心跳忽略、断线退避、401 单次 session 刷新和 409 cursor-ahead 快照校准。

## 7. 配置中心

配置分为基础研究、模型、搜索与证据、Agent 调度、HITL、报告、质量门禁和长期记忆。字段来自 capabilities 白名单并以开关、滑块、数字框、文本框或枚举选择器呈现。

设置使用版本键 `odr.frontend.settings.v1`，支持默认恢复与 JSON 导入导出。导入会按 capabilities schema 校验并剔除未知字段。JWT、研究内容和管理员配置不进入 localStorage。

## 8. 认证、部署与安全

本地默认前端 3000、FastAPI 2024，CORS 只允许配置的 localhost Origin。`LOCAL_DEV_AUTH_BYPASS=true` 时显示醒目标识并绕过登录。生产使用 Supabase SSR，access token 通过 Authorization Bearer 附加到 REST 和 fetch-based SSE。

同源部署的浏览器 API 基址为 `/api/research`。反向代理需移除此路径前缀，并对 SSE 关闭缓冲和缓存、延长读取超时。

Markdown 禁止原始 HTML并经过 sanitize；外链只接受 http/https 并添加 `noopener noreferrer`。服务端继续执行 configurable 与 metadata 的最终安全校验。

## 9. 测试与验收

后端覆盖 owner 隔离、分页排序、旧 manifest 回退、capabilities 白名单、clarification 暂停/回答/取消/恢复、human action 类型匹配、SSE v1/v2、cursor-ahead 和 CORS/认证边界。

前端覆盖 reducer 映射与去重、SSE 重连、任务波次、来源、Findings、HITL、配置导入导出、Markdown XSS、中英文和键盘操作。Playwright 覆盖本地完整运行、刷新恢复、clarification、计划/大纲审批、反馈、失败恢复、Supabase session 和 1440/1024/390 三种布局。

验收要求：所有进度由真实公开事件驱动；刷新和短暂断网不产生重复卡片；pending action 可恢复；无 LangGraph SDK 与浏览器模型密钥；桌面 Lighthouse Accessibility 不低于 90；除 clarification 暂停语义和公开契约外不修改研究 Agent 核心调度与报告逻辑。
