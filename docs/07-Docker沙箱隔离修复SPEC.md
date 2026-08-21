# Docker 沙箱隔离修复 SPEC

状态：Approved for implementation
基线：`e43827f`
目标版本：Sandbox Policy / Run Config Schema V7
支持平台：Linux、WSL2；macOS 与原生 Windows 仅预留 Provider 接口

## 1. 背景与问题

当前 Docker 沙箱仅包裹异步 Researcher 的部分执行路径，并且存在以下生产阻断问题：真实任务状态包含不可 JSON 序列化的回调；除 `no-network` 外的网络模式仍使用普通 Docker bridge；Provider 与 AWS 凭据进入容器环境；宿主直接读取容器可写目录并跟随符号链接；取消、更新、超时与恢复无法可靠终止或收养容器；容器结果丢失证据登记字段；安全审批由 Supervisor 工具而不是独立人工决策完成。

本 SPEC 将沙箱改造成三个独立信任域：

- API：运行、预算、审批、IAM 与持久化的唯一权威点。
- Sandbox Controller：唯一持有 Docker Socket，只执行固定生命周期操作。
- Sandbox Gateway：唯一持有模型、搜索和 MCP 凭据，负责 Provider 物理调用、工具 RPC 和受控出网。
- Worker：无真实凭据，仅执行 Researcher loop、压缩与显式允许的本地工具。

## 2. 目标与非目标

### 2.1 目标

- `SANDBOX_ENABLED=true` 时不存在同步研究旁路或宿主静默回退。
- Worker 无法读取真实 Provider/Search/MCP 凭据，也无法直接访问公网、私网或其他任务。
- 文件、网络、命令与工具权限使用统一 Profile 和 `deny > ask > allow` 决策顺序。
- 模型 fallback、熔断、首包探测、token-limit 分类与 usage 捕获只有一份物理实现。
- 预算在任何 Provider 调用前原子预占，RPC 重试不能重复调用或重复记账。
- 多个并发安全审批可持久化、恢复、审计，并受 run owner 与 fence 约束。
- 容器在取消、超时、更新、Lease 丢失或服务重启后不会继续成为孤儿工作负载。

### 2.2 非目标

- 首版不支持 macOS Seatbelt 或原生 Windows Restricted Token/ACL/Firewall Provider。
- 通用 CONNECT 代理不终止 TLS，不检查 HTTPS 明文内容。
- 不提供 HTTP 可选的 `danger-full-access` 或非沙箱逃逸参数。
- 不修复基线中与本特性无关的既有测试失败。

## 3. 威胁模型

受威胁对象包括恶意或被提示注入操纵的 Worker、恶意依赖、异常 Provider 响应、跨租户任务、过期 Lead epoch、伪造 RPC、路径穿越、符号链接/硬链接、DNS rebinding、SSRF、资源耗尽、重放 token 以及进程/宿主重启。

信任假设：API、Controller 和 Gateway 镜像及其管理员配置可信；Docker daemon 与宿主内核属于可信计算基；Worker 镜像内运行时代码按不可信处理；HTTP 租户输入、网页、MCP 描述与工具结果均不可信。

已知风险：Task token 到期与 RPC 时间戳校验使用可信宿主墙钟；能够持续回拨宿主时钟的攻击者已控制可信计算基，超出本威胁模型。生产部署必须保持宿主时钟同步，Controller 的运行时 watchdog 仍使用单调时钟限制单任务执行。

## 4. 强制需求

### 4.1 运行与配置

- **SAN-RUN-001**：`SANDBOX_ENABLED=true` 必须同时要求 `ENABLE_ASYNC_RESEARCH=true`，否则启动和 Run 创建均返回 `sandbox_requires_async_research`。
- **SAN-RUN-002**：沙箱模式下 Supervisor 不得装配同步 `ConductResearch`。
- **SAN-RUN-003**：缺少 Controller、Gateway、镜像 digest、有效策略、Root Signing Key 或运行时依赖时必须 fail-closed。
- **SAN-RUN-004**：Run Config Schema 升为 V7，并冻结 `sandbox_profile_id`、`sandbox_policy_digest`、`sandbox_runtime_digest`、`gateway_protocol_version`。
- **SAN-RUN-005**：V1 至 V6 Run 不允许恢复，返回 `run_schema_not_resumable:sandbox_policy_v7_required`。
- **SAN-RUN-006**：删除全部旧 Docker 沙箱平铺配置；输入包含旧字段时返回迁移错误和本 SPEC 路径。
- **SAN-RUN-007**：V7 还必须冻结 `sandbox_enabled` 与 `enable_async_research`；恢复时不得因进程环境漂移把原沙箱 Run 降级为宿主执行。

### 4.2 控制平面与生命周期

- **SAN-LIFE-001**：API 不得直接访问 Docker Socket；只有 Controller 可以访问。
- **SAN-LIFE-002**：Controller 仅接受固定 Worker/Gateway 镜像 digest、固定命令和策略允许的卷/网络参数。
- **SAN-LIFE-003**：容器、卷与网络必须带 deployment/run/task/fence/profile/policy digest 标签。
- **SAN-LIFE-004**：Controller 只 reconcile 当前 deployment label 命名空间内的资源，不得操作其他容器。
- **SAN-LIFE-005**：正常任务状态机固定为 `PREPARING → RUNNING → RESULT_READY → COLLECTING → STOPPING → EXITED → REMOVED`；超时、取消和崩溃可从 `RUNNING` 直接进入 `STOPPING`，但不得把 retention 与停止语义混合。
- **SAN-LIFE-006**：取消、更新、超时和 Lease 丢失均执行 TERM、5 秒 grace、KILL；retention 只决定已停止资源是否保留。
- **SAN-LIFE-007**：恢复时只收养相同 run/task/current fence 的唯一容器；旧 epoch、重复和无法归属的本部署容器必须终止。
- **SAN-LIFE-008**：Watchdog 必须把每个容器的状态探测放入相互隔离的有界任务；单容器的 Docker exec 阻塞不得阻塞其他容器的 deadline、孤儿与 fence 回收。
- **SAN-LIFE-009**：任务 deadline 从容器实际 `StartedAt` 起算；检查 deadline 前必须先检查结果哨兵，避免边界时刻误杀已完成任务。
- **SAN-LIFE-010**：相同 fence 的运行中容器可以幂等收养，相同 fence 的已停止容器必须清理后重建；create 请求被取消时，Manager 必须等待有界 create 结果并停止已创建资源，消除未登记孤儿窗口。
- **SAN-LIFE-011**：Manager 不得接受或返回动态注入的 Docker SDK client；遗留宿主 bind-mount 执行入口即使被直接调用也必须以 `sandbox_controller_required` 硬失败。

### 4.3 Worker 数据与文件隔离

- **SAN-FS-001**：输入必须使用版本化白名单 `SandboxTaskPayloadV1`，禁止序列化函数、回调、宿主持久化对象和 secret。
- **SAN-FS-002**：结果必须使用 `SandboxTaskResultV1`，完整包含摘要、raw notes、metrics、candidate/document/evidence registries、Web iterations、permission denials 与 coverage 数据。
- **SAN-FS-003**：输入使用只读 task volume；output、logs、artifacts、tmp 使用受 cgroup 约束的 tmpfs。
- **SAN-FS-004**：Collector 必须拒绝绝对路径、`..`、symlink、hardlink、设备、FIFO、socket、超量路径、超量文件和超量字节。
- **SAN-FS-005**：宿主不得向容器可控路径写日志，也不得以跟随链接的文件 API 读取产物。
- **SAN-FS-006**：Worker 不创建 SQLite、事件日志或宿主 checkpoint；运行事件、checkpoint 和 usage 通过 Gateway Runtime RPC 写回 API。
- **SAN-FS-007**：tmpfs 在容器停止时即丢失，因此 Worker 写入版本化结果与有界退出码哨兵后必须保持存活；Controller 只能使用固定参数、无 shell 的只读 `head`/`tar` exec 在停止前回收，并对导出 tar 再执行 SAN-FS-004 全部校验。不得依赖 Docker archive API 穿透 tmpfs，也不得先停止再收集。
- **SAN-FS-008**：读取结果与退出码哨兵前必须以固定参数 `stat` 验证其为有界 regular file；`stat`、`head`、归档收集和客户端请求均必须具有硬超时。FIFO、socket 或其他特殊文件视为 Worker 协议违规并停止容器。

### 4.4 网络与凭据

- **SAN-NET-001**：每个任务使用独立 `internal=true` Docker 网络；Worker 只连接 Gateway，Gateway另接外网。
- **SAN-NET-002**：Worker 不得拥有默认公网路由、宿主 Docker Socket、其他任务网络或 API 控制网络。
- **SAN-NET-003**：通用 HTTP/CONNECT 请求必须认证 task token，并验证域名、端口、全部 DNS 结果与固定连接 IP。
- **SAN-NET-004**：默认拒绝回环、私网、链路本地、组播、保留地址、未指定地址和云 metadata 目标。
- **SAN-NET-005**：普通 HTTP 默认只允许 GET、HEAD、OPTIONS；CONNECT 不解密 TLS，只按目标域名、端口和 IP 决策。
- **SAN-NET-006**：代理必须对主机名执行 IDNA/尾点/IP literal 归一化，要求全部 DNS 结果为全局可路由地址，并拒绝 CGNAT、benchmark 等非公网范围；校验后连接固定 IP，响应状态行和头字段不得包含调用方可控 CR/LF。
- **SAN-SEC-001**：Worker 环境、payload、日志、result、archive 与 `docker inspect` 中不得包含真实 Provider/Search/MCP 凭据。
- **SAN-SEC-002**：Gateway 是唯一 Credential Vault；per-run OAP 凭据只在内存中保存，恢复时必须重新提供。
- **SAN-SEC-003**：一个至少 32 字节的 Base64 Root Key 通过 HKDF-SHA256 派生 task-token、policy-signature 与 service-auth 子密钥。
- **SAN-SEC-004**：Task token 绑定 run、task、fence、profile、policy digest、expiry 和 jti；每个 mutating RPC/代理连接另带 timestamp+nonce，重复 nonce 在 token TTL 内拒绝。
- **SAN-SEC-005**：`sandbox_profile`、Profile 定义和 Gateway 授权主机元数据均属于管理员边界，HTTP metadata/config 不得覆盖。
- **SAN-SEC-006**：`/internal/sandbox/*` 只能位于服务控制网络；边缘 Nginx 必须在通用 `/api/research/*` 转发前显式拒绝该路径，内部 HMAC 不替代网络层隔离。
- **SAN-SEC-007**：API unregister 遇到网络或 5xx 故障必须使用新 nonce 有界重试；Gateway 必须把 Run Credential Vault 条目绑定到已注册 task token 的最大到期时间，周期驱逐时同步擦除 API keys、MCP/OAuth vault 与 operation locks。

### 4.5 模型与预算

- **SAN-BUDGET-001**：Provider SDK、fallback、circuit、first-packet probe、token error、structured output 与 usage capture 只在 Gateway 物理调用点运行。
- **SAN-BUDGET-002**：Worker 只能使用 `GatewayChatModel` 与 wire types，不得导入 Provider SDK 或凭据解析。
- **SAN-BUDGET-003**：Gateway 调用 Provider 前必须向 API Budget Service reserve；结束后 settle/release/mark_uncertain。
- **SAN-BUDGET-004**：所有请求携带 stage、logical operation ID 和确定性 physical attempt ID。
- **SAN-BUDGET-005**：Model Operation Journal 状态机为 `reserved → dispatched → completed|failed|uncertain`。
- **SAN-BUDGET-006**：流中断后 Worker 查询原 operation；已完成则复用结果，已 dispatch 且未知则标记 uncertain，禁止自动重复同一 physical attempt。
- **SAN-BUDGET-007**：Gateway 记录 provider TTFT；Worker 记录 RPC TTFT，两者不得混用。
- **SAN-BUDGET-008**：API 可观测回调识别 `GatewayChatModel` 及其 Runnable 包装并跳过本地模型 BudgetGate；只有 Gateway RemoteBudgetGate 对同一物理调用记账。远程 reserve 必须在 Provider dispatch 前异步完成，settle/fail 在 attempt 结束后异步完成，不得以同步 HTTP 阻塞 Gateway 事件循环。

### 4.6 审批、IAM 与 MCP

- **SAN-APP-001**：API `SecurityApprovalStore` 是唯一审批事实源，并允许一个 Run 同时存在多个 pending approval。
- **SAN-APP-002**：Gateway 使用 `after_version` 长轮询获取决定；最长单次轮询 25 秒。
- **SAN-APP-003**：审批等待上限为 `min(run deadline, profile approval timeout)`，默认 15 分钟，超时 deny。
- **SAN-APP-004**：等待期间 TeammatePool 必须续约 Lead Lease；fence 丢失后旧审批返回 `stale_fence`。
- **SAN-APP-005**：决策只允许 `allow_once`、`allow_run`、`deny`；永久规则只能由管理员修改策略文件。
- **SAN-APP-006**：Supervisor 不得拥有安全审批工具；删除旧 `ApproveResearchDomain` 与进程内 DomainApprovalRegistry。
- **SAN-APP-007**：安全审批使用独立 `security.approval.required/resolved` 事件，不复用计划/大纲 `approval.*` 或单值 `pending_human_action`。
- **SAN-APP-008**：MCP loader、信任校验和 OAuth 在 Gateway 执行；URL elicitation 转为 `mcp_oauth` 安全交互，token 仅进入 Credential Vault。
- **SAN-APP-009**：本地认证 bypass 必须通过统一 ownership checker 工作且不得依赖 IAM 数据库；legacy/无事件 Run 返回空审批队列；Run 进入终态时必须确定性拒绝并清除 pending 审批，禁止 hydrate 复活僵尸卡片。

### 4.7 Developer 权限工具

- **SAN-OS-001**：增加 `ToolExecutionZone.SANDBOX_LOCAL | GATEWAY | HOST_CONTROL`，执行端必须使用权威 schema/effect/permission 重新验证。
- **SAN-OS-002**：ShellExec、ReadFile、WriteFile 默认不装配，仅管理员把 developer/admin 映射到 `developer-workspace` 后启用。
- **SAN-OS-003**：Linux/WSL2 宿主使用 bubblewrap+socat；task workspace 可写，系统运行库只读，home、凭据目录和 `/run` 默认遮蔽。
- **SAN-OS-004**：bubblewrap 子进程继承 PID/IPC/UTS/network namespace；网络只能经过 Gateway Unix Socket。
- **SAN-OS-005**：WSL2 默认不暴露 `/mnt/c` 与 Windows 互操作 socket。
- **SAN-OS-006**：API 自身运行在容器时不嵌套 bubblewrap，Developer 命令改由 Controller 创建短生命周期 command task。
- **SAN-OS-007**：bubblewrap 不得以 `--ro-bind / /` 暴露完整宿主根；只读映射必须收敛到执行所需系统运行库，workspace 单独映射，`/home`、`/root`、`/run`、`/tmp` 与宿主挂载根默认遮蔽。
- **SAN-OS-008**：`sandbox doctor` 在管理员把 `developer-workspace` 映射到非 Linux 平台时必须输出未通过发布验收的显式 warning。

## 5. 策略模型

管理员策略文件映射为：

- `SandboxPolicyBundle(version, deployment_id, role_priority, profile_by_role, profiles)`
- `SandboxProfile(provider, approval_policy, filesystem, network, commands, tools, resources, runtime)`

默认 Profile 为 `research-gateway-only`。`developer-workspace` 只在管理员 role mapping 中显式启用。`approval_policy=never` 的含义是未知操作拒绝，而不是绕过。

默认单任务上限：1 GiB 内存、1 CPU、256 PID、只读 RootFS、UID/GID 65532、64 MiB output、16 MiB logs、256 MiB artifacts、10,000 个文件。Admission Controller 必须按最大并发与 Gateway/Controller 保留量计算最坏内存，资源不足时拒绝新任务。

## 6. 公共与内部接口

公共审批接口：

- `GET /runs/{run_id}/security-approvals?status=pending`
- `POST /runs/{run_id}/security-approvals/{approval_id}`
- 请求体：`{decision: allow_once|allow_run|deny, reason?: string}`

Run Snapshot 增加 `pending_security_approvals[]`。

Controller 经 Unix Socket 提供固定 create/start/stop/status/collect/reconcile 操作。Gateway 提供模型 invoke/stream/operation lookup、权威 tool call、runtime checkpoint/activity/result、Credential Vault 控制接口和 HTTP/CONNECT proxy。

API 内部提供预算 reserve/settle/release/uncertain、Model Operation Journal 与审批 request/wait/resolve 服务。所有内部接口必须使用派生 service-auth key、请求时间戳和 nonce。

## 7. 状态与错误语义

- 沙箱不可用：`sandbox_unavailable:<component>`
- 同步旁路组合：`sandbox_requires_async_research`
- 旧配置：`legacy_sandbox_config_removed`
- 旧 Run：`run_schema_not_resumable:sandbox_policy_v7_required`
- 预算拒绝：保留 `budget_exhausted:<dimension>`
- Provider 结果未知：`model_operation_uncertain`
- 审批超时：`security_approval_timeout`
- 过期 epoch：`stale_fence`
- 网络拒绝：`sandbox_egress_denied`
- 重放：`sandbox_nonce_replayed`

## 8. IAM 与事件迁移

新增权限：

- `research.security_approval.read_own`
- `research.security_approval.resolve_own`
- `research.security_approval.read_any`
- `research.security_approval.resolve_any`
- `research.tool.shell.execute`
- `research.tool.file.read`
- `research.tool.file.write`

权限目录、系统角色 seed、schema current 版本和 Alembic revision 必须同步更新。新端点统一使用 `require_permissions` 与 `require_run_owner`。

## 9. 验收矩阵

M1 验收必须覆盖 SAN-RUN、SAN-LIFE、SAN-FS、SAN-NET、SAN-SEC、SAN-BUDGET 全部需求；M2 覆盖 SAN-APP；M3 覆盖 SAN-OS。

关键攻击测试包括：符号链接、硬链接、路径穿越、特殊文件、FIFO 哨兵、输出炸弹、tmpfs 停止前两阶段回收、逐容器 watchdog 隔离、直接 socket、metadata、私网、CGNAT、benchmark、IDN、DNS rebinding、跨任务横向访问、伪造/重放 token、流中断、重复 settle、Gateway/API/Controller 崩溃、stale fence、同 fence stopped 重建和 create 取消孤儿回收。审批 HTTP 层必须覆盖无 IAM DB bypass、legacy 空投影、终态清理和并发 allow-once。

性能门：Ubuntu CI 上 task container+network 创建 p95 不超过 5 秒；使用确定性假 Provider 的 5 并发研究相对进程内基线总时长增加不超过 20%。

## 10. 基线测试附录

命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=short
```

在 `e43827f` 的实测结果为 `10 failed, 1059 passed, 55 skipped`。本轮不修复下列既有失败，但不得新增失败：

1. `tests/test_auth.py::test_local_dev_auth_bypass_returns_fixed_user`
2. `tests/test_report_genres.py::test_faq_genre_produces_structured_json_artifact`
3. `tests/test_tool_governance.py::TestExecuteGovernedToolCall::test_user_role_tool_blacklist_denied`
4. `tests/test_tool_governance.py::TestExecuteGovernedToolCall::test_user_role_origin_blacklist_denied`
5. `tests/test_tool_governance.py::TestJwtRoleExtraction::test_extract_single_role`
6. `tests/test_tool_governance.py::TestJwtRoleExtraction::test_extract_roles_list`
7. `tests/test_tool_governance.py::TestJwtRoleExtraction::test_extract_dedupes`
8. `tests/test_tool_governance.py::TestJwtRoleExtraction::test_extract_none_metadata`
9. `tests/test_tool_governance.py::TestJwtRoleExtraction::test_extract_empty_metadata`
10. `tests/test_tool_governance.py::TestSupervisorToolsIntegration::test_id_filtering_preserves_same_name_valid_call`

## 11. 上线顺序

- M1 未通过前不得宣称 Docker 沙箱可用。
- M1 上线时未知域名和未知副作用操作一律 deny。
- M2 通过后才启用运行时人工审批。
- M3 通过后才允许管理员映射 `developer-workspace`。
- 沙箱默认关闭，启用前必须通过 `sandbox doctor`。

## 12. 当前实施门状态（2026-08-21）

- M1 的代码闭环、确定性真实 Docker Researcher、两阶段 tmpfs 回收、FIFO/逐容器 watchdog 防阻塞、凭据扫描和本地 doctor 已通过；真实 Docker 预算对账为 8 次物理 attempt、8 次 journal、8 次 reserve 且无本地重复 usage key。生产发布仍必须在 Ubuntu Docker runner 补齐完整网络攻击与 p95 性能门。
- M2 的持久化审批、IAM、API/SSE、前端并发审批和 MCP OAuth elicitation 已接线；独立 Browser Gateway 镜像尚未进入发布门，沙箱模式下不得启用 Browser MCP。
- M3 只完成 execution-zone、默认关闭的 Shell/Read/Write、最小系统只读映射与 bubblewrap fail-closed 基础实现；bwrap+socat Unix Socket 出网和容器化 API 的短生命周期 command task 尚未通过 Linux/WSL2 E2E。默认策略没有把任何角色映射到 `developer-workspace`，管理员不得提前启用。

本轮二次加固后的 Windows 单元/集成测试结果为 `10 failed, 1084 passed, 55 skipped`；失败集合与第 10 节基线完全一致。新增回归覆盖 Controller 生命周期、恶意 FIFO、跨 deployment reconcile、egress proxy、审批 HTTP 层、预算单一记账、Credential Vault TTL/retry 和 transient journal retry。
