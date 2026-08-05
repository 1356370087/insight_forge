# Agent 面试业务亮点分析：Docker 沙箱与网络治理

## 一、业务亮点简述

这项工作的重点不是简单地"用 Docker 跑 Agent"，而是为并发执行的异步 Researcher 建立明确的运行边界。

Researcher 需要处理不可信网页、调用外部工具并持续运行，如果缺少隔离，单个任务的文件误写、资源耗尽或恶意网络请求可能影响主服务、宿主机和同批研究任务。

项目在开启沙箱功能后，将完整的 Researcher Runtime 放入任务级 Docker 容器，并从五个维度进行治理：

- **文件隔离：** 每个任务使用独立工作区，输入只读挂载，输出、临时文件、日志和制品按用途开放写入。
- **最小权限：** RootFS 默认只读，使用非 Root 用户，删除全部 Linux Capabilities，并启用 `no-new-privileges`。
- **资源治理：** 对 CPU、内存、PID、并发任务数和执行时间设置上限。
- **网络治理：** 提供从禁止联网到开放联网的四种策略，对受治理的工具调用执行域名检查和运行时审批。
- **安全抓取：** 在 HTTP 抓取链路中校验 URL、DNS 解析地址、可获取的真实连接对端及重定向目标，降低 SSRF 风险。

它带来的业务价值是：在保留 Agent 外部搜索能力的同时，缩小单任务故障半径，并让任务能够按 `run_id/task_id` 取消、清理、归档和排查。



## 二、源码体现出的核心设计

### 1. 任务级沙箱与生命周期

异步任务经 Teammate Pool 进入执行器；当 `enable_docker_sandbox` 开启时，执行器将该任务交给 `DockerSandboxManager`。Manager 按 `run_id/task_id` 创建独立工作区，并检查解析后的路径不能逃逸配置根目录。

输入目录只读，其他目录按用途可写；只读 RootFS 用于限制容器可写层，只允许任务写入明确挂载的目录。容器使用非 Root UID/GID，并组合 `cap_drop=ALL` 与 `no-new-privileges` 降低提权风险。

容器创建时设置 CPU、内存和 PID 配额，任务系统同时限制并发量和运行时间。任务主动取消时会根据容器 ID 执行停止和强制删除；正常结束后归档输出与日志，并按清理策略回收临时目录和容器。宿主侧另行记录沙箱生命周期事件，避免只依赖容器内可写日志进行排查。



### 2. 四种网络策略⭐

| 网络模式 | 核心行为 | 适用场景 |
|---|---|---|
| `no-network` | Docker 使用 `network_mode=none`，并拒绝联网搜索和远程 MCP 配置 | 使用本地能力的离线、高敏感任务 |
| `allow-search-only` | 允许受控的模型、搜索和 Web 研究链路，URL 型工具仍经过出口治理 | 常规深度研究，默认网络策略 |
| `allowlist-domain` | 受治理的 URL 型工具只访问静态允许或运行时批准的域名 | 只需访问少量指定站点的任务 |
| `open-network` | 跳过应用层域名白名单；受控 HTTP 抓取仍保留 SSRF 校验 | 外部已有出口防护或确需广泛联网的环境 |

其中两个中间模式侧重点不同：`allow-search-only` 偏向限制可使用的研究链路，`allowlist-domain` 偏向限制访问目标，不能把它们理解为严格的线性强弱关系。

静态允许域名来自人工配置，也会根据研究模型、摘要模型、搜索提供商和 MCP 地址自动推导必要主机，减少安全策略开启后的配置成本。



### 3. 运行时域名审批⭐

当受治理的 URL 工具访问未知域名时，系统以 `(run_id, domain)` 为键查询审批结果。

进程内 Researcher 会进入 `WAITING_FOR_CONFIRMATION`，等待 Supervisor 通过 `ApproveResearchDomain` 批准或拒绝；决定在同一次研究运行内共享。进程内缓存会在最后一个活动任务结束后清理，跨进程决定则保存在对应 `run_id` 的协调目录中，因此不会自动应用到其他研究运行。

Docker Worker 无法共享宿主进程中的 `Future`，因此返回结构化的 `egress_domain_pending`，并由宿主侧持久化审批决定。源码在这里提供的是"待审批、可重试"协议；是否自动重试以及重试工具调用还是整个任务，取决于上层调度策略，不能表述为容器内原始请求会跨进程透明恢复。

当前源码中的批准者是 Supervisor 编排层，不应直接表述为人工审批。高风险生产场景可进一步接入确定性策略或人工确认，避免 LLM 自批形成弱信任边界。



### 4. SSRF 与目标校验

域名获批只代表"业务上允许访问"，并不代表"最终连接目标一定安全"。受控抓取链路还会：

- 只允许 `http/https`，禁止 URL 中携带用户凭据；
- 拒绝本机、云元数据域名，以及私网、回环、链路本地、组播、保留和未指定地址；
- 检查域名解析得到的全部 IP；
- 关闭自动重定向，对每一跳重新校验，跨域跳转还需重新满足域名策略；
- 在 Transport 能提供 `peername` 时检查实际连接对端，并限制超时、响应类型、响应大小和重定向次数。

真实对端检查是连接后的一致性补充，能够发现部分 DNS 变化，但不能单独消除检查与使用之间的竞态。若要约束任意容器网络流量或进一步防御 DNS Rebinding，生产环境还需要不可绕过的 Egress Proxy、网络策略或防火墙，并由代理端执行目标解析和私网阻断。



## 三、面试时可直接复述的版本⭐

我为异步运行的 Researcher 设计了任务级 Docker 沙箱。**每个任务使用独立的工作区和容器**，输入只读，只开放必要的输出目录；

容器采用只读 RootFS、非 Root 用户、删除全部 Capabilities，并限制 CPU、内存、并发量和执行时间。

网络侧提供四种策略，**受治理的工具调用先经过静态域名白名单或 Supervisor 运行时审批**，HTTP 抓取再校验 URL、DNS、连接对端和重定向目标。

这样既保留了 Agent 的搜索能力，又把单任务风险限制在可取消、可回收、可审计的范围内。



## 四、面试官可能提问及参考答案

### 1. 这项工作解决的核心问题是什么？

异步 Researcher 会并发处理外部内容并调用工具，若全部直接运行在主进程中，一个任务的异常可能拖慢整批研究或扩大到宿主环境。我负责将任务调度、Docker 生命周期、文件与资源隔离、网络策略、域名审批和 SSRF 校验串成一条完整的安全执行链路。



### 2. 为什么选择 Docker，而不是只启动 Python 子进程？

Python 子进程默认仍共享宿主文件、用户权限和网络边界，也缺少统一的 CPU、内存、PID 与 RootFS 约束。Docker 能将这些限制组合成标准运行规格。它仍共享宿主内核，所以这里强调的是降低风险和故障半径，而不是把 Docker 描述成绝对安全边界。



### 3. 文件隔离和最小权限如何实现？

每个任务拥有独立的工作区，输入只读挂载，输出、临时文件、日志和制品目录按需写入，同时校验工作区真实路径不能越过配置根目录。容器再叠加只读 RootFS、非 Root 身份、删除全部 Capabilities 和 `no-new-privileges`。



### 4. 如何防止任务耗尽资源或无法退出？

单容器设置 CPU、内存和 PID 配额，任务系统限制最大并发量，并使用任务级或沙箱级超时。主动取消时停止并删除对应容器，结束后归档结果并按策略清理。当前配额未覆盖磁盘、inode 和网络带宽，生产环境需要补充宿主级配额和孤儿容器巡检。



### 5. 四种网络模式有什么区别？

`no-network` 是 Docker 网络层硬隔离；`allow-search-only` 允许标准研究链路；`allowlist-domain` 对受治理工具的目标域名做静态或动态授权；`open-network` 跳过应用层域名白名单。即使是 `open-network`，框架内置 HTTP 抓取仍会执行 SSRF 校验。

### 6. 未知域名的审批流程是怎样的？

系统先检查静态域名，再查询本次 `run_id` 的动态决定。进程内任务可等待 Supervisor 决策；容器任务返回待审批结果，并把后续重试交给上层调度策略。授权作用域是一次研究运行，因此可被同一 `run_id` 下的多个 Researcher 共享，但不会自动扩散到其他 `run_id`。面试中应将其描述为审批与可重试协议，不要声称已经实现容器内透明续跑。

### 7. Supervisor 审批是否等同于人工安全审批？

不等同。当前它是主 Agent 编排层的一项治理工具，独立性弱于人工或确定性策略。生产环境应根据风险等级设置自动白名单、规则审批和人工确认，并记录批准者、原因和有效期；不确定或超时场景应默认拒绝。

### 8. 已有域名白名单，为什么还需要 SSRF 防护？

白名单回答"允许访问谁"，SSRF 校验回答"这个 URL 最终连到哪里"。合法域名仍可能解析到私网 IP，或通过重定向转向内部服务，因此需要同时检查协议、DNS 结果、重定向目标，并在客户端可获得时复核真实连接对端。

### 9. 这套方案最大的边界和后续改进是什么？

只有 `no-network` 是明确的 Docker 网络层硬隔离；其余模式主要治理已经接入框架的工具和 Web 抓取链路，不能阻止容器内任意程序绕过框架直接创建 Socket。生产环境应强制出口代理或防火墙、禁止直连，并进一步补充 seccomp、AppArmor/SELinux、用户命名空间、磁盘配额、短期任务凭据和孤儿容器回收。

### 10. 如何证明这些能力不是只停留在设计上？

源码中有针对工作区挂载、只读 RootFS、Capabilities、资源参数和网络模式的沙箱测试；有域名允许、拒绝、按运行隔离、等待与恢复的治理测试；也有私网、云元数据地址和真实连接对端的 SSRF 回归测试。生命周期事件、任务状态、输出归档和容器日志用于运行时排查，但更完整的生产验证还应增加真实 Docker、代理绕过和攻击用例集成测试。

## 五、主要源码依据

| 能力 | 主要源码 |
|---|---|
| 异步任务调度与沙箱接入 | `src/open_deep_research/tasks/async_tools.py`、`src/open_deep_research/tasks/teammate_pool.py`、`src/open_deep_research/tasks/executor.py` |
| 工作区、权限、配额、超时和清理 | `src/open_deep_research/sandbox/manager.py`、`src/open_deep_research/sandbox/worker.py` |
| 四种网络模式与静态允许域名 | `src/open_deep_research/configuration.py`、`src/open_deep_research/sandbox/policy.py` |
| 动态域名审批 | `src/open_deep_research/tools/governance.py`、`src/open_deep_research/tasks/domain_approvals.py` |
| SSRF 和 HTTP 抓取防护 | `src/open_deep_research/security/network.py`、`src/open_deep_research/tools/utils.py`、`src/open_deep_research/web/pipeline.py` |
| 回归测试 | `tests/test_sandbox.py`、`tests/test_domain_allowlist.py`、`tests/test_tool_governance.py`、`tests/test_prompt_injection_security.py`、`tests/test_web_pipeline.py` |
