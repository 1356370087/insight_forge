---
name: git-batch-commit
description: 把当前 git 工作区里尚未提交的改动，按"职责"拆成多个有条理的提交，并统一推送到远程仓库（默认 Gitee）。每当用户想把一堆散乱的改动整理成多个有意义的提交、或提到"分批提交""按职责/模块提交""提交并推送到 gitee""帮我提交代码""把工作区改动推上去""整理一下改动"时，都应使用本 skill。即便用户只是笼统地说"提交一下""推一下"，只要工作区存在跨多个模块的改动，也要主动用它来拆分提交，避免一个臃肿难读的大 commit。
---

# 分批提交工作区改动到远程仓库

本 skill 把工作区里所有待提交的改动，按**功能职责**归组成若干批次，为每批生成一条中文约定式提交信息（`type(scope): 描述`），逐批提交后**统一一次性推送**到远程仓库（默认推到 `gitee` 远程）。

## 为什么按职责拆分

一个塞满各种改动的巨型 commit 很难评审、很难回滚、也很难讲清楚做了什么。按"这批改动在履行什么职责"来拆——比如"接入文件信箱协作"是一批、"重构研究代理的调用路径"是另一批——每个提交都是一个能独立讲清楚、能独立评审的单元。这正是本 skill 的核心价值，也是判断"该怎么分批"的依据。

## 安全原则（始终遵守）

提交、尤其是推送，是难以撤销的外发动作：

1. **先出方案、获明确确认、再执行。** 在用户说"可以/确认/提交吧"之前，不要 `git commit`，更不要 `git push`。
2. **逐文件精确暂存，绝不一把梭。** 只 `git add` 当前批次里的那些文件。永远不要 `git add .` 或 `git add -A`——它们会把临时文件和不同职责的改动混进同一个提交。
3. **永不 `--force` 推送。** 推送被拒绝（非快进）时停下、报告、问用户，绝不强推。
4. **临时产物不进提交。** `.tmp/`、`tmp/`、`dist/`、`build/`、`__pycache__/`、`*.log`、`.cache/`、`node_modules/` 等，一律跳过；必要时建议用户把它们加进 `.gitignore`。

## 工作流程

### 第 1 步：摸清现状（只读，安全）

并行收集：

- `git status --porcelain` —— 列出所有改动（`M`=已修改、`A`=已暂存新增、`??`=未跟踪）
- `git remote -v` —— 找远程，确定待会推到哪
- `git branch --show-current` —— 当前分支
- 对每个**已修改**的文件看 `git diff -- <file>`，理解改动意图。这一步决定 type 判定，别偷懒只看文件名。

未跟踪文件（`??`）若属于本次工作（如新写的测试、新模块），纳入相应批次；若是临时目录/产物，按安全原则第 4 条跳过。

### 第 2 步：按职责分组 + 判定 type/scope

对每份改动判断两件事：

**A. 归到哪个功能职责（决定 `scope`）**——按改动的功能领域归组。**scope 用英文**（通常就是模块/目录名，与历史提交保持一致），**描述用中文**。常见映射（按实际仓库调整）：

| 改动位置 | scope |
|---|---|
| `agents/` | agents |
| `tasks/` | tasks |
| `tools/` | tools |
| `observability/` | observability |
| `report/` | report |
| `skills/` | skills |
| `configuration.py` / `config*` | config |
| `server.py` | server |
| `quality.py` | quality |
| 横跨多个核心模块 | core |

纯测试改动用 `test:` 不带 scope（"测试"本身就是职责）。

**单个文件横跨多个职责时**（如某配置文件同时含可观测性字段和任务调度校验）：默认把整个文件归到**主导职责**那批，并在方案里说明这个取舍——这样每个中间提交都自洽（不会引用尚未落地的字段/符号），也避开了需要交互的 hunk 拆分。仅当用户明确要求最细粒度时，才建议其本人在终端用 `git add -p` 手动拆分（本环境无法交互式选 hunk）。

**B. 这批改动的性质（决定 `type`）**：

| type | 何时用 |
|---|---|
| `feat` | 新增了之前没有的能力/功能 |
| `fix` | 修正了错误的行为（bug 修复） |
| `refactor` | 搬家/改名/抽取函数等，**对外行为不变** |
| `perf` | 性能优化 |
| `test` | 新增或修改测试 |
| `docs` | 文档 |
| `chore` | 构建、依赖、配置等杂项 |
| `ci` | CI 配置 |

判定要点：看这批改动的**主导意图**。同一模块里既有新功能又有重构时，看哪个是主导；若两者都很重且界限清晰，就**拆成两个批次**——它们是不同职责。纯测试默认走 `test`；但如果某些测试是随某个新功能一起加的、和功能代码同属一个"职责单元"，可并入那个 `feat` 批次（判断标准：这些改动作为一个整体，能否被独立评审、独立讲清楚）。

合成提交信息：`type(scope): 中文描述`（scope 英文、描述中文），如 `feat(tasks): 新增基于文件信箱的子任务协作机制`。描述写"做了什么"，简洁、动词开头。

### 第 3 步：出方案，等确认

把分批方案清晰列给用户，例如：

```
拟分 5 个批次提交，最后统一推送到 gitee/main：

批次 1  feat(tasks): 新增基于文件信箱的子任务协作机制
        src/open_deep_research/tasks/mailbox.py
        src/open_deep_research/tasks/coordination.py
        …（共 8 个文件）

批次 2  refactor(agents): 研究代理改走新的工具/技能/报告接口
        src/open_deep_research/agents/deep_researcher.py
        src/open_deep_research/agents/query_engine.py

批次 3  feat(observability): 补全遥测与核心可观测接口
        src/open_deep_research/observability/__init__.py
        src/open_deep_research/observability/core.py
        src/open_deep_research/observability/telemetry.py

批次 4  refactor(tools): 调整工具契约与适配
        src/open_deep_research/tools/governance.py
        src/open_deep_research/tools/utils.py

批次 5  test: 同步更新测试与评估器
        tests/evaluators.py …（含新增 test_evaluators_compat.py）

注：.tmp/ 为临时目录，已跳过（建议加入 .gitignore）。
确认后我将逐批提交，全部完成后一次性 git push gitee main。
```

明确请用户确认。若用户只想看方案、暂不执行（或你在测试本 skill），到这一步停下即可。

### 第 4 步：逐批提交

确认后，对每个批次：

1. 暂存**仅该批次**的文件：`git add <file1> <file2> …`
2. 提交。**中文信息用文件传入，避免引号/编码问题**：先用 Write 工具把提交信息写进临时文件（如 `.git/BATCH_MSG`），再执行 `git commit -F .git/BATCH_MSG`，提交后删掉该文件。
3. 每提交完一批，简短报告一句（如"批次 1 已提交"）。

> 为什么用 `-F`：在 Windows/PowerShell 下直接 `git commit -m "中文…"` 容易踩引号转义和编码的坑；写进文件再 `-F` 读取最稳。

### 第 5 步：统一推送

所有批次提交完后，再确认一次，然后推送：

- **确定远程**：存在名为 `gitee` 的远程 → `git push gitee <分支>`；否则找 URL 含 `gitee.com` 的远程；都没有就问用户要推到哪，或先 `git remote add gitee <url>`。
- 推送命令：`git push <remote> <branch>`。
- **（可选）推送前自检**：若改动可能影响测试/构建，可提醒用户先跑 `ruff check`、`pytest -q` 之类再推；但这只是提醒，别让它阻塞——提交流程本身不依赖测试通过。
- **认证失败别死磕**：HTTPS 推 Gitee 需要账号/令牌，本环境无法交互输入。若 push 报权限/认证错误，直接告诉用户，建议其自行执行 `! git push gitee <分支>`，或配置凭据助手 / SSH 密钥。
- **被拒绝（非快进）时停下**：远端有本地没有的提交。不要强推，报告情况让用户决定（通常是先 `git pull --rebase`）。

## 边界情况

- **没有改动**：直接告诉用户工作区是干净的，无需提交。
- **改动只属于单一职责**：那就一个批次，不必硬拆。拆分依据是职责边界，不是"凑够 N 个提交"。
- **部分文件已暂存（staged）**：先 `git status` 看清；按职责重新组织暂存，必要时先 `git reset`（默认混合重置，只取消暂存、不动文件内容）再按批次 `git add`。
- **新增未跟踪文件**：属于本次工作的纳入相应批次；临时产物按安全原则第 4 条跳过。
- **用户只想提交、不想推送**：尊重用户，做完第 4 步即停下，不执行第 5 步。
