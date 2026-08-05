import { AppShell } from "@/components/app-shell";
import { ResearchComposer } from "@/components/research-composer";

function NewInspector() { return <><h2 className="inspector-title">任务启动协议</h2><div className="inspector-block"><div className="summary-grid"><div><b>06</b><span>真实事件阶段</span></div><div><b>SSE</b><span>可恢复事件流</span></div></div></div><div className="inspector-block"><p className="eyebrow">执行说明</p><p className="empty-note">创建后立即进入工作区。任务、来源、发现与审批都由公开事件驱动，不显示伪进度。</p></div><div className="inspector-block"><p className="eyebrow">安全边界</p><p className="empty-note">浏览器只发送服务端明确公开的用户配置。API Key、MCP 和管理员策略始终留在服务端。</p></div></>; }

export default function NewResearchPage() { return <AppShell inspector={<NewInspector />}><div className="page hero-console"><span className="hero-number">01 / RESEARCH INTAKE</span><h1 className="hero-title">把一个问题，<br />推进到<em>可验证结论</em>。</h1><p className="hero-copy">输入目标、约束和你真正关心的判断标准。研究工作台会拆分任务、并行收集证据，并把每一步实时呈现在事件轨道上。</p><ResearchComposer /><div className="config-strip"><span>PLAN · EVENT DRIVEN</span><span>SOURCES · DEDUPED</span><span>REPORT · CITED</span><span>HITL · RECOVERABLE</span></div></div></AppShell>; }
