"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { AlertTriangle, Bot, CheckCircle2, ChevronDown, Database, ExternalLink, LoaderCircle, RotateCcw, Search, ShieldAlert, TerminalSquare, Wrench, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTaskActivity } from "@/hooks/use-task-activity";
import type { ResearchTask, TaskActivityEvent, TaskActivityKind, TaskActivityPhase } from "@/lib/types";

const filters: Array<{ value?: TaskActivityKind; label: string }> = [
  { label: "全部" }, { value: "model", label: "模型" }, { value: "tool", label: "工具" },
  { value: "source", label: "来源" }, { value: "quality", label: "质量" },
  { value: "security", label: "安全" }, { value: "error", label: "错误" },
];
const phaseNames: Record<TaskActivityPhase, string> = {
  queued: "排队", initializing: "准备", reasoning: "模型规划", tool_execution: "工具执行",
  evidence_review: "证据评估", quality_check: "质量复核", gap_recovery: "补证恢复",
  compressing: "压缩", handoff: "交接", terminal: "终态",
};
const phaseRail: TaskActivityPhase[] = ["initializing", "reasoning", "tool_execution", "evidence_review", "quality_check", "gap_recovery", "compressing", "handoff"];

function ActivityIcon({ event }: { event: TaskActivityEvent }) {
  const props = { size: 15, "aria-hidden": true };
  if (event.kind === "model") return <Bot {...props} />;
  if (event.kind === "tool") return <Wrench {...props} />;
  if (event.kind === "source") return <Search {...props} />;
  if (event.kind === "quality") return <CheckCircle2 {...props} />;
  if (event.kind === "security" || event.kind === "error") return <ShieldAlert {...props} />;
  if (event.kind === "checkpoint") return <Database {...props} />;
  return <TerminalSquare {...props} />;
}

function safeUrl(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  try { const url = new URL(value); return ["http:", "https:"].includes(url.protocol) ? url.toString() : undefined; } catch { return undefined; }
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return <span><b>{value}</b>{label}</span>;
}

function TimelineEvent({ event }: { event: TaskActivityEvent }) {
  const url = safeUrl(event.payload.url);
  return <article className="activity-event" data-kind={event.kind} data-status={event.status}>
    <div className="activity-marker"><ActivityIcon event={event} /></div>
    <div className="activity-event-body">
      <header><div><span className="activity-kind">{event.kind}</span><h4>{event.title}</h4></div><time>{new Date(event.timestamp).toLocaleTimeString("zh-CN", { hour12: false })}</time></header>
      <p>{event.summary}</p>
      <div className="activity-metrics">
        {event.iteration !== undefined && <span>ITER {event.iteration}</span>}
        {event.duration_ms !== undefined && <span>{(event.duration_ms / 1000).toFixed(2)}s</span>}
        {typeof event.payload.model === "string" && <span>{event.payload.model}</span>}
        {typeof event.payload.tool_name === "string" && <span>{event.payload.tool_name}</span>}
        {typeof event.payload.source_count === "number" && <span>SRC {event.payload.source_count}</span>}
        {typeof event.payload.result_chars === "number" && <span>{event.payload.result_chars} CHARS</span>}
        {typeof event.payload.error_code === "string" && <span>ERR {event.payload.error_code}</span>}
      </div>
      {url && <a href={url} target="_blank" rel="noopener noreferrer">{String(event.payload.title ?? event.payload.domain ?? url)} <ExternalLink size={11} /></a>}
      {event.payload.preview !== undefined && <details><summary>安全诊断预览 <ChevronDown size={12} /></summary><pre>{JSON.stringify(event.payload.preview, null, 2)}</pre></details>}
    </div>
  </article>;
}

function SourceEventGroup({ events }: { events: TaskActivityEvent[] }) {
  return <details className="activity-source-group">
    <summary>
      <span className="activity-marker"><Search size={15} aria-hidden /></span>
      <span><b>发现 {events.length} 个来源</b><small>连续证据采集事件已折叠</small></span>
      <ChevronDown size={14} aria-hidden />
    </summary>
    <div className="activity-source-list">
      {events.map((event) => {
        const url = safeUrl(event.payload.url);
        const label = String(event.payload.title ?? event.payload.domain ?? url ?? event.title);
        return <div key={event.event_id}>
          <time>{new Date(event.timestamp).toLocaleTimeString("zh-CN", { hour12: false })}</time>
          {url ? <a href={url} target="_blank" rel="noopener noreferrer">{label}<ExternalLink size={11} /></a> : <span>{label}</span>}
        </div>;
      })}
    </div>
  </details>;
}

export function TaskActivityDrawer({ runId, task, onClose }: { runId: string; task?: ResearchTask; onClose: () => void }) {
  const [kind, setKind] = useState<TaskActivityKind | undefined>();
  const { events, connection, loading, error, source, detailLevel, hasMore, loadOlder } = useTaskActivity(runId, task?.task_id ?? "");
  const viewportRef = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);
  const [unread, setUnread] = useState(0);
  const current = events.at(-1);
  const lastCount = useRef(0);

  useEffect(() => {
    const added = Math.max(0, events.length - lastCount.current);
    lastCount.current = events.length;
    if (!added) return;
    if (following) requestAnimationFrame(() => viewportRef.current?.scrollTo({ top: viewportRef.current.scrollHeight, behavior: "smooth" }));
    else setUnread((count) => count + added);
  }, [events.length, following]);

  const phasesSeen = useMemo(() => new Set(events.map((event) => event.phase)), [events]);
  const visibleEvents = useMemo(() => kind ? events.filter((event) => event.kind === kind) : events, [events, kind]);
  const timelineItems = useMemo(() => visibleEvents.reduce<Array<{ events: TaskActivityEvent[]; sourceGroup: boolean }>>((items, event) => {
    const previous = items.at(-1);
    if (event.kind === "source" && previous?.sourceGroup && previous.events.at(-1)?.iteration === event.iteration) {
      previous.events.push(event);
    } else {
      items.push({ events: [event], sourceGroup: event.kind === "source" });
    }
    return items;
  }, []), [visibleEvents]);
  if (!task) return null;
  return <Dialog.Root open onOpenChange={(open) => { if (!open) onClose(); }}>
    <Dialog.Portal>
      <Dialog.Overlay className="activity-overlay" />
      <Dialog.Content className="activity-drawer" aria-describedby="task-activity-description" onCloseAutoFocus={(event) => {
        event.preventDefault();
        requestAnimationFrame(() => {
          const taskCard = document.querySelector<HTMLButtonElement>(`[data-task-id="${CSS.escape(task.task_id)}"]`);
          (taskCard ?? document.querySelector<HTMLButtonElement>("[data-task-focus-fallback]"))?.focus();
        });
      }}>
        <header className="activity-header">
          <div><span className="eyebrow mono">SUBAGENT / {task.task_id}</span><Dialog.Title>{task.title ?? task.task_id}</Dialog.Title><Dialog.Description id="task-activity-description">真实执行事件与安全业务摘要</Dialog.Description></div>
          <Dialog.Close className="activity-close" aria-label="关闭任务详情"><X size={18} /></Dialog.Close>
        </header>
        <section className="activity-summary">
          <div className="activity-summary-row"><span className="status-chip" data-status={task.status}>{task.status ?? "pending"}</span><span className={`activity-connection ${connection}`}><i />{connection}</span><span className="activity-origin">{source === "native" ? "NATIVE" : source === "derived_trace" ? "历史 TRACE 推导" : "SUMMARY ONLY"}</span></div>
          <div className="activity-stat-grid">
            <Metric label="模型调用" value={task.model_call_count ?? events.filter((event) => event.type === "model.completed").length} />
            <Metric label="工具调用" value={task.tool_call_count ?? events.filter((event) => event.type === "tool.completed").length} />
            <Metric label="来源" value={task.source_count ?? 0} />
            <Metric label="警告 / 重试" value={`${task.warning_count ?? 0} / ${task.retry_count ?? 0}`} />
          </div>
        </section>
        <section className="activity-live" data-status={current?.status ?? "pending"}>
          <div><span>CURRENT ACTIVITY</span><b>{current ? phaseNames[current.phase] : task.activity_label ?? "等待活动事件"}</b></div>
          <p>{connection === "reconnecting" ? "连接中断，正在保留最后已知活动并重连。" : current?.summary ?? "任务存在，但暂无可公开的细粒度事件。"}</p>
          {connection === "reconnecting" && <RotateCcw className="activity-spin" size={18} />}
        </section>
        <section className="activity-phase-rail" aria-label="研究循环阶段">
          {phaseRail.map((phase) => <span key={phase} data-active={current?.phase === phase} data-seen={phasesSeen.has(phase)}>{phaseNames[phase]}</span>)}
        </section>
        <nav className="activity-filters" aria-label="事件筛选">
          {filters.map((filter) => <button key={filter.label} className={filter.value === kind ? "active" : ""} onClick={() => setKind(filter.value)}>{filter.label}</button>)}
          <span>{detailLevel.toUpperCase()}</span>
        </nav>
        <div className="activity-timeline" ref={viewportRef} onScroll={(event) => {
          const element = event.currentTarget;
          const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 80;
          setFollowing(atBottom);
          if (atBottom) setUnread(0);
        }}>
          {hasMore && <button className="activity-load-more" onClick={() => void loadOlder()}>加载更早事件</button>}
          {loading && <div className="activity-empty"><LoaderCircle className="activity-spin" />正在读取任务事件……</div>}
          {error && <div className="activity-empty error"><AlertTriangle />任务活动暂时不可用</div>}
          {!loading && !error && visibleEvents.length === 0 && <div className="activity-empty"><Database />{events.length ? "当前筛选没有匹配事件。" : "此任务仅有摘要状态，未发现可安全关联的历史事件。"}</div>}
          {timelineItems.map((item, index) => {
            const event = item.events[0];
            const previous = timelineItems[index - 1]?.events.at(-1);
            return <div className="activity-iteration" key={event.event_id}>
              {(index === 0 || previous?.iteration !== event.iteration) && event.iteration !== undefined && <div className="activity-iteration-label">ITERATION {event.iteration}</div>}
              {item.sourceGroup ? <SourceEventGroup events={item.events} /> : <TimelineEvent event={event} />}
            </div>;
          })}
        </div>
        {!following && <button className="activity-unread" onClick={() => { setFollowing(true); setUnread(0); viewportRef.current?.scrollTo({ top: viewportRef.current.scrollHeight, behavior: "smooth" }); }}>{unread ? `${unread} 条新事件` : "回到最新"}</button>}
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>;
}
