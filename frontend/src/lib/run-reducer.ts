import type { PublicEvent, ResearchRunState, ResearchSource, ResearchTask, RunSnapshot, RunStatus, StageId } from "./types";

export const STAGES: StageId[] = ["preparing", "planning", "researching", "synthesizing", "writing", "finalizing"];
const TERMINAL = new Set(["run.completed", "run.failed", "run.cancelled"]);

export function deriveWaveStatus(tasks: ResearchTask[], explicitStatus?: string): string {
  if (explicitStatus) return explicitStatus;
  const statuses = tasks.map((task) => String(task.status ?? task.phase ?? "pending").toLowerCase());
  if (statuses.some((status) => status === "failed")) return "failed";
  if (statuses.length > 0 && statuses.every((status) => status === "completed")) return "completed";
  if (statuses.some((status) => ["running", "researching", "compressing"].includes(status))) return "running";
  return "queued";
}

export function emptyRunState(runId = ""): ResearchRunState {
  return {
    runId, title: runId, status: "pending", connectionState: "idle", stageProgress: {},
    plan: {}, wavesById: {}, tasksById: {}, sourcesById: {}, findingsByTaskId: {},
    report: "", artifacts: [], warnings: [], diagnostics: [], lastEventId: 0,
    isHydrated: false, isReconnecting: false, terminal: false,
  };
}

function sourceKey(source: Partial<ResearchSource>): string {
  try {
    const url = new URL(source.url ?? "");
    url.hash = "";
    url.search = "";
    return url.toString().replace(/\/$/, "").toLowerCase() || source.source_id || "";
  } catch { return source.source_id || source.url || ""; }
}

export function hydrateSnapshot(state: ResearchRunState, snapshot: RunSnapshot): ResearchRunState {
  const progress = snapshot.progress ?? {};
  const terminal = ["completed", "failed", "cancelled"].includes(snapshot.status);
  const stageProgress = snapshot.status === "completed"
    ? Object.fromEntries(STAGES.map((stage) => [stage, "completed" as const]))
    : progress.current_stage && snapshot.status === "failed"
      ? { ...state.stageProgress, [progress.current_stage]: "failed" as const }
      : state.stageProgress;
  const findings = Object.fromEntries((progress.latest_findings ?? []).map((item) => [
    String(item.task_id ?? "unknown"),
    { ...item, task_id: String(item.task_id ?? "unknown"), updatedAt: new Date().toISOString() },
  ]));
  const sources = Object.fromEntries((progress.sources ?? []).map((item) => [sourceKey(item), item]));
  return {
    ...state,
    runId: snapshot.run_id,
    title: snapshot.title ?? snapshot.run_id,
    status: snapshot.status,
    connectionState: terminal ? "closed" : state.connectionState,
    currentStage: progress.current_stage,
    stageProgress,
    plan: progress.plan ?? {},
    tasksById: progress.task_items ?? {},
    sourcesById: sources,
    findingsByTaskId: findings,
    pendingHumanAction: snapshot.pending_human_action ?? progress.pending_human_action,
    report: snapshot.output?.markdown ?? "",
    artifacts: snapshot.output?.artifacts ?? [],
    qualityGate: snapshot.output?.quality_gate,
    lastEventId: snapshot.last_event_id ?? progress.last_event_id ?? 0,
    isHydrated: true,
    terminal,
  };
}

export function reducePublicEvent(state: ResearchRunState, event: PublicEvent): ResearchRunState {
  if (event.sequence <= state.lastEventId) return state;
  const payload = event.payload;
  const next: ResearchRunState = { ...state, lastEventId: event.sequence, connectionState: "connected" };
  const status = String(payload.status ?? "") as RunStatus;
  if (event.type.startsWith("run.") && status) next.status = status;
  if (event.type.startsWith("stage.") && payload.stage_id) {
    const stage = String(payload.stage_id) as StageId;
    next.currentStage = stage;
    next.stageProgress = { ...state.stageProgress, [stage]: event.type.split(".")[1] as "running" | "completed" | "failed" };
  } else if (event.type === "plan.created" || event.type === "plan.revised") {
    next.plan = { ...state.plan, ...payload };
  } else if (event.type === "plan.task.added" || event.type.startsWith("research.task.")) {
    const id = String(payload.task_id ?? "");
    if (id) {
      const previous = state.tasksById[id];
      const task = { ...previous, ...payload, task_id: id } as ResearchTask;
      if (typeof previous?.iteration === "number" && typeof task.iteration === "number") task.iteration = Math.max(previous.iteration, task.iteration);
      if (typeof previous?.source_count === "number" && typeof task.source_count === "number") task.source_count = Math.max(previous.source_count, task.source_count);
      next.tasksById = { ...state.tasksById, [id]: task };
    }
  } else if (event.type.startsWith("research.wave.")) {
    const id = String(payload.wave_id ?? "");
    if (id) next.wavesById = { ...state.wavesById, [id]: { ...state.wavesById[id], ...payload, wave_id: id, status: event.type.endsWith("completed") ? "completed" : "running", task_ids: (payload.task_ids as string[]) ?? state.wavesById[id]?.task_ids ?? [] } };
  } else if (event.type === "research.source.discovered") {
    const source = payload as unknown as ResearchSource;
    const key = sourceKey(source);
    if (key) next.sourcesById = { ...state.sourcesById, [key]: { ...state.sourcesById[key], ...source } };
  } else if (event.type === "findings.updated") {
    const id = String(payload.task_id ?? "");
    if (id) next.findingsByTaskId = { ...state.findingsByTaskId, [id]: { ...payload, task_id: id, updatedAt: event.timestamp } };
  } else if (event.type === "approval.required") {
    const kind = `${String(payload.approval_type)}_approval` as "plan_approval" | "outline_approval";
    next.status = `awaiting_${String(payload.approval_type)}_approval` as RunStatus;
    next.pendingHumanAction = { action_id: String(payload.action_id), type: kind, payload: { content_markdown: String(payload.content_markdown ?? "") }, allowed_actions: payload.allowed_actions as never };
  } else if (event.type === "clarification.required") {
    next.status = "awaiting_clarification";
    next.pendingHumanAction = { action_id: String(payload.action_id), type: "clarification", payload: { question: String(payload.question ?? "") }, allowed_actions: payload.allowed_actions as never };
  } else if (event.type === "approval.resolved" || event.type === "clarification.resolved") {
    next.pendingHumanAction = undefined;
    next.status = "running";
  } else if (event.type === "system.warning") {
    next.warnings = [...state.warnings, { code: String(payload.warning_code ?? "warning"), message: String(payload.message ?? "") }];
  } else if (event.type === "model.circuit_state") {
    if (payload.to_state === "open") {
      next.warnings = [...state.warnings, {
        code: "model_circuit_open",
        message: `Model ${String(payload.provider ?? "unknown")}:${String(payload.model ?? "unknown")} is temporarily isolated.`,
      }];
    }
  } else if (!TERMINAL.has(event.type) && !event.type.startsWith("report.") && !event.type.startsWith("feedback.")) {
    next.diagnostics = [...state.diagnostics, `Unknown event ${event.type} (v${event.schema_version})`].slice(-50);
  }
  if (TERMINAL.has(event.type)) {
    next.terminal = true;
    next.pendingHumanAction = undefined;
    next.connectionState = "closed";
  }
  return next;
}
