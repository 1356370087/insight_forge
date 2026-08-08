export type RunStatus =
  | "pending" | "running" | "awaiting_clarification" | "awaiting_plan_approval"
  | "awaiting_outline_approval" | "cancelling" | "completed" | "failed" | "cancelled";

export type ConnectionState = "idle" | "connecting" | "connected" | "reconnecting" | "closed" | "error";
export type StageId = "preparing" | "planning" | "researching" | "synthesizing" | "writing" | "finalizing";

export interface PublicEvent {
  schema_version: 1 | 2 | number;
  event_id: string;
  sequence: number;
  run_id: string;
  type: string;
  timestamp: string;
  stage?: StageId;
  payload: Record<string, unknown>;
}

export interface ResearchTask {
  task_id: string;
  wave_id?: string;
  title?: string;
  status?: string;
  phase?: string;
  iteration?: number;
  source_count?: number;
  elapsed_ms?: number;
  mode?: string;
  activity_phase?: TaskActivityPhase;
  activity_label?: string;
  last_activity_at?: string;
  activity_event_count?: number;
  model_call_count?: number;
  tool_call_count?: number;
  retry_count?: number;
  warning_count?: number;
  activity_available?: boolean;
}

export type TaskActivityKind = "lifecycle" | "model" | "tool" | "source" | "quality" | "checkpoint" | "control" | "security" | "error";
export type TaskActivityPhase = "queued" | "initializing" | "reasoning" | "tool_execution" | "evidence_review" | "quality_check" | "gap_recovery" | "compressing" | "handoff" | "terminal";
export type TaskActivityStatus = "pending" | "running" | "success" | "warning" | "error" | "cancelled";

export interface TaskActivityEvent {
  schema_version: number;
  event_id: string;
  sequence: number;
  run_id: string;
  task_id: string;
  timestamp: string;
  type: string;
  kind: TaskActivityKind;
  phase: TaskActivityPhase;
  status: TaskActivityStatus;
  title: string;
  summary: string;
  iteration?: number;
  duration_ms?: number;
  payload: Record<string, unknown>;
}

export interface TaskActivityPage {
  items: TaskActivityEvent[];
  oldest_sequence: number;
  last_event_id: number;
  has_more: boolean;
  detail_level: "summary" | "preview";
  source: "native" | "derived_trace" | "summary_only";
  stream_url: string;
}

export interface ResearchSource {
  source_id: string;
  task_id?: string;
  title?: string;
  domain?: string;
  url: string;
}

export interface PendingHumanAction {
  action_id: string;
  type: "clarification" | "plan_approval" | "outline_approval";
  payload: { question?: string; content_markdown?: string; research_plan?: string; report_outline?: string };
  allowed_actions?: Array<"approve" | "revise" | "answer" | "cancel">;
}

export interface Artifact { name?: string; type?: string; url?: string; path?: string; content?: unknown }

export interface ResearchRunState {
  runId: string;
  title: string;
  status: RunStatus;
  connectionState: ConnectionState;
  currentStage?: StageId;
  stageProgress: Record<string, "pending" | "running" | "completed" | "failed">;
  plan: Record<string, unknown>;
  wavesById: Record<string, { wave_id: string; status: string; task_ids: string[]; mode?: string }>;
  tasksById: Record<string, ResearchTask>;
  sourcesById: Record<string, ResearchSource>;
  findingsByTaskId: Record<string, { task_id: string; summary?: string; sources?: unknown[]; updatedAt: string }>;
  pendingHumanAction?: PendingHumanAction;
  report: string;
  artifacts: Artifact[];
  qualityGate?: Record<string, unknown>;
  warnings: Array<{ code: string; message: string }>;
  diagnostics: string[];
  lastEventId: number;
  isHydrated: boolean;
  isReconnecting: boolean;
  terminal: boolean;
}

export interface RunSnapshot {
  run_id: string;
  title?: string;
  status: RunStatus;
  pending_human_action?: PendingHumanAction;
  progress?: {
    status?: RunStatus;
    current_stage?: StageId;
    task_items?: Record<string, ResearchTask>;
    sources?: ResearchSource[];
    latest_findings?: Array<Record<string, unknown>>;
    pending_human_action?: PendingHumanAction;
    plan?: Record<string, unknown>;
    last_event_id?: number;
  };
  output?: { markdown?: string; artifacts?: Artifact[]; quality_gate?: Record<string, unknown> };
  last_event_id: number;
}
