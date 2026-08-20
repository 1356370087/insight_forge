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

export interface TokenVector {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cached_input_tokens: number;
  cache_creation_input_tokens: number;
  reasoning_tokens: number;
}

export type UsageAccountingStatus = "complete" | "partial" | "unavailable";
export type UsageCostSource = "provider_reported" | "configured_estimate" | "unavailable";

export interface UsageBucket {
  key: string;
  label: string;
  reported: TokenVector;
  estimated: TokenVector;
  call_count: number;
  estimated_cost_micro_usd: number | null;
  cost_source: UsageCostSource;
  average_latency_ms?: number;
  completeness?: UsageAccountingStatus;
}

export interface RunUsageResponse {
  schema_version: 1;
  run_id: string;
  status: string;
  duration_ms: number | null;
  revision: number;
  updated_at: number | null;
  accounting_status: UsageAccountingStatus;
  unavailable_reason?: "no_usage_events" | "run_not_observed" | "accounting_disabled" | "storage_unavailable";
  totals: {
    reported: TokenVector;
    estimated: TokenVector;
    calls: {
      attempts: number;
      successful_responses: number;
      provider_reported: number;
      provider_partial: number;
      estimated: number;
      missing: number;
      unknown_failed_attempts: number;
      legacy_unclassified: number;
      coverage_ratio: number;
    };
    cost: {
      estimated_cost_micro_usd: number | null;
      cost_source: UsageCostSource;
      price_table_hash: string | null;
    };
    budgets: Record<string, { settled: number | null; estimated: number; reserved: number; limit: number | null }>;
  };
  breakdowns: {
    by_stage: UsageBucket[];
    by_agent_role: UsageBucket[];
    by_model: UsageBucket[];
    by_task: UsageBucket[];
  };
  timeline: Array<{
    timestamp: number;
    reported_tokens: number;
    estimated_tokens: number;
    reported_cumulative: number;
    estimated_cumulative: number;
    call_count: number;
    retry_count: number;
  }>;
  operations: {
    llm_call_count: number;
    retry_count: number;
    rate_limited_count: number;
    rate_429: number;
    cache_hit_rate: number;
    cache_input_ratio: number;
    reasoning_output_ratio: number;
    output_tokens_per_second: number;
    tool_call_count: number;
    tool_success_rate: number;
    empty_tool_result_count: number;
    zero_source_search_count: number;
  };
}

export interface UsageAnalyticsResponse {
  schema_version: 1;
  range: "7d" | "30d" | "retained";
  timezone: string;
  retention_days: number;
  actual_range_days: number;
  summary: {
    run_count: number;
    reported: TokenVector;
    estimated: TokenVector;
    estimated_cost_micro_usd: number | null;
    coverage_ratio: number;
  };
  daily: Array<{
    date: string;
    reported_tokens: number;
    estimated_tokens: number;
    run_count: number;
    coverage_ratio: number;
    rate_429: number;
    cache_hit_rate: number;
    output_tokens_per_second: number;
  }>;
  distributions: {
    provider: Array<{ key: string; reported_tokens: number; estimated_tokens: number; call_count: number }>;
    model: Array<{ key: string; reported_tokens: number; estimated_tokens: number; call_count: number }>;
    status: Array<{ key: string; reported_tokens: number; estimated_tokens: number; run_count: number }>;
  };
  runs: Array<{
    run_id: string;
    title: string;
    status: string;
    started_at: number;
    ended_at: number | null;
    duration_ms: number | null;
    accounting_status: UsageAccountingStatus;
    reported: TokenVector;
    estimated: TokenVector;
    calls: RunUsageResponse["totals"]["calls"];
    cost: RunUsageResponse["totals"]["cost"];
    operations: RunUsageResponse["operations"];
  }>;
  next_cursor: string | null;
}
