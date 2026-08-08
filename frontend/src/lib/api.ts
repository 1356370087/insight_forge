import { fetchEventSource } from "@microsoft/fetch-event-source";
import { csrfHeaders, refreshBrowserSession } from "./auth";
import type { PublicEvent, RunSnapshot, TaskActivityEvent, TaskActivityKind, TaskActivityPage } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_RESEARCH_API_BASE ?? "/api/research";

async function headers(extra?: HeadersInit): Promise<Headers> {
  return csrfHeaders(extra);
}

export async function apiFetch<T>(path: string, init: RequestInit = {}, refresh = false): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "same-origin",
    headers: await headers(init.headers),
  });
  if (response.status === 401 && !refresh && await refreshBrowserSession()) return apiFetch<T>(path, init, true);
  if (response.status === 401 && typeof window !== "undefined") {
    window.location.replace("/login");
  }
  if (!response.ok) throw new Error(`${response.status}:${await response.text()}`);
  return response.json() as Promise<T>;
}

export const researchApi = {
  capabilities: () => apiFetch<Record<string, unknown>>("/capabilities"),
  listRuns: (status?: string) => apiFetch<{ items: Array<Record<string, unknown>>; next_cursor?: string }>(`/runs?limit=50${status ? `&status=${encodeURIComponent(status)}` : ""}`),
  getRun: (id: string) => apiFetch<RunSnapshot>(`/runs/${encodeURIComponent(id)}`),
  createRun: (query: string, configurable: Record<string, unknown>, title?: string) => apiFetch<{ run_id: string }>("/runs", {
    method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": crypto.randomUUID() },
    body: JSON.stringify({ title, messages: [{ role: "user", content: query }], configurable }),
  }),
  humanAction: (runId: string, actionId: string, action: string, message = "") => apiFetch(`/runs/${runId}/human-actions/${actionId}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action, message }) }),
  feedback: (runId: string, payload: Record<string, unknown>) => apiFetch(`/runs/${runId}/feedback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
  cancel: (runId: string) => apiFetch(`/runs/${runId}/cancel`, { method: "POST" }),
  resume: (runId: string) => apiFetch(`/runs/${runId}/resume`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }),
  taskActivity: (runId: string, taskId: string, options: { before?: number; limit?: number; kind?: TaskActivityKind } = {}) => {
    const query = new URLSearchParams({ limit: String(options.limit ?? 100) });
    if (options.before !== undefined) query.set("before", String(options.before));
    if (options.kind) query.set("kind", options.kind);
    return apiFetch<TaskActivityPage>(`/runs/${encodeURIComponent(runId)}/tasks/${encodeURIComponent(taskId)}/activity?${query}`);
  },
};

export async function subscribeToRun(options: {
  runId: string; after: number; signal: AbortSignal;
  onOpen: () => void; onEvent: (event: PublicEvent) => void;
  onReconnect: () => void; onCursorAhead: () => Promise<void>;
}): Promise<void> {
  let authRetried = false;
  let retryAttempt = 0;
  await fetchEventSource(`${API_BASE}/runs/${options.runId}/events?after=${options.after}`, {
    method: "GET", signal: options.signal, openWhenHidden: true,
    headers: Object.fromEntries((await headers({ Accept: "text/event-stream", "Last-Event-ID": String(options.after) })).entries()),
    async onopen(response) {
      if (response.ok) { retryAttempt = 0; options.onOpen(); return; }
      if (response.status === 401 && !authRetried) { authRetried = true; await refreshBrowserSession(); throw new Error("sse-auth-refreshed"); }
      if (response.status === 409) { await options.onCursorAhead(); throw new Error("cursor-ahead"); }
      throw new Error(`sse-${response.status}`);
    },
    onmessage(message) { if (message.data) options.onEvent(JSON.parse(message.data) as PublicEvent); },
    onerror(error) {
      const message = error instanceof Error ? error.message : String(error);
      if (message.includes("cursor-ahead") || message.includes("sse-auth") || message.includes("sse-401")) throw error;
      options.onReconnect();
      const delay = Math.min(30_000, 1_000 * 2 ** retryAttempt);
      retryAttempt += 1;
      return delay;
    },
  });
}

export async function subscribeToTaskActivity(options: {
  runId: string; taskId: string; after: number; signal: AbortSignal;
  onOpen: () => void; onEvent: (event: TaskActivityEvent) => void;
  onReconnect: () => void; onCursorAhead: () => Promise<void>;
}): Promise<void> {
  let authRetried = false;
  let retryAttempt = 0;
  const path = `/runs/${encodeURIComponent(options.runId)}/tasks/${encodeURIComponent(options.taskId)}/activity/stream?after=${options.after}`;
  await fetchEventSource(`${API_BASE}${path}`, {
    method: "GET", signal: options.signal, openWhenHidden: true,
    headers: Object.fromEntries((await headers({ Accept: "text/event-stream", "Last-Event-ID": String(options.after) })).entries()),
    async onopen(response) {
      if (response.ok) { retryAttempt = 0; options.onOpen(); return; }
      if (response.status === 401 && !authRetried) { authRetried = true; await refreshBrowserSession(); throw new Error("task-sse-auth-refreshed"); }
      if (response.status === 409) { await options.onCursorAhead(); throw new Error("task-cursor-ahead"); }
      throw new Error(`task-sse-${response.status}`);
    },
    onmessage(message) { if (message.data) options.onEvent(JSON.parse(message.data) as TaskActivityEvent); },
    onerror(error) {
      const message = error instanceof Error ? error.message : String(error);
      if (message.includes("cursor-ahead") || message.includes("sse-auth")) throw error;
      options.onReconnect();
      const delay = Math.min(30_000, 1_000 * 2 ** retryAttempt);
      retryAttempt += 1;
      return delay;
    },
  });
}
