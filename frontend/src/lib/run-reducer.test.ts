import { describe, expect, it } from "vitest";
import { emptyRunState, reducePublicEvent } from "./run-reducer";
import type { PublicEvent } from "./types";

const event = (sequence: number, type: string, payload: Record<string, unknown>): PublicEvent => ({
  schema_version: 2,
  event_id: `e-${sequence}`,
  sequence,
  run_id: "run-1",
  type,
  timestamp: "2026-01-01T00:00:00Z",
  payload,
});

describe("reducePublicEvent", () => {
  it("ignores repeated or older sequences", () => {
    const state = reducePublicEvent(emptyRunState("run-1"), event(2, "run.started", { status: "running" }));
    expect(reducePublicEvent(state, event(2, "run.failed", { status: "failed" }))).toBe(state);
  });

  it("upserts tasks and sources by stable identity", () => {
    let state = reducePublicEvent(emptyRunState("run-1"), event(1, "research.task.started", { task_id: "t1", status: "running" }));
    state = reducePublicEvent(state, event(2, "research.task.progress", { task_id: "t1", iteration: 2 }));
    state = reducePublicEvent(state, event(3, "research.source.discovered", { task_id: "t1", source_id: "s1", url: "https://example.com/a?x=1" }));
    state = reducePublicEvent(state, event(4, "research.source.discovered", { task_id: "t1", source_id: "s2", url: "https://example.com/a?x=2" }));
    expect(Object.keys(state.tasksById)).toEqual(["t1"]);
    expect(state.tasksById.t1.iteration).toBe(2);
    expect(Object.keys(state.sourcesById)).toHaveLength(1);
  });

  it("restores and resolves clarification", () => {
    let state = reducePublicEvent(emptyRunState("run-1"), event(1, "clarification.required", { action_id: "a1", question: "范围？", allowed_actions: ["answer", "cancel"] }));
    expect(state.status).toBe("awaiting_clarification");
    expect(state.pendingHumanAction?.payload.question).toBe("范围？");
    state = reducePublicEvent(state, event(2, "clarification.resolved", { action_id: "a1", action: "answer" }));
    expect(state.pendingHumanAction).toBeUndefined();
  });

  it("closes on a terminal event without failing on unknown v1 events", () => {
    let state = reducePublicEvent(emptyRunState("run-1"), { ...event(1, "future.event", {}), schema_version: 1 });
    expect(state.diagnostics).toHaveLength(1);
    state = reducePublicEvent(state, event(2, "run.completed", { status: "completed" }));
    expect(state.terminal).toBe(true);
    expect(state.connectionState).toBe("closed");
  });
});
