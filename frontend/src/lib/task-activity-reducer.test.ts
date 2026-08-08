import { describe, expect, it } from "vitest";
import { emptyTaskActivityState, mergeTaskActivityHistory, reduceTaskActivity } from "./task-activity-reducer";
import type { TaskActivityEvent } from "./types";

function event(sequence: number, eventId = `event-${sequence}`): TaskActivityEvent {
  return { schema_version: 1, event_id: eventId, sequence, run_id: "run-1", task_id: "task-1", timestamp: new Date(sequence * 1000).toISOString(), type: "model.completed", kind: "model", phase: "reasoning", status: "success", title: "模型规划", summary: "done", payload: {} };
}

describe("task activity reducer", () => {
  it("ignores duplicate and stale sequences", () => {
    let state = reduceTaskActivity(emptyTaskActivityState(), event(1));
    state = reduceTaskActivity(state, event(1, "duplicate-id"));
    state = reduceTaskActivity(state, event(2));
    expect(state.events.map((item) => item.sequence)).toEqual([1, 2]);
  });

  it("merges older history without duplicating live events", () => {
    const live = reduceTaskActivity(emptyTaskActivityState(), [event(3), event(4)]);
    const state = mergeTaskActivityHistory(live, [event(1), event(2), event(3)]);
    expect(state.events.map((item) => item.sequence)).toEqual([1, 2, 3, 4]);
    expect(state.cursor).toBe(4);
  });
});
