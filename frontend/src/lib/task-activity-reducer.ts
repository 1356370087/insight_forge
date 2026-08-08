import type { TaskActivityEvent } from "./types";

export interface TaskActivityTimelineState {
  events: TaskActivityEvent[];
  cursor: number;
  eventIds: Set<string>;
}

export function emptyTaskActivityState(): TaskActivityTimelineState {
  return { events: [], cursor: 0, eventIds: new Set() };
}

export function reduceTaskActivity(
  state: TaskActivityTimelineState,
  incoming: TaskActivityEvent | TaskActivityEvent[],
): TaskActivityTimelineState {
  const candidates = Array.isArray(incoming) ? incoming : [incoming];
  const eventIds = new Set(state.eventIds);
  const events = [...state.events];
  let cursor = state.cursor;
  for (const event of candidates.sort((left, right) => left.sequence - right.sequence)) {
    if (event.sequence <= cursor || eventIds.has(event.event_id)) continue;
    events.push(event);
    eventIds.add(event.event_id);
    cursor = event.sequence;
  }
  return { events, cursor, eventIds };
}

export function mergeTaskActivityHistory(
  state: TaskActivityTimelineState,
  history: TaskActivityEvent[],
): TaskActivityTimelineState {
  const byId = new Map<string, TaskActivityEvent>();
  for (const event of [...history, ...state.events]) byId.set(event.event_id, event);
  const events = [...byId.values()].sort((left, right) => left.sequence - right.sequence);
  return {
    events,
    cursor: Math.max(state.cursor, ...events.map((event) => event.sequence), 0),
    eventIds: new Set(events.map((event) => event.event_id)),
  };
}
