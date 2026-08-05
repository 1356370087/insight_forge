"use client";

import { create } from "zustand";
import { emptyRunState, hydrateSnapshot, reducePublicEvent } from "@/lib/run-reducer";
import type { ConnectionState, PublicEvent, ResearchRunState, RunSnapshot } from "@/lib/types";

interface ResearchRunActions {
  reset: (runId: string) => void;
  hydrate: (snapshot: RunSnapshot) => void;
  applyEvent: (event: PublicEvent) => void;
  setConnection: (connectionState: ConnectionState, reconnecting?: boolean) => void;
}

export const useResearchRunStore = create<ResearchRunState & ResearchRunActions>((set) => ({
  ...emptyRunState(),
  reset: (runId) => set({ ...emptyRunState(runId) }),
  hydrate: (snapshot) => set((state) => hydrateSnapshot(state, snapshot)),
  applyEvent: (event) => set((state) => reducePublicEvent(state, event)),
  setConnection: (connectionState, isReconnecting = false) => set({ connectionState, isReconnecting }),
}));
