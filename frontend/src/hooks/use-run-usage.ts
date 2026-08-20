"use client";

import { useQuery } from "@tanstack/react-query";
import { researchApi } from "@/lib/api";

export function useRunUsage(runId: string, visible = true, terminal = false) {
  return useQuery({
    queryKey: ["run-usage", runId],
    queryFn: () => researchApi.runUsage(runId),
    enabled: Boolean(runId) && visible,
    refetchInterval: visible && !terminal ? 5_000 : false,
    staleTime: 250,
    retry: 2,
  });
}
