import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TokenUsageDashboard } from "./token-usage-dashboard";
import type { RunUsageResponse } from "@/lib/types";

const useRunUsage = vi.fn();
vi.mock("@/hooks/use-run-usage", () => ({ useRunUsage: (...args: unknown[]) => useRunUsage(...args) }));
vi.mock("recharts", () => {
  const Node = ({ children }: { children?: React.ReactNode }) => <div>{children}</div>;
  return { ResponsiveContainer: Node, BarChart: Node, PieChart: Node, LineChart: Node, ComposedChart: Node, Bar: Node, Pie: Node, Line: Node, Area: Node, Cell: Node, CartesianGrid: Node, Legend: Node, Tooltip: Node, XAxis: Node, YAxis: Node };
});

const vector = (total_tokens = 0) => ({ input_tokens: Math.floor(total_tokens * 0.7), output_tokens: Math.ceil(total_tokens * 0.3), total_tokens, cached_input_tokens: 0, cache_creation_input_tokens: 0, reasoning_tokens: 0 });
const response: RunUsageResponse = {
  schema_version: 1,
  run_id: "run-1",
  status: "running",
  duration_ms: 12_000,
  revision: 4,
  updated_at: 1,
  accounting_status: "partial",
  totals: {
    reported: vector(120), estimated: vector(30),
    calls: { attempts: 3, successful_responses: 2, provider_reported: 1, provider_partial: 0, estimated: 1, missing: 1, unknown_failed_attempts: 1, legacy_unclassified: 0, coverage_ratio: 0.5 },
    cost: { estimated_cost_micro_usd: null, cost_source: "unavailable", price_table_hash: null },
    budgets: { input_tokens: { settled: 84, estimated: 21, reserved: 0, limit: null }, output_tokens: { settled: 36, estimated: 9, reserved: 0, limit: null }, model_calls: { settled: 3, estimated: 0, reserved: 0, limit: null }, cost_micro_usd: { settled: null, estimated: 0, reserved: 0, limit: null } },
  },
  breakdowns: { by_stage: [], by_agent_role: [], by_model: [], by_task: [] },
  timeline: [],
  operations: { llm_call_count: 2, retry_count: 1, rate_limited_count: 1, rate_429: 0.5, cache_hit_rate: 0, cache_input_ratio: 0, reasoning_output_ratio: 0, output_tokens_per_second: 12, tool_call_count: 2, tool_success_rate: 1, empty_tool_result_count: 0, zero_source_search_count: 0 },
};

describe("TokenUsageDashboard", () => {
  beforeEach(() => useRunUsage.mockReturnValue({ isLoading: false, isError: false, data: response }));

  it("renders reported and estimated tracks without calling estimates real", () => {
    render(<TokenUsageDashboard runId="run-1" visible terminal={false} />);
    expect(screen.getAllByText("Provider 实报").length).toBeGreaterThan(0);
    expect(screen.getAllByText("估算补位").length).toBeGreaterThan(0);
    expect(screen.getAllByText("50.0%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("未配置").length).toBeGreaterThan(0);
    expect(screen.queryByText("真实 Token")).not.toBeInTheDocument();
  });

  it("distinguishes unavailable accounting from zero usage", () => {
    useRunUsage.mockReturnValue({ isLoading: false, isError: false, data: { ...response, accounting_status: "unavailable" } });
    render(<TokenUsageDashboard runId="run-1" visible terminal />);
    expect(screen.getByText("Token 核算不可用")).toBeInTheDocument();
  });

  it("formats cost budget limits in micro USD instead of token units", () => {
    useRunUsage.mockReturnValue({
      isLoading: false,
      isError: false,
      data: {
        ...response,
        totals: {
          ...response.totals,
          budgets: {
            ...response.totals.budgets,
            cost_micro_usd: { settled: 1_000, estimated: 0, reserved: 0, limit: 2_000 },
          },
        },
      },
    });
    render(<TokenUsageDashboard runId="run-1" visible terminal />);
    expect(screen.getByText("$0.0010 / $0.0020")).toBeInTheDocument();
  });
});
