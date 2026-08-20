import { expect, test } from "@playwright/test";

test.skip(
  process.env.NEXT_PUBLIC_LOCAL_DEV_AUTH_BYPASS !== "true",
  "Usage UI fixture runs with the explicit local authentication bypass.",
);

test("usage analytics keeps reported and estimated tracks visible", async ({ page }) => {
  await page.route("**/api/research/runs**", async (route) => {
    await route.fulfill({ json: { items: [] } });
  });
  await page.route("**/api/research/usage/analytics**", async (route) => {
    await route.fulfill({
      json: {
        schema_version: 1,
        range: "30d",
        retention_days: 30,
        actual_range_days: 2,
        summary: {
          reported: { input_tokens: 900, output_tokens: 300, total_tokens: 1200, cached_input_tokens: 100, cache_creation_input_tokens: 0, reasoning_tokens: 0 },
          estimated: { input_tokens: 160, output_tokens: 40, total_tokens: 200, cached_input_tokens: 0, cache_creation_input_tokens: 0, reasoning_tokens: 0 },
          run_count: 1,
          coverage_ratio: 0.8,
          estimated_cost_micro_usd: 4200,
        },
        daily: [{ date: "2026-08-20", reported_tokens: 1200, estimated_tokens: 200, coverage_ratio: 0.8, cache_hit_rate: 0.1, rate_429: 0, output_tokens_per_second: 18 }],
        distributions: {
          provider: [{ key: "openai", label: "openai", reported_tokens: 1200, estimated_tokens: 200 }],
          model: [{ key: "gpt-test", label: "gpt-test", reported_tokens: 1200, estimated_tokens: 200 }],
          status: [{ key: "completed", label: "completed", reported_tokens: 1200, estimated_tokens: 200 }],
        },
        runs: [{
          run_id: "usage-e2e",
          title: "Token 核算回归",
          status: "completed",
          accounting_status: "partial",
          reported: { input_tokens: 900, output_tokens: 300, total_tokens: 1200, cached_input_tokens: 100, cache_creation_input_tokens: 0, reasoning_tokens: 0 },
          estimated: { input_tokens: 160, output_tokens: 40, total_tokens: 200, cached_input_tokens: 0, cache_creation_input_tokens: 0, reasoning_tokens: 0 },
          calls: { coverage_ratio: 0.8 },
          cost: { estimated_cost_micro_usd: 4200, cost_source: "configured_estimate" },
          operations: { rate_429: 0, output_tokens_per_second: 18 },
        }],
        next_cursor: null,
      },
    });
  });

  await page.goto("/usage");
  await expect(page.getByRole("heading", { name: "用量分析" })).toBeVisible();
  await expect(page.getByText("1,200").first()).toBeVisible();
  await expect(page.getByText("200").first()).toBeVisible();
  await expect(page.getByRole("link", { name: /Token 核算回归/ })).toBeVisible();
  await expect(page.getByText("80.0%").first()).toBeVisible();
});
