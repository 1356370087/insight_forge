import { expect, test } from "@playwright/test";

test.skip(
  process.env.NEXT_PUBLIC_LOCAL_DEV_AUTH_BYPASS !== "true",
  "This scenario only applies when the explicit local authentication bypass is enabled.",
);

test("local bypass opens the research intake at all breakpoints", async ({ page }) => {
  await page.goto("/research/new");
  await expect(page.getByRole("heading", { name: /把一个问题/ })).toBeVisible();
  await expect(page.getByLabel("研究问题")).toBeVisible();
});
