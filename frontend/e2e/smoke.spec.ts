import { expect, test } from "@playwright/test";

test("local bypass opens the research intake at all breakpoints", async ({ page }) => {
  await page.goto("/research/new");
  await expect(page.getByRole("heading", { name: /把一个问题/ })).toBeVisible();
  await expect(page.getByLabel("研究问题")).toBeVisible();
});
