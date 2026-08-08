import { expect, test } from "@playwright/test";

const email = process.env.E2E_ADMIN_EMAIL;
const password = process.env.E2E_ADMIN_PASSWORD;

test.describe("self-hosted IAM", () => {
  test.skip(!email || !password, "Set E2E_ADMIN_EMAIL and E2E_ADMIN_PASSWORD for live IAM E2E");

  test("logs in through BFF, exposes no JWT, and opens admin control plane", async ({ page, context }) => {
    await page.goto("/login");
    await page.getByLabel("工作邮箱").fill(email!);
    await page.getByLabel("密码").fill(password!);
    await page.getByRole("button", { name: "进入研究台" }).click();
    await expect(page).toHaveURL(/\/research\/new$/);
    await expect(page.getByText("E2E Administrator")).toBeVisible();

    const cookies = await context.cookies();
    expect(cookies.find((cookie) => cookie.name === "odr.access")?.httpOnly).toBe(true);
    expect(cookies.find((cookie) => cookie.name === "odr.refresh")?.httpOnly).toBe(true);
    const browserStorage = await page.evaluate(() => JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage } }));
    expect(browserStorage).not.toContain("eyJ");

    if ((page.viewportSize()?.width ?? 1440) < 768) {
      await page.getByRole("button", { name: "打开导航" }).click();
    }
    await page.getByRole("link", { name: "身份管理" }).click();
    await expect(page.getByRole("heading", { name: "身份与权限管理" })).toBeVisible();
    await expect(page.getByText("管理员权限不会赋予读取其他用户研究内容的能力")).toBeVisible();
  });

  test("creates a real research run when explicitly enabled", async ({ page }) => {
    test.skip(process.env.E2E_RUN_RESEARCH !== "true", "Set E2E_RUN_RESEARCH=true for the paid live model run");
    await page.goto("/login");
    await page.getByLabel("工作邮箱").fill(email!);
    await page.getByLabel("密码").fill(password!);
    await page.getByRole("button", { name: "进入研究台" }).click();
    await page.getByLabel("研究问题").fill("请比较 2025 年 PostgreSQL 17 与 16 在 JSON 与查询规划方面的关键变化，并给出可核验来源。");
    await page.getByRole("button", { name: /启动研究/ }).click();
    await expect(page).toHaveURL(/\/research\/[^/]+$/);
    await expect(page.getByText(/已连接|连接中|运行中/).first()).toBeVisible({ timeout: 20_000 });
  });
});
