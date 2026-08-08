import { afterEach, describe, expect, it, vi } from "vitest";
import { authFetch, csrfHeaders, refreshBrowserSession } from "./auth";

afterEach(() => {
  vi.restoreAllMocks();
  document.cookie = "odr.csrf=; Max-Age=0; path=/";
});

describe("browser IAM client", () => {
  it("copies the double-submit CSRF value without exposing JWTs", () => {
    document.cookie = "odr.csrf=csrf-token-123456; path=/";
    const headers = csrfHeaders({ "Content-Type": "application/json" });
    expect(headers.get("X-CSRF-Token")).toBe("csrf-token-123456");
    expect(headers.has("Authorization")).toBe(false);
  });

  it("sends same-origin credentials and parses successful JSON", async () => {
    document.cookie = "odr.csrf=csrf-token-123456; path=/";
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    await expect(authFetch<{ ok: boolean }>("/api/iam/auth/me")).resolves.toEqual({ ok: true });
    expect(fetchMock).toHaveBeenCalledWith("/api/iam/auth/me", expect.objectContaining({ credentials: "same-origin" }));
  });

  it("reports a failed refresh without returning token material", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 401 }));
    await expect(refreshBrowserSession()).resolves.toBe(false);
  });
});
