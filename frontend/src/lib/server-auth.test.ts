import { beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

vi.mock("server-only", () => ({}));

describe("authenticatedProxy local development bypass", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.stubEnv("NODE_ENV", "development");
    vi.stubEnv("NEXT_PUBLIC_LOCAL_DEV_AUTH_BYPASS", "true");
  });

  it("forwards mutation requests without auth cookies or CSRF", async () => {
    const upstream = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ run_id: "run-local" }), {
        status: 201,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const { authenticatedProxy } = await import("./server-auth");
    const request = new NextRequest("http://localhost/api/research/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [] }),
    });

    const response = await authenticatedProxy(request, "/runs");

    expect(response.status).toBe(201);
    expect(await response.json()).toEqual({ run_id: "run-local" });
    const [, init] = upstream.mock.calls[0];
    expect(new Headers(init?.headers).get("Authorization")).toBe(
      "Bearer local-dev-bypass",
    );
  });
});
