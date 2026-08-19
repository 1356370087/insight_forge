import { beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

vi.mock("server-only", () => ({}));

describe("sameOriginValid", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  it("accepts requests without an Origin header (non-browser clients)", async () => {
    const { sameOriginValid } = await import("./server-auth");
    const request = new NextRequest("http://localhost:3000/api/auth/login", {
      method: "POST",
    });
    expect(sameOriginValid(request)).toBe(true);
  });

  it("accepts a matching same-origin request", async () => {
    const { sameOriginValid } = await import("./server-auth");
    const request = new NextRequest("http://localhost:3000/api/auth/login", {
      method: "POST",
      headers: { Origin: "http://localhost:3000" },
    });
    expect(sameOriginValid(request)).toBe(true);
  });

  it("rejects a foreign origin (login CSRF)", async () => {
    const { sameOriginValid } = await import("./server-auth");
    const request = new NextRequest("http://localhost:3000/api/auth/login", {
      method: "POST",
      headers: { Origin: "https://evil.example" },
    });
    expect(sameOriginValid(request)).toBe(false);
  });

  it("rejects a different port on the same host", async () => {
    const { sameOriginValid } = await import("./server-auth");
    const request = new NextRequest("http://localhost:3000/api/auth/login", {
      method: "POST",
      headers: { Origin: "http://localhost:3001" },
    });
    expect(sameOriginValid(request)).toBe(false);
  });

  it("rejects an unparseable Origin value", async () => {
    const { sameOriginValid } = await import("./server-auth");
    const request = new NextRequest("http://localhost:3000/api/auth/login", {
      method: "POST",
      headers: { Origin: "null" },
    });
    // "null" parses as a relative URL whose host is empty → mismatch.
    expect(sameOriginValid(request)).toBe(false);
  });
});

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
