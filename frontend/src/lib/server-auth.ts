import "server-only";

import { createHash } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";

export const ACCESS_COOKIE = "odr.access";
export const REFRESH_COOKIE = "odr.refresh";
export const CSRF_COOKIE = "odr.csrf";
export const backendOrigin = process.env.RESEARCH_API_ORIGIN ?? "http://127.0.0.1:2024";

type TokenPair = {
  access_token: string;
  refresh_token: string;
  expires_in: number;
  refresh_expires_in: number;
};

const secure = process.env.NODE_ENV === "production";
const refreshFlights = new Map<string, Promise<TokenPair | null>>();
const hopByHopHeaders = new Set([
  "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
  "trailer", "transfer-encoding", "upgrade", "host", "content-length", "cookie", "authorization",
]);

function localDevAuthBypassEnabled(): boolean {
  return process.env.NODE_ENV !== "production"
    && process.env.NEXT_PUBLIC_LOCAL_DEV_AUTH_BYPASS === "true";
}

export function csrfValid(request: NextRequest): boolean {
  const cookie = request.cookies.get(CSRF_COOKIE)?.value;
  const header = request.headers.get("x-csrf-token");
  return Boolean(cookie && header && cookie.length >= 16 && cookie === header);
}

// Login-CSRF guard for cookie-less auth actions: a browser form/fetch from
// another site carries a foreign Origin header, while non-browser clients
// send none. Compares host (name + port); the scheme may legitimately differ
// behind a TLS-terminating proxy.
export function sameOriginValid(request: NextRequest): boolean {
  const origin = request.headers.get("origin");
  if (!origin) return true;
  try {
    return new URL(origin).host === request.nextUrl.host;
  } catch {
    return false;
  }
}

export function clearSession(response: NextResponse): void {
  for (const name of [ACCESS_COOKIE, REFRESH_COOKIE, CSRF_COOKIE]) {
    response.cookies.set(name, "", { path: "/", maxAge: 0, httpOnly: name !== CSRF_COOKIE, secure, sameSite: "lax" });
  }
}

export function setSession(response: NextResponse, pair: TokenPair, csrf?: string): void {
  response.cookies.set(ACCESS_COOKIE, pair.access_token, {
    httpOnly: true, secure, sameSite: "lax", path: "/", maxAge: pair.expires_in,
  });
  response.cookies.set(REFRESH_COOKIE, pair.refresh_token, {
    httpOnly: true, secure, sameSite: "lax", path: "/", maxAge: pair.refresh_expires_in,
  });
  response.cookies.set(CSRF_COOKIE, csrf ?? crypto.randomUUID(), {
    httpOnly: false, secure, sameSite: "lax", path: "/", maxAge: pair.refresh_expires_in,
  });
}

async function performRefresh(refresh: string): Promise<TokenPair | null> {
  const upstream = await fetch(new URL("/auth/refresh", backendOrigin), {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ refresh_token: refresh }),
    cache: "no-store",
  });
  if (!upstream.ok) return null;
  return upstream.json() as Promise<TokenPair>;
}

export async function refreshSession(request: NextRequest): Promise<TokenPair | null> {
  const refresh = request.cookies.get(REFRESH_COOKIE)?.value;
  if (!refresh) return null;
  const key = createHash("sha256").update(refresh).digest("hex");
  const active = refreshFlights.get(key);
  if (active) return active;
  const flight = performRefresh(refresh).finally(() => refreshFlights.delete(key));
  refreshFlights.set(key, flight);
  return flight;
}

function cleanHeaders(request: NextRequest, accessToken: string): Headers {
  const headers = new Headers(request.headers);
  for (const name of hopByHopHeaders) headers.delete(name);
  headers.delete("x-csrf-token");
  headers.set("Authorization", `Bearer ${accessToken}`);
  return headers;
}

async function callUpstream(request: NextRequest, target: URL, accessToken: string, body?: ArrayBuffer): Promise<Response> {
  return fetch(target, {
    method: request.method,
    headers: cleanHeaders(request, accessToken),
    body,
    cache: "no-store",
    redirect: "manual",
  });
}

export async function authenticatedProxy(request: NextRequest, upstreamPath: string): Promise<NextResponse> {
  const localDevBypass = localDevAuthBypassEnabled();
  if (!localDevBypass && !["GET", "HEAD", "OPTIONS"].includes(request.method) && !csrfValid(request)) {
    return NextResponse.json({ detail: "csrf_validation_failed" }, { status: 403 });
  }
  const target = new URL(upstreamPath, backendOrigin);
  target.search = request.nextUrl.search;
  const body = ["GET", "HEAD"].includes(request.method) ? undefined : await request.arrayBuffer();
  if (localDevBypass) {
    const upstream = await callUpstream(request, target, "local-dev-bypass", body);
    const headers = new Headers(upstream.headers);
    for (const name of hopByHopHeaders) headers.delete(name);
    headers.set("Cache-Control", upstream.headers.get("content-type")?.includes("text/event-stream") ? "no-cache, no-transform" : "no-store");
    headers.set("X-Accel-Buffering", "no");
    return new NextResponse(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers,
    });
  }
  let access = request.cookies.get(ACCESS_COOKIE)?.value;
  let pair: TokenPair | null = null;
  if (!access) {
    pair = await refreshSession(request);
    access = pair?.access_token;
  }
  if (!access) {
    const response = NextResponse.json({ detail: "not_authenticated" }, { status: 401 });
    clearSession(response);
    return response;
  }
  let upstream = await callUpstream(request, target, access, body);
  if (upstream.status === 401 && !pair) {
    pair = await refreshSession(request);
    if (pair) upstream = await callUpstream(request, target, pair.access_token, body);
  }
  const headers = new Headers(upstream.headers);
  for (const name of hopByHopHeaders) headers.delete(name);
  headers.set("Cache-Control", upstream.headers.get("content-type")?.includes("text/event-stream") ? "no-cache, no-transform" : "no-store");
  headers.set("X-Accel-Buffering", "no");
  const response = new NextResponse(upstream.body, { status: upstream.status, statusText: upstream.statusText, headers });
  if (pair) setSession(response, pair, request.cookies.get(CSRF_COOKIE)?.value);
  if (upstream.status === 401) clearSession(response);
  return response;
}
