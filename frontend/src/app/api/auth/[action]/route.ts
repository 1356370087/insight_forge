import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import {
  ACCESS_COOKIE,
  backendOrigin,
  clearSession,
  csrfValid,
  refreshSession,
  sameOriginValid,
  setSession,
} from "@/lib/server-auth";

const paths: Record<string, string> = {
  login: "/auth/login",
  register: "/auth/register",
  "verify-email": "/auth/verify-email",
  "resend-verification": "/auth/email-verification/resend",
  "forgot-password": "/auth/password/forgot",
  "reset-password": "/auth/password/reset",
};

async function revokeUpstream(accessToken: string): Promise<Response> {
  return fetch(new URL("/auth/logout", backendOrigin), {
    method: "POST",
    headers: { Authorization: `Bearer ${accessToken}` },
    cache: "no-store",
  });
}

export async function POST(request: NextRequest, context: { params: Promise<{ action: string }> }) {
  const { action } = await context.params;
  if (action === "refresh") {
    if (!csrfValid(request) || !sameOriginValid(request)) {
      return NextResponse.json({ detail: "csrf_validation_failed" }, { status: 403 });
    }
    const pair = await refreshSession(request);
    if (!pair) {
      const response = NextResponse.json({ detail: "session_expired" }, { status: 401 });
      clearSession(response);
      return response;
    }
    const response = NextResponse.json({ ok: true });
    setSession(response, pair, request.cookies.get("odr.csrf")?.value);
    return response;
  }
  if (action === "logout") {
    if (!csrfValid(request) || !sameOriginValid(request)) {
      return NextResponse.json({ detail: "csrf_validation_failed" }, { status: 403 });
    }
    let access = request.cookies.get(ACCESS_COOKIE)?.value;
    if (access) {
      const first = await revokeUpstream(access);
      // An expired access token still identifies the session holder: rotate
      // the refresh cookie once and revoke through the fresh pair so the
      // server-side session really dies instead of lingering until TTL.
      if (first.status === 401) access = undefined;
    }
    if (!access) {
      const pair = await refreshSession(request);
      if (pair?.access_token) await revokeUpstream(pair.access_token);
    }
    const response = NextResponse.json({ ok: true });
    clearSession(response);
    return response;
  }
  const upstreamPath = paths[action];
  if (!upstreamPath) return NextResponse.json({ detail: "auth_action_not_allowed" }, { status: 404 });
  if (!sameOriginValid(request)) return NextResponse.json({ detail: "csrf_validation_failed" }, { status: 403 });
  const upstream = await fetch(new URL(upstreamPath, backendOrigin), {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: await request.text(),
    cache: "no-store",
  });
  const data = await upstream.json().catch(() => ({ detail: upstream.statusText }));
  const response = NextResponse.json(action === "login" && upstream.ok ? { ok: true } : data, { status: upstream.status });
  if (action === "login" && upstream.ok) setSession(response, data);
  return response;
}
