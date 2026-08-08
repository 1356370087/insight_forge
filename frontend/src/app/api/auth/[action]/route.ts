import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { backendOrigin, clearSession, csrfValid, refreshSession, setSession } from "@/lib/server-auth";

const paths: Record<string, string> = {
  login: "/auth/login",
  register: "/auth/register",
  "verify-email": "/auth/verify-email",
  "resend-verification": "/auth/email-verification/resend",
  "forgot-password": "/auth/password/forgot",
  "reset-password": "/auth/password/reset",
};

export async function POST(request: NextRequest, context: { params: Promise<{ action: string }> }) {
  const { action } = await context.params;
  if (action === "refresh") {
    if (!csrfValid(request)) return NextResponse.json({ detail: "csrf_validation_failed" }, { status: 403 });
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
    if (!csrfValid(request)) return NextResponse.json({ detail: "csrf_validation_failed" }, { status: 403 });
    const access = request.cookies.get("odr.access")?.value;
    if (access) {
      await fetch(new URL("/auth/logout", backendOrigin), { method: "POST", headers: { Authorization: `Bearer ${access}` }, cache: "no-store" });
    }
    const response = NextResponse.json({ ok: true });
    clearSession(response);
    return response;
  }
  const upstreamPath = paths[action];
  if (!upstreamPath) return NextResponse.json({ detail: "auth_action_not_allowed" }, { status: 404 });
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
