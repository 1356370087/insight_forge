import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { authenticatedProxy } from "@/lib/server-auth";

const allowed = /^(auth\/(me|sessions(?:\/[^/]+)?|logout-all|password\/change)|admin\/(users(?:\/.*)?|roles(?:\/.*)?|permissions|audit-events))$/;

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  // Dot segments would survive encodeURIComponent and let new URL() rewrite
  // the path below the allowlist after the regex has already matched.
  if (path.some((segment) => segment === "." || segment === "..")) {
    return NextResponse.json({ detail: "proxy_path_not_allowed" }, { status: 404 });
  }
  const joined = path.join("/");
  if (!allowed.test(joined)) return NextResponse.json({ detail: "proxy_path_not_allowed" }, { status: 404 });
  return authenticatedProxy(request, `/${path.map(encodeURIComponent).join("/")}`);
}

export const dynamic = "force-dynamic";
export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
