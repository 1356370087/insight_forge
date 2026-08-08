import { NextResponse, type NextRequest } from "next/server";

const publicPages = ["/login", "/register", "/verify-email", "/forgot-password", "/reset-password"];

export function proxy(request: NextRequest) {
  if (process.env.NEXT_PUBLIC_LOCAL_DEV_AUTH_BYPASS === "true") return NextResponse.next();
  const path = request.nextUrl.pathname;
  const isPublic = publicPages.some((item) => path === item || path.startsWith(`${item}/`));
  const hasSession = Boolean(request.cookies.get("odr.access") || request.cookies.get("odr.refresh"));
  if (!hasSession && !isPublic) return NextResponse.redirect(new URL("/login", request.url));
  if (hasSession && ["/login", "/register"].includes(path)) return NextResponse.redirect(new URL("/research/new", request.url));
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|api/).*)"],
};
