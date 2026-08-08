export const localAuthBypass = process.env.NEXT_PUBLIC_LOCAL_DEV_AUTH_BYPASS === "true";

export type AuthUser = {
  id: string;
  email: string;
  display_name: string | null;
  status: "pending_email" | "pending_approval" | "active" | "disabled" | "password_reset_required";
  roles: string[];
  permissions: string[];
  authz_version: number;
  session_id?: string | null;
};

function cookieValue(name: string): string | null {
  if (typeof document === "undefined") return null;
  const prefix = `${encodeURIComponent(name)}=`;
  const item = document.cookie.split("; ").find((part) => part.startsWith(prefix));
  return item ? decodeURIComponent(item.slice(prefix.length)) : null;
}

export function csrfHeaders(init?: HeadersInit): Headers {
  const headers = new Headers(init);
  const token = cookieValue("odr.csrf");
  if (token) headers.set("X-CSRF-Token", token);
  return headers;
}

export async function refreshBrowserSession(): Promise<boolean> {
  if (localAuthBypass) return true;
  const response = await fetch("/api/auth/refresh", {
    method: "POST",
    credentials: "same-origin",
    headers: csrfHeaders(),
  });
  return response.ok;
}

export async function authFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    credentials: "same-origin",
    headers: csrfHeaders(init.headers),
  });
  if (!response.ok) {
    if (response.status === 401 && typeof window !== "undefined") {
      window.location.replace("/login");
    }
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(String(body.detail ?? body.message ?? `request_failed:${response.status}`));
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const iamApi = {
  me: () => authFetch<AuthUser>("/api/iam/auth/me"),
  sessions: () => authFetch<Array<Record<string, unknown>>>("/api/iam/auth/sessions"),
  revokeSession: (id: string) => authFetch(`/api/iam/auth/sessions/${encodeURIComponent(id)}`, { method: "DELETE" }),
  logoutAll: () => authFetch<{ revoked_count: number }>("/api/iam/auth/logout-all", { method: "POST" }),
  changePassword: (current_password: string, new_password: string) => authFetch(
    "/api/iam/auth/password/change",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ current_password, new_password }) },
  ),
};
