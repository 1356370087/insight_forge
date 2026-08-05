import { createBrowserClient } from "@supabase/ssr";

export const localAuthBypass = process.env.NEXT_PUBLIC_LOCAL_DEV_AUTH_BYPASS === "true";

export function getSupabaseBrowserClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  return url && key ? createBrowserClient(url, key) : null;
}

export async function getAccessToken(refresh = false): Promise<string | null> {
  if (localAuthBypass) return null;
  const client = getSupabaseBrowserClient();
  if (!client) return null;
  const result = refresh ? await client.auth.refreshSession() : await client.auth.getSession();
  return result.data.session?.access_token ?? null;
}
