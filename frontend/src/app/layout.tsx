import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/noto-sans-sc/400.css";
import "./globals.css";
import type { Metadata } from "next";
import { cookies } from "next/headers";
import { Providers } from "./providers";
import type { Locale } from "@/i18n/messages";

export const metadata: Metadata = { title: "Open Deep Research Console", description: "Event-driven research command center" };

export default async function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const cookieStore = await cookies();
  const raw = cookieStore.get("odr.locale")?.value;
  const locale: Locale = raw === "en" ? "en" : "zh-CN";
  return <html lang={locale}><body><Providers locale={locale}>{children}</Providers></body></html>;
}
