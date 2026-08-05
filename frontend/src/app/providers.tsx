"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NextIntlClientProvider } from "next-intl";
import { useState, type ReactNode } from "react";
import { messages, type Locale } from "@/i18n/messages";

export function Providers({ children, locale }: { children: ReactNode; locale: Locale }) {
  const [queryClient] = useState(() => new QueryClient({ defaultOptions: { queries: { staleTime: 10_000, retry: 1 } } }));
  return <NextIntlClientProvider locale={locale} messages={messages[locale]}><QueryClientProvider client={queryClient}>{children}</QueryClientProvider></NextIntlClientProvider>;
}
