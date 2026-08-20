import { AppShell } from "@/components/app-shell";
import { UsageAnalyticsDashboard } from "@/components/usage-analytics-dashboard";

export default function UsagePage() {
  return <AppShell><main className="analytics-page"><UsageAnalyticsDashboard /></main></AppShell>;
}
