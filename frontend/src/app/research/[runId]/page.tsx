import { ResearchWorkspace } from "@/components/research-workspace";

export default async function RunPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  return <ResearchWorkspace runId={runId} />;
}
