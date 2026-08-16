"use client";

import { AssistantRuntimeProvider, ComposerPrimitive, useLocalRuntime, type ChatModelAdapter } from "@assistant-ui/react";
import { ArrowUpRight } from "lucide-react";
import { useRouter } from "next/navigation";
import { useMemo } from "react";
import { researchApi } from "@/lib/api";
import { loadSettings } from "@/lib/settings";
import { Button } from "@/components/ui/button";

export function ResearchComposer() {
  const router = useRouter();
  const adapter = useMemo<ChatModelAdapter>(() => ({
    async run({ messages, abortSignal }) {
      const last = messages.at(-1);
      const query = last?.content.filter((item) => item.type === "text").map((item) => item.text).join("\n").trim() ?? "";
      if (!query) return { content: [{ type: "text", text: "请输入研究问题。" }] };
      const created = await researchApi.createRun(query, loadSettings(), query.slice(0, 80));
      if (!abortSignal.aborted) router.push(`/research/${created.run_id}`);
      return { content: [{ type: "text", text: "研究任务已创建，正在连接实时事件流。" }] };
    },
  }), [router]);
  const runtime = useLocalRuntime(adapter);
  return <AssistantRuntimeProvider runtime={runtime}>
    <ComposerPrimitive.Root className="research-composer">
      <ComposerPrimitive.Input asChild><textarea aria-label="研究问题" placeholder={"描述一个值得彻底调查的问题……\n例如：比较 2026 年主要端侧推理框架的性能、生态和部署风险。"} /></ComposerPrimitive.Input>
      <div className="composer-footer"><span>ENTER 发送 · SHIFT+ENTER 换行</span><ComposerPrimitive.Send asChild><Button>启动研究 <ArrowUpRight size={16} /></Button></ComposerPrimitive.Send></div>
    </ComposerPrimitive.Root>
  </AssistantRuntimeProvider>;
}
