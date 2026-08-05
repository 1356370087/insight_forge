export const messages = {
  "zh-CN": { app: { name: "开放深度研究", newResearch: "新建研究", history: "研究历史", settings: "配置中心", sources: "实时来源", findings: "研究发现", report: "最终报告", process: "过程视图" } },
  en: { app: { name: "Open Deep Research", newResearch: "New research", history: "Research history", settings: "Settings", sources: "Live sources", findings: "Findings", report: "Final report", process: "Process" } },
} as const;

export type Locale = keyof typeof messages;
