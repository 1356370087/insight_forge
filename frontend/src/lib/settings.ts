import { z } from "zod";

export const SETTINGS_KEY = "odr.frontend.settings.v1";
export const PRESETS_KEY = "odr.frontend.presets.v1";

export const settingGroups: Array<{ id: string; label: string; keys: string[] }> = [
  { id: "basic", label: "基础研究", keys: ["allow_clarification", "enable_async_research"] },
  { id: "models", label: "模型", keys: ["summarization_model", "summarization_model_max_tokens", "research_model", "research_model_max_tokens", "compression_model", "compression_model_max_tokens", "final_report_model", "final_report_model_max_tokens"] },
  { id: "evidence", label: "搜索与证据", keys: ["search_api", "web_pipeline_mode", "web_pipeline_shadow_sample_rate", "web_min_source_authority", "search_candidate_limit", "max_fetches_per_researcher"] },
  { id: "agents", label: "Agent 调度", keys: ["max_concurrent_research_units", "max_researcher_iterations", "max_react_tool_calls"] },
  { id: "hitl", label: "HITL", keys: ["enable_human_in_loop", "hitl_require_plan_approval", "hitl_require_outline_approval", "hitl_max_plan_revisions", "hitl_feedback_mode"] },
  { id: "report", label: "报告", keys: ["report_type", "output_format"] },
  { id: "quality", label: "质量门禁", keys: ["quality_evaluation_enabled", "quality_evaluation_model", "quality_evaluation_model_max_tokens", "quality_evaluation_rigor", "quality_evaluation_min_sources", "quality_evaluation_max_input_chars", "quality_risk_mode", "quality_evaluation_fail_open", "quality_caveat_admission_enabled", "quality_gap_recovery_max_attempts"] },
  { id: "memory", label: "长期记忆", keys: ["enable_memory", "memory_top_k", "memory_min_confidence", "memory_auto_write", "memory_write_after_report", "memory_fail_open", "memory_advanced_enabled", "memory_decay_enabled", "memory_reflection_enabled", "memory_profile_enabled", "memory_soft_forgetting_enabled", "memory_verified_insights_enabled", "memory_search_threshold", "memory_search_rerank", "memory_importance_weight", "memory_relevance_weight", "memory_recency_weight", "memory_reflection_observation_threshold", "memory_reflection_importance_threshold", "memory_reflection_max_age_hours", "memory_profile_max_chars"] },
];

const objectSchema = z.record(z.string(), z.unknown());

export function loadSettings(defaults: Record<string, unknown> = {}) {
  if (typeof window === "undefined") return defaults;
  try { return { ...defaults, ...objectSchema.parse(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}")) }; }
  catch { return defaults; }
}

export function sanitizeSettings(value: unknown, capabilities: { editable_config_keys?: string[]; config_schema?: { properties?: Record<string, { type?: string; minimum?: number; maximum?: number }> } }) {
  const raw = objectSchema.parse(value);
  const allowed = new Set(capabilities.editable_config_keys ?? []);
  const properties = capabilities.config_schema?.properties ?? {};
  return Object.fromEntries(Object.entries(raw).filter(([key, item]) => {
    if (!allowed.has(key)) return false;
    const rule = properties[key];
    if (rule?.type === "boolean" && typeof item !== "boolean") return false;
    if ((rule?.type === "number" || rule?.type === "integer") && typeof item !== "number") return false;
    if (typeof item === "number" && rule?.minimum !== undefined && item < rule.minimum) return false;
    if (typeof item === "number" && rule?.maximum !== undefined && item > rule.maximum) return false;
    return true;
  }));
}

export function saveSettings(value: Record<string, unknown>) { localStorage.setItem(SETTINGS_KEY, JSON.stringify(value)); }
