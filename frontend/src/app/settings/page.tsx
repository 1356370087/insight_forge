"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useQuery } from "@tanstack/react-query";
import { Download, RotateCcw, Save, Upload } from "lucide-react";
import { useEffect, useRef } from "react";
import { useForm, useWatch } from "react-hook-form";
import { z } from "zod";
import { AppShell } from "@/components/app-shell";
import { researchApi } from "@/lib/api";
import { loadSettings, sanitizeSettings, saveSettings, settingGroups } from "@/lib/settings";

type FieldSchema = { type?: string; title?: string; description?: string; default?: unknown; minimum?: number; maximum?: number; enum?: unknown[]; anyOf?: Array<{ type?: string; enum?: unknown[] }> };
type Capabilities = { editable_config_keys: string[]; defaults: Record<string, unknown>; config_schema: { properties: Record<string, FieldSchema>; $defs?: Record<string, FieldSchema> }; features?: { memory?: boolean } };
const settingsSchema = z.record(z.string(), z.unknown());

function SettingControl({ name, schema, value, onChange }: { name: string; schema: FieldSchema; value: unknown; onChange: (value: unknown) => void }) {
  const enums = schema.enum ?? schema.anyOf?.flatMap((item) => item.enum ?? []) ?? [];
  const type = schema.type ?? (typeof value === "boolean" ? "boolean" : typeof value === "number" ? "number" : "string");
  if (type === "boolean") return <button type="button" role="switch" aria-label={name} aria-checked={Boolean(value)} className={`switch ${value ? "on" : ""}`} onClick={() => onChange(!value)}><i /></button>;
  if (enums.length) return <select aria-label={name} value={String(value ?? "")} onChange={(event) => onChange(event.target.value || null)}>{enums.map((item) => <option key={String(item)} value={String(item)}>{String(item)}</option>)}</select>;
  if (type === "number" || type === "integer") {
    const min = schema.minimum ?? 0; const max = schema.maximum ?? Math.max(Number(value || 100) * 2, 100);
    return <><input type="range" min={min} max={max} step={type === "integer" ? 1 : .01} value={Number(value ?? min)} onChange={(event) => onChange(Number(event.target.value))} /><input type="number" min={min} max={max} value={Number(value ?? min)} onChange={(event) => onChange(Number(event.target.value))} /></>;
  }
  return <input type="text" value={String(value ?? "")} onChange={(event) => onChange(event.target.value)} />;
}

export default function SettingsPage() {
  const fileRef = useRef<HTMLInputElement>(null);
  const { data } = useQuery({ queryKey: ["capabilities"], queryFn: () => researchApi.capabilities() as Promise<Capabilities> });
  const { control, setValue, reset, handleSubmit } = useForm<Record<string, unknown>>({ resolver: zodResolver(settingsSchema), defaultValues: {} });
  const values = useWatch({ control }) as Record<string, unknown>;
  useEffect(() => { if (data) reset(loadSettings(data.defaults)); }, [data, reset]);
  const save = handleSubmit((form) => saveSettings(sanitizeSettings(form, data ?? {})));
  function exportJson() { const blob = new Blob([JSON.stringify(values, null, 2)], { type: "application/json" }); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = "odr-settings.json"; a.click(); URL.revokeObjectURL(url); }
  async function importJson(file?: File) { if (!file || !data) return; const parsed = JSON.parse(await file.text()); reset({ ...data.defaults, ...sanitizeSettings(parsed, data) }); }
  const inspector = <><h2 className="inspector-title">配置安全边界</h2><div className="inspector-block"><p className="empty-note">仅展示 capabilities 明确白名单中的用户字段。导入时未知字段会被剔除，最终仍由服务端再次校验。</p></div><div className="inspector-block"><p className="eyebrow">永不落盘到浏览器</p><p className="empty-note">JWT、模型 API Key、MCP 配置、端点凭据、沙箱和域名白名单。</p></div></>;
  return <AppShell inspector={inspector}><div className="page"><header className="page-header"><div><span className="eyebrow">USER CONFIG / VERSION 1</span><h1>配置研究行为，<br />不越过安全边界。</h1><p>所有设置保存在版本化本地预设中。运行创建时发送，服务端白名单是最终权威。</p></div></header>{!data ? <div className="panel panel-body">正在读取服务端 capabilities…</div> : <form onSubmit={save} className="settings-grid">{settingGroups.filter((group) => group.id !== "memory" || data.features?.memory).map((group, index) => <details className="settings-group" key={group.id} open={index < 2}><summary><strong>{group.label}</strong><span className="mono">{group.keys.filter((key) => data.editable_config_keys.includes(key)).length} FIELDS</span></summary><div className="settings-fields">{group.keys.filter((key) => data.editable_config_keys.includes(key)).map((key) => { const schema = data.config_schema.properties[key] ?? {}; return <div className="setting-row" key={key}><header><label htmlFor={key}>{schema.title ?? key.replaceAll("_", " ")}</label><code>{key}</code></header><SettingControl name={key} schema={schema} value={values[key] ?? data.defaults[key]} onChange={(value) => setValue(key, value, { shouldDirty: true, shouldValidate: true })} />{schema.description && <small>{schema.description}</small>}</div>; })}</div></details>)}<div className="settings-actions"><button className="primary" type="submit"><Save size={15} /> 保存设置</button><button className="secondary" type="button" onClick={() => reset(data.defaults)}><RotateCcw size={15} /> 恢复默认</button><button className="secondary" type="button" onClick={exportJson}><Download size={15} /> 导出 JSON</button><button className="secondary" type="button" onClick={() => fileRef.current?.click()}><Upload size={15} /> 导入 JSON</button><input ref={fileRef} hidden type="file" accept="application/json" onChange={(event) => void importJson(event.target.files?.[0])} /></div></form>}</div></AppShell>;
}
