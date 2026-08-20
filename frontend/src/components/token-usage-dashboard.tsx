"use client";

import { AlertTriangle, Coins, Gauge, LoaderCircle, Radio, ShieldCheck, TimerReset, Zap } from "lucide-react";
import { useMemo, useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useRunUsage } from "@/hooks/use-run-usage";
import type { RunUsageResponse, UsageBucket } from "@/lib/types";

const palette = ["#42dcc7", "#78a9ff", "#f6bd60", "#a78bfa", "#ff7b72", "#9ad27b"];
const stageLabels: Record<string, string> = { preparing: "准备", planning: "规划", researching: "研究", synthesizing: "汇总", writing: "撰写", finalizing: "完成", unknown: "未知" };

function tokens(value: number) {
  return new Intl.NumberFormat("zh-CN", { notation: value >= 10_000 ? "compact" : "standard", maximumFractionDigits: 1 }).format(value);
}

function percent(value: number) { return `${(value * 100).toFixed(value >= 0.1 ? 1 : 2)}%`; }
function cost(value: number | null) { return value === null ? "未配置" : `$${(value / 1_000_000).toFixed(value < 10_000 ? 4 : 2)}`; }
function elapsed(value: number | null) {
  if (value === null) return "运行中";
  const seconds = Math.max(0, Math.round(value / 1000));
  if (seconds < 60) return `${seconds} 秒`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} 分 ${seconds % 60} 秒`;
}

function AccountingBadge({ status }: { status: RunUsageResponse["accounting_status"] }) {
  const label = status === "complete" ? "实报完整" : status === "partial" ? "部分核算" : "核算不可用";
  return <span className="usage-status" data-status={status}>{status === "complete" ? <ShieldCheck size={13} /> : <AlertTriangle size={13} />}{label}</span>;
}

function EmptyUsage({ status }: { status: "loading" | "error" | "unavailable" | "empty" }) {
  const copy = {
    loading: ["读取核算记录", "正在同步当前运行的最小化 Token 事件。"],
    error: ["无法读取用量", "接口暂时不可用；研究运行本身不受影响。"],
    unavailable: ["Token 核算不可用", "该运行关闭了核算，或持久化存储当前不可用。"],
    empty: ["尚无模型调用", "首个模型响应完成后，实报或估算补位会显示在这里。"],
  }[status];
  return <div className="usage-empty">{status === "loading" ? <LoaderCircle className="spin" /> : <Gauge />}<h2>{copy[0]}</h2><p>{copy[1]}</p></div>;
}

function DualTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name?: string; value?: number; color?: string }>; label?: string }) {
  if (!active || !payload?.length) return null;
  const reported = payload.filter((item) => item.name?.includes("实报")).reduce((sum, item) => sum + Number(item.value || 0), 0);
  const estimated = payload.filter((item) => item.name?.includes("估算")).reduce((sum, item) => sum + Number(item.value || 0), 0);
  return <div className="usage-tooltip"><b>{label}</b><span><i className="reported-dot" />Provider 实报 <strong>{tokens(reported)}</strong></span><span><i className="estimated-dot" />估算补位 <strong>{tokens(estimated)}</strong></span></div>;
}

function UsageKpis({ data }: { data: RunUsageResponse }) {
  const { reported, estimated, calls, cost: usageCost } = data.totals;
  const items = [
    { label: "Provider 实报", value: tokens(reported.total_tokens), detail: `${tokens(reported.input_tokens)} IN / ${tokens(reported.output_tokens)} OUT`, icon: Radio, tone: "reported" },
    { label: "估算补位", value: tokens(estimated.total_tokens), detail: `${tokens(estimated.input_tokens)} IN / ${tokens(estimated.output_tokens)} OUT`, icon: TimerReset, tone: "estimated" },
    { label: "调用覆盖率", value: percent(calls.coverage_ratio), detail: `${calls.provider_reported} / ${calls.successful_responses} 成功响应`, icon: ShieldCheck, tone: calls.coverage_ratio === 1 ? "reported" : "warning" },
    { label: "核算费用", value: cost(usageCost.estimated_cost_micro_usd), detail: usageCost.cost_source === "configured_estimate" ? "冻结价格表估算" : usageCost.cost_source === "provider_reported" ? "Provider 实报" : "未配置价格", icon: Coins, tone: "neutral" },
    { label: "模型尝试", value: tokens(calls.attempts), detail: `${calls.unknown_failed_attempts} 次未知失败`, icon: Zap, tone: calls.unknown_failed_attempts ? "danger" : "neutral" },
    { label: "运行耗时", value: elapsed(data.duration_ms), detail: data.status === "running" ? "运行结束后冻结" : "服务端运行计时", icon: Gauge, tone: "neutral" },
  ];
  return <div className="usage-kpis">{items.map(({ label, value, detail, icon: Icon, tone }) => <div className="usage-kpi" data-tone={tone} key={label}><div><span>{label}</span><Icon size={16} /></div><strong>{value}</strong><small>{detail}</small></div>)}</div>;
}

function BudgetTracks({ data }: { data: RunUsageResponse }) {
  return <section className="usage-panel budget-panel"><header><div><span className="eyebrow">BUDGET / SETTLEMENT</span><h2>预算轨道</h2></div><small>已结算 · 估算补位 · 未决预留</small></header><div className="budget-tracks">{Object.entries(data.totals.budgets).map(([key, item]) => {
    const settled = item.settled ?? 0;
    const total = settled + item.estimated + item.reserved;
    const denominator = item.limit || Math.max(1, total);
    const labels: Record<string, string> = { input_tokens: "输入 Token", output_tokens: "输出 Token", model_calls: "模型调用", cost_micro_usd: "费用" };
    const formatValue = (value: number | null) => key === "cost_micro_usd" ? cost(value) : tokens(value ?? 0);
    return <div className="budget-row" key={key}><div><b>{labels[key] ?? key}</b><span>{formatValue(item.settled)}{item.limit ? ` / ${formatValue(item.limit)}` : " / 未设上限"}</span></div><div className="budget-bar" aria-label={`${labels[key]}已使用 ${formatValue(total)}`}><i className="budget-settled" style={{ width: `${Math.min(100, settled / denominator * 100)}%` }} /><i className="budget-estimated" style={{ width: `${Math.min(100, item.estimated / denominator * 100)}%` }} /><i className="budget-reserved" style={{ width: `${Math.min(100, item.reserved / denominator * 100)}%` }} /></div></div>;
  })}</div></section>;
}

interface PieTrack {
  name: string;
  reported: number;
  estimated: number;
  total: number;
}

function DonutTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload?: PieTrack }> }) {
  const item = payload?.find((entry) => entry.payload)?.payload;
  if (!active || !item) return null;
  return <div className="usage-tooltip"><b>{item.name}</b><span><i className="reported-dot" />Provider 实报 <strong>{tokens(item.reported)}</strong></span><span><i className="estimated-dot" />估算补位 <strong>{tokens(item.estimated)}</strong></span></div>;
}

function BreakdownCharts({ data }: { data: RunUsageResponse }) {
  const [dimension, setDimension] = useState<"role" | "provider" | "model">("role");
  const stageData = data.breakdowns.by_stage.map((bucket) => ({
    name: stageLabels[bucket.key] ?? bucket.label,
    reportedInput: bucket.reported.input_tokens,
    reportedOutput: bucket.reported.output_tokens,
    estimatedInput: bucket.estimated.input_tokens,
    estimatedOutput: bucket.estimated.output_tokens,
  }));
  const pieData = useMemo(() => {
    const source = dimension === "role" ? data.breakdowns.by_agent_role : data.breakdowns.by_model;
    const grouped = new Map<string, { reported: number; estimated: number }>();
    source.forEach((bucket) => {
      const key = dimension === "provider"
        ? bucket.key.split(":", 1)[0] || "unknown"
        : dimension === "model"
          ? bucket.key.split(":").slice(1).join(":") || bucket.key
          : bucket.key;
      const previous = grouped.get(key) ?? { reported: 0, estimated: 0 };
      grouped.set(key, {
        reported: previous.reported + bucket.reported.total_tokens,
        estimated: previous.estimated + bucket.estimated.total_tokens,
      });
    });
    return [...grouped].map(([name, value]) => ({
      name,
      ...value,
      total: value.reported + value.estimated,
    })).sort((a, b) => b.total - a.total);
  }, [data, dimension]);
  const total = pieData.reduce((sum, item) => sum + item.total, 0);
  return <div className="usage-chart-grid"><section className="usage-panel"><header><div><span className="eyebrow">STAGE VECTOR</span><h2>阶段输入 / 输出</h2></div><small>六阶段堆叠柱</small></header><div className="chart-frame"><ResponsiveContainer width="100%" height="100%"><BarChart data={stageData} accessibilityLayer><defs><pattern id="usage-estimated-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="7" height="7" fill="#4a3b22" /><line x1="0" y1="0" x2="0" y2="7" stroke="#f6bd60" strokeWidth="2" /></pattern></defs><CartesianGrid stroke="#243139" vertical={false} /><XAxis dataKey="name" tick={{ fill: "#91a2aa", fontSize: 11 }} axisLine={false} tickLine={false} /><YAxis tickFormatter={tokens} tick={{ fill: "#71828b", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip content={<DualTooltip />} /><Legend /><Bar dataKey="reportedInput" name="实报 · 输入" stackId="usage" fill="#42dcc7" radius={[2, 2, 0, 0]} /><Bar dataKey="reportedOutput" name="实报 · 输出" stackId="usage" fill="#78a9ff" /><Bar dataKey="estimatedInput" name="估算 · 输入" stackId="usage" fill="url(#usage-estimated-hatch)" /><Bar dataKey="estimatedOutput" name="估算 · 输出" stackId="usage" fill="url(#usage-estimated-hatch)" /></BarChart></ResponsiveContainer></div><details className="chart-data"><summary>查看阶段数据表</summary><BucketTable buckets={data.breakdowns.by_stage} /></details></section><section className="usage-panel"><header><div><span className="eyebrow">SHARE OF LOAD</span><h2>用量构成</h2></div><div className="chart-switch" role="group" aria-label="切换占比维度">{(["role", "provider", "model"] as const).map((key) => <button key={key} className={dimension === key ? "active" : ""} onClick={() => setDimension(key)}>{key === "role" ? "角色" : key === "provider" ? "Provider" : "模型"}</button>)}</div></header><div className="chart-frame donut-frame"><ResponsiveContainer width="100%" height="100%"><PieChart accessibilityLayer><defs>{pieData.map((entry, index) => <pattern id={`usage-donut-hatch-${index}`} width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)" key={entry.name}><rect width="7" height="7" fill="#172027" /><line y2="7" stroke={palette[index % palette.length]} strokeWidth="2" /></pattern>)}</defs><Pie data={pieData} dataKey="reported" nameKey="name" innerRadius="42%" outerRadius="62%" paddingAngle={2}>{pieData.map((entry, index) => <Cell key={`reported-${entry.name}`} fill={palette[index % palette.length]} />)}</Pie><Pie data={pieData} dataKey="estimated" nameKey="name" innerRadius="67%" outerRadius="82%" paddingAngle={2}>{pieData.map((entry, index) => <Cell key={`estimated-${entry.name}`} fill={`url(#usage-donut-hatch-${index})`} />)}</Pie><Tooltip content={<DonutTooltip />} /></PieChart></ResponsiveContainer><div className="donut-total"><b>{tokens(total)}</b><span>核算总量</span></div></div><ul className="usage-donut-legend" aria-label="用量构成图例">{pieData.map((entry, index) => <li key={entry.name} tabIndex={0}><i style={{ backgroundColor: palette[index % palette.length] }} /><span>{entry.name}</span><b>{tokens(entry.total)}</b></li>)}</ul><details className="chart-data"><summary>查看占比数据表</summary><table className="usage-table"><thead><tr><th>维度</th><th>Provider 实报</th><th>估算补位</th></tr></thead><tbody>{pieData.map((entry) => <tr key={entry.name}><td>{entry.name}</td><td>{tokens(entry.reported)}</td><td className="estimated-cell">{tokens(entry.estimated)}</td></tr>)}</tbody></table></details><p className="chart-summary">最大负载：{pieData[0]?.name ?? "暂无"}，占核算总量 {pieData.length ? percent(pieData[0].total / Math.max(1, total)) : "0%"}。</p></section></div>;
}

function TimelineChart({ data }: { data: RunUsageResponse }) {
  const timeline = data.timeline.map((item, index) => ({ ...item, label: `T+${index + 1}` }));
  return <section className="usage-panel timeline-panel"><header><div><span className="eyebrow">TOKEN FLOW / LIVE</span><h2>Token 时间图</h2></div><small>{data.timeline.length} 个服务端聚合桶 · 最多 120</small></header><div className="chart-frame timeline-frame"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={timeline} accessibilityLayer><defs><pattern id="timeline-estimated-hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line y2="8" stroke="#f6bd60" strokeWidth="3" /></pattern></defs><CartesianGrid stroke="#243139" vertical={false} /><XAxis dataKey="label" minTickGap={28} tick={{ fill: "#71828b", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis tickFormatter={tokens} tick={{ fill: "#71828b", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip content={<DualTooltip />} /><Area type="monotone" dataKey="reported_cumulative" name="实报 · 累计" stroke="#42dcc7" fill="#42dcc72b" strokeWidth={2} /><Line type="monotone" dataKey="estimated_cumulative" name="估算 · 累计" stroke="#f6bd60" strokeWidth={2} strokeDasharray="7 5" dot={false} /><Bar dataKey="call_count" name="调用事件" fill="url(#timeline-estimated-hatch)" opacity={0.38} /><Bar dataKey="retry_count" name="重试事件" fill="#ff7b72" maxBarSize={5} /></ComposedChart></ResponsiveContainer></div><p className="chart-summary">最后一个桶累计：实报 {tokens(timeline.at(-1)?.reported_cumulative ?? 0)}，估算补位 {tokens(timeline.at(-1)?.estimated_cumulative ?? 0)}；重试事件 {tokens(timeline.reduce((sum, item) => sum + item.retry_count, 0))} 次。</p></section>;
}

function Efficiency({ data }: { data: RunUsageResponse }) {
  const items = [
    ["缓存命中", percent(data.operations.cache_hit_rate), `缓存输入占比 ${percent(data.operations.cache_input_ratio)}`],
    ["推理占比", percent(data.operations.reasoning_output_ratio), "仅 Provider 可识别维度"],
    ["输出吞吐", `${data.operations.output_tokens_per_second.toFixed(1)} tok/s`, `${data.operations.llm_call_count} 次 LLM 调用`],
    ["429 比率", percent(data.operations.rate_429), `${data.operations.rate_limited_count} 次限流调用`],
    ["重试", tokens(data.operations.retry_count), "每次物理尝试独立核算"],
    ["工具成功率", percent(data.operations.tool_success_rate), `${data.operations.tool_call_count} 次工具调用`],
  ];
  return <section className="usage-panel efficiency-panel"><header><div><span className="eyebrow">EFFICIENCY SIGNALS</span><h2>运行效能</h2></div><Gauge size={17} /></header><div className="efficiency-grid">{items.map(([label, value, detail]) => <div key={label}><span>{label}</span><b>{value}</b><small>{detail}</small></div>)}</div>{(data.operations.empty_tool_result_count > 0 || data.operations.zero_source_search_count > 0) && <div className="usage-warning"><AlertTriangle size={14} />空工具结果 {data.operations.empty_tool_result_count} 次 · 零来源搜索 {data.operations.zero_source_search_count} 次</div>}</section>;
}

function BucketTable({ buckets }: { buckets: UsageBucket[] }) {
  return <div className="usage-table-wrap"><table className="usage-table"><thead><tr><th>维度</th><th>实报</th><th>估算</th><th>费用</th><th>调用</th><th>平均延迟</th><th>完整度</th></tr></thead><tbody>{buckets.map((bucket) => <tr key={bucket.key}><td><b>{bucket.label}</b><small>{bucket.key}</small></td><td>{tokens(bucket.reported.total_tokens)}</td><td className="estimated-cell">{tokens(bucket.estimated.total_tokens)}</td><td>{cost(bucket.estimated_cost_micro_usd)}</td><td>{bucket.call_count}</td><td>{bucket.average_latency_ms ? `${bucket.average_latency_ms} ms` : "—"}</td><td><span className="mini-status" data-status={bucket.completeness}>{bucket.completeness === "complete" ? "完整" : "部分"}</span></td></tr>)}{!buckets.length && <tr><td colSpan={7} className="empty-cell">暂无明细记录</td></tr>}</tbody></table></div>;
}

export function TokenUsageDashboard({ runId, visible, terminal }: { runId: string; visible: boolean; terminal: boolean }) {
  const query = useRunUsage(runId, visible, terminal);
  if (query.isLoading) return <EmptyUsage status="loading" />;
  if (query.isError || !query.data) return <EmptyUsage status="error" />;
  if (query.data.accounting_status === "unavailable" && query.data.unavailable_reason === "no_usage_events") return <EmptyUsage status="empty" />;
  if (query.data.accounting_status === "unavailable") return <EmptyUsage status="unavailable" />;
  if (!query.data.totals.calls.attempts) return <EmptyUsage status="empty" />;
  const data = query.data;
  return <div className="usage-dashboard"><div className="usage-heading"><div><span className="eyebrow mono">USAGE ACCOUNTING / REV {data.revision}</span><h2>研究用量监控</h2><p>实报与估算双轨展示；核算总量用于预算保护，财务对账以 Provider 账单为准。</p></div><AccountingBadge status={data.accounting_status} /></div><UsageKpis data={data} /><BudgetTracks data={data} /><BreakdownCharts data={data} /><TimelineChart data={data} /><Efficiency data={data} /><section className="usage-panel"><header><div><span className="eyebrow">CALL LEDGER</span><h2>调用明细</h2></div><small>阶段 · 角色 · 模型 · 任务</small></header><div className="detail-tabs"><details open><summary>按模型</summary><BucketTable buckets={data.breakdowns.by_model} /></details><details><summary>按任务</summary><BucketTable buckets={data.breakdowns.by_task} /></details><details><summary>按角色</summary><BucketTable buckets={data.breakdowns.by_agent_role} /></details></div></section>{data.totals.calls.legacy_unclassified > 0 && <div className="usage-legacy"><AlertTriangle size={15} /><span><b>历史记录，完整度未知</b> 旧版非零 usage 已纳入实报兼容汇总，但不伪装为完整 Provider 数据。</span></div>}</div>;
}

export function UsageCompactSummary({ runId, terminal }: { runId: string; terminal: boolean }) {
  const { data } = useRunUsage(runId, true, terminal);
  if (!data || data.accounting_status === "unavailable") return <div className="inspector-block"><p className="eyebrow">TOKEN / ACCOUNTING</p><p className="empty-note">等待核算数据。</p></div>;
  return <div className="inspector-block"><div className="usage-compact-head"><p className="eyebrow">TOKEN / ACCOUNTING</p><AccountingBadge status={data.accounting_status} /></div><div className="usage-compact"><div><span>实报</span><b>{tokens(data.totals.reported.total_tokens)}</b></div><div className="estimated"><span>估算</span><b>{tokens(data.totals.estimated.total_tokens)}</b></div><div><span>覆盖</span><b>{percent(data.totals.calls.coverage_ratio)}</b></div></div></div>;
}
