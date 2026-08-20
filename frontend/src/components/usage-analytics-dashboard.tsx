"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, BarChart3, ChevronLeft, ChevronRight, Filter, Search } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { researchApi } from "@/lib/api";

const colors = ["#42dcc7", "#78a9ff", "#f6bd60", "#a78bfa", "#ff7b72", "#9ad27b"];
const number = (value: number) => new Intl.NumberFormat("zh-CN", { notation: value > 9_999 ? "compact" : "standard", maximumFractionDigits: 1 }).format(value);
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const statusLabels: Record<string, string> = { running: "运行中", success: "已完成", completed: "已完成", error: "失败", failed: "失败", cancelled: "已取消", interrupted: "已中断" };

function DualTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name?: string; value?: number }>; label?: string }) {
  if (!active || !payload?.length) return null;
  const reported = payload.filter((item) => item.name?.includes("实报")).reduce((sum, item) => sum + Number(item.value || 0), 0);
  const estimated = payload.filter((item) => item.name?.includes("估算")).reduce((sum, item) => sum + Number(item.value || 0), 0);
  return <div className="usage-tooltip"><b>{label}</b><span><i className="reported-dot" />Provider 实报 <strong>{number(reported)}</strong></span><span><i className="estimated-dot" />估算补位 <strong>{number(estimated)}</strong></span></div>;
}

interface DistributionTrack { name: string; reported: number; estimated: number; total: number }

function DistributionTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload?: DistributionTrack }> }) {
  const item = payload?.find((entry) => entry.payload)?.payload;
  if (!active || !item) return null;
  return <div className="usage-tooltip"><b>{item.name}</b><span><i className="reported-dot" />Provider 实报 <strong>{number(item.reported)}</strong></span><span><i className="estimated-dot" />估算补位 <strong>{number(item.estimated)}</strong></span></div>;
}

export function UsageAnalyticsDashboard() {
  const [range, setRange] = useState<"7d" | "30d" | "retained">("30d");
  const [status, setStatus] = useState("");
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [search, setSearch] = useState("");
  const [cursor, setCursor] = useState<string | undefined>();
  const [cursorStack, setCursorStack] = useState<Array<string | undefined>>([]);
  const [distributionDimension, setDistributionDimension] = useState<"provider" | "model">("provider");
  const query = useQuery({ queryKey: ["usage-analytics", range, status, provider, model, search, cursor], queryFn: () => researchApi.usageAnalytics({ range, status, provider, model, query: search, cursor, timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "Asia/Shanghai" }) });
  const data = query.data;
  const topRuns = useMemo(() => [...(data?.runs ?? [])].sort((a, b) => (b.reported.total_tokens + b.estimated.total_tokens) - (a.reported.total_tokens + a.estimated.total_tokens)).slice(0, 8).map((run) => ({ name: run.title.length > 18 ? `${run.title.slice(0, 18)}…` : run.title, reported: run.reported.total_tokens, estimated: run.estimated.total_tokens })), [data]);
  const distributionData = useMemo<DistributionTrack[]>(() => {
    if (!data) return [];
    return data.distributions[distributionDimension].map((item) => ({ name: item.key, reported: item.reported_tokens, estimated: item.estimated_tokens, total: item.reported_tokens + item.estimated_tokens }));
  }, [data, distributionDimension]);

  function resetPage() { setCursor(undefined); setCursorStack([]); }
  function next() { if (!data?.next_cursor) return; setCursorStack((items) => [...items, cursor]); setCursor(data.next_cursor); }
  function previous() { const previousCursor = cursorStack.at(-1); setCursorStack((items) => items.slice(0, -1)); setCursor(previousCursor); }

  if (query.isLoading) return <div className="usage-empty"><Activity className="spin" /><h2>加载历史用量</h2><p>正在按当前用户和保留期聚合运行数据。</p></div>;
  if (query.isError || !data) return <div className="usage-empty"><BarChart3 /><h2>历史分析不可用</h2><p>无法读取聚合接口，请稍后重试。</p></div>;

  const retainedLabel = data.retention_days <= 0 ? "FOREVER" : `${data.retention_days} DAYS`;
  const totalDistribution = distributionData.reduce((sum, item) => sum + item.total, 0);
  return <div className="analytics-dashboard">
    <header className="analytics-hero"><div><span className="eyebrow mono">OWNER SCOPE / RETAINED {retainedLabel}</span><h1>用量分析</h1><p>跨运行比较 Token 核算、覆盖率、缓存、限流与吞吐。仅包含当前账号拥有的研究运行。</p></div><div className="analytics-range" role="group" aria-label="分析时间范围">{(["7d", "30d", "retained"] as const).map((value) => <button className={range === value ? "active" : ""} key={value} onClick={() => { setRange(value); resetPage(); }}>{value === "retained" ? "全部保留期" : value === "7d" ? "7 天" : "30 天"}</button>)}</div></header>

    <section className="analytics-filters" aria-label="用量筛选"><Filter size={15} /><select aria-label="运行状态" value={status} onChange={(event) => { setStatus(event.target.value); resetPage(); }}><option value="">全部状态</option><option value="running">运行中</option><option value="success">已完成</option><option value="error">失败</option><option value="cancelled">已取消</option></select><select aria-label="Provider" value={provider} onChange={(event) => { setProvider(event.target.value); resetPage(); }}><option value="">全部 Provider</option>{data.distributions.provider.map((item) => <option key={item.key} value={item.key}>{item.key}</option>)}</select><select aria-label="模型" value={model} onChange={(event) => { setModel(event.target.value); resetPage(); }}><option value="">全部模型</option>{data.distributions.model.map((item) => <option key={item.key} value={item.key}>{item.key}</option>)}</select><label><Search size={14} /><input value={search} onChange={(event) => { setSearch(event.target.value); resetPage(); }} placeholder="搜索标题或 Run ID" /></label></section>

    <div className="analytics-kpis"><div><span>Provider 实报</span><b>{number(data.summary.reported.total_tokens)}</b><small>{number(data.summary.reported.input_tokens)} IN · {number(data.summary.reported.output_tokens)} OUT</small></div><div className="estimated"><span>估算补位</span><b>{number(data.summary.estimated.total_tokens)}</b><small>与实报独立核算</small></div><div><span>调用覆盖率</span><b>{percent(data.summary.coverage_ratio)}</b><small>{data.summary.run_count} 个运行，加权口径</small></div><div><span>核算费用</span><b>{data.summary.estimated_cost_micro_usd === null ? "未配置" : `$${(data.summary.estimated_cost_micro_usd / 1_000_000).toFixed(2)}`}</b><small>冻结价格表估算</small></div></div>

    <div className="usage-chart-grid">
      <section className="usage-panel"><header><div><span className="eyebrow">DAILY LOAD</span><h2>日 Token 趋势</h2></div><small>实际范围 {data.actual_range_days.toFixed(0)} 天</small></header><div className="chart-frame"><ResponsiveContainer width="100%" height="100%"><BarChart data={data.daily} accessibilityLayer><defs><pattern id="analytics-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line y2="7" stroke="#f6bd60" strokeWidth="3" /></pattern></defs><CartesianGrid stroke="#243139" vertical={false} /><XAxis dataKey="date" tick={{ fill: "#71828b", fontSize: 10 }} tickFormatter={(value) => String(value).slice(5)} axisLine={false} tickLine={false} /><YAxis tickFormatter={number} tick={{ fill: "#71828b", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip content={<DualTooltip />} /><Legend /><Bar dataKey="reported_tokens" name="Provider 实报" stackId="daily" fill="#42dcc7" /><Bar dataKey="estimated_tokens" name="估算补位" stackId="daily" fill="url(#analytics-hatch)" /></BarChart></ResponsiveContainer></div><details className="chart-data"><summary>查看每日数据</summary><table className="usage-table"><thead><tr><th>日期</th><th>实报</th><th>估算</th><th>覆盖率</th></tr></thead><tbody>{data.daily.map((day) => <tr key={day.date}><td>{day.date}</td><td>{number(day.reported_tokens)}</td><td className="estimated-cell">{number(day.estimated_tokens)}</td><td>{percent(day.coverage_ratio)}</td></tr>)}</tbody></table></details></section>

      <section className="usage-panel"><header><div><span className="eyebrow">LOAD MIX</span><h2>{distributionDimension === "provider" ? "Provider" : "模型"}占比</h2></div><div className="chart-switch" role="group" aria-label="切换占比维度"><button className={distributionDimension === "provider" ? "active" : ""} onClick={() => setDistributionDimension("provider")}>Provider</button><button className={distributionDimension === "model" ? "active" : ""} onClick={() => setDistributionDimension("model")}>模型</button></div></header><div className="chart-frame donut-frame"><ResponsiveContainer width="100%" height="100%"><PieChart accessibilityLayer><defs>{distributionData.map((item, index) => <pattern id={`analytics-donut-hatch-${index}`} width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)" key={item.name}><rect width="7" height="7" fill="#172027" /><line y2="7" stroke={colors[index % colors.length]} strokeWidth="2" /></pattern>)}</defs><Pie data={distributionData} dataKey="reported" nameKey="name" innerRadius="42%" outerRadius="62%">{distributionData.map((item, index) => <Cell key={`reported-${item.name}`} fill={colors[index % colors.length]} />)}</Pie><Pie data={distributionData} dataKey="estimated" nameKey="name" innerRadius="67%" outerRadius="82%">{distributionData.map((item, index) => <Cell key={`estimated-${item.name}`} fill={`url(#analytics-donut-hatch-${index})`} />)}</Pie><Tooltip content={<DistributionTooltip />} /></PieChart></ResponsiveContainer><div className="donut-total"><b>{number(totalDistribution)}</b><span>核算总量</span></div></div><ul className="usage-donut-legend" aria-label="历史用量占比图例">{distributionData.map((item, index) => <li key={item.name} tabIndex={0}><i style={{ backgroundColor: colors[index % colors.length] }} /><span>{item.name}</span><b>{number(item.total)}</b></li>)}</ul><details className="chart-data"><summary>查看占比数据</summary><table className="usage-table"><thead><tr><th>维度</th><th>Provider 实报</th><th>估算补位</th></tr></thead><tbody>{distributionData.map((item) => <tr key={item.name}><td>{item.name}</td><td>{number(item.reported)}</td><td className="estimated-cell">{number(item.estimated)}</td></tr>)}</tbody></table></details></section>
    </div>

    <div className="usage-chart-grid">
      <section className="usage-panel"><header><div><span className="eyebrow">TOP RUNS</span><h2>高用量运行</h2></div><small>当前结果页 Top 8</small></header><div className="chart-frame"><ResponsiveContainer width="100%" height="100%"><BarChart data={topRuns} layout="vertical" accessibilityLayer><defs><pattern id="top-runs-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line y2="7" stroke="#f6bd60" strokeWidth="3" /></pattern></defs><CartesianGrid stroke="#243139" horizontal={false} /><XAxis type="number" tickFormatter={number} tick={{ fill: "#71828b", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis dataKey="name" type="category" width={130} tick={{ fill: "#aebbc1", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip content={<DualTooltip />} /><Legend /><Bar dataKey="reported" name="Provider 实报" stackId="runs" fill="#42dcc7" /><Bar dataKey="estimated" name="估算补位" stackId="runs" fill="url(#top-runs-hatch)" /></BarChart></ResponsiveContainer></div><p className="chart-summary">当前页最高用量运行：{topRuns[0]?.name ?? "暂无数据"}。</p></section>

      <section className="usage-panel"><header><div><span className="eyebrow">QUALITY / OPERATIONS</span><h2>覆盖率与运行信号</h2></div><small>日均趋势</small></header><div className="chart-frame"><ResponsiveContainer width="100%" height="100%"><LineChart data={data.daily} accessibilityLayer><CartesianGrid stroke="#243139" vertical={false} /><XAxis dataKey="date" tickFormatter={(value) => String(value).slice(5)} tick={{ fill: "#71828b", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis yAxisId="ratio" domain={[0, 1]} tickFormatter={percent} tick={{ fill: "#71828b", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis yAxisId="speed" orientation="right" tick={{ fill: "#71828b", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip /><Legend /><Line yAxisId="ratio" dataKey="coverage_ratio" name="覆盖率" stroke="#42dcc7" strokeWidth={2} /><Line yAxisId="ratio" dataKey="cache_hit_rate" name="缓存命中" stroke="#f6bd60" strokeWidth={2} /><Line yAxisId="ratio" dataKey="rate_429" name="429 比率" stroke="#ff7b72" strokeWidth={2} /><Line yAxisId="speed" dataKey="output_tokens_per_second" name="输出 tok/s" stroke="#78a9ff" strokeWidth={2} /></LineChart></ResponsiveContainer></div><p className="chart-summary">同时展示覆盖率、缓存命中、429 与输出吞吐；比例与速度使用独立纵轴。</p></section>
    </div>

    <section className="usage-panel analytics-runs"><header><div><span className="eyebrow">RUN LEDGER</span><h2>运行明细</h2></div><small>点击进入该运行用量视图</small></header><div className="usage-table-wrap"><table className="usage-table"><thead><tr><th>运行</th><th>状态</th><th>实报</th><th>估算</th><th>覆盖率</th><th>费用</th><th>429</th><th>吞吐</th></tr></thead><tbody>{data.runs.map((run) => <tr key={run.run_id}><td><Link href={`/research/${encodeURIComponent(run.run_id)}?view=usage`}><b>{run.title}</b><small>{run.run_id}</small></Link></td><td><span className="mini-status" data-status={run.accounting_status}>{statusLabels[run.status] ?? run.status}</span></td><td>{number(run.reported.total_tokens)}</td><td className="estimated-cell">{number(run.estimated.total_tokens)}</td><td>{percent(run.calls.coverage_ratio)}</td><td>{run.cost.estimated_cost_micro_usd === null ? "未配置" : `$${(run.cost.estimated_cost_micro_usd / 1_000_000).toFixed(3)}`}</td><td>{percent(run.operations.rate_429)}</td><td>{run.operations.output_tokens_per_second.toFixed(1)} tok/s</td></tr>)}{!data.runs.length && <tr><td colSpan={8} className="empty-cell">当前筛选条件下没有保留的核算记录。</td></tr>}</tbody></table></div><footer className="table-pagination"><button disabled={!cursorStack.length} onClick={previous}><ChevronLeft size={15} />上一页</button><span>{data.runs.length} 条 / 页</span><button disabled={!data.next_cursor} onClick={next}>下一页<ChevronRight size={15} /></button></footer></section>
  </div>;
}
