"use client";

import { useQuery } from "@tanstack/react-query";
import { ChevronRight, CircleDot, FlaskConical, Languages, Menu, PanelRight, Plus, Settings, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";
import { researchApi } from "@/lib/api";
import { localAuthBypass } from "@/lib/auth";

const statusLabel: Record<string, string> = { pending: "排队", running: "运行中", awaiting_clarification: "待澄清", awaiting_plan_approval: "待审批", awaiting_outline_approval: "待审批", completed: "已完成", failed: "失败", cancelled: "已取消" };

export function AppShell({ children, inspector }: { children: ReactNode; inspector?: ReactNode }) {
  const pathname = usePathname();
  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(false);
  const { data } = useQuery({ queryKey: ["runs"], queryFn: () => researchApi.listRuns() });
  const switchLocale = () => { const next = document.documentElement.lang === "en" ? "zh-CN" : "en"; document.cookie = `odr.locale=${next};path=/;max-age=31536000;samesite=lax`; location.reload(); };
  const sidebar = <>
    <div className="brand-lockup"><div className="brand-mark"><FlaskConical size={18} /></div><div><strong>OPEN DEEP</strong><span>RESEARCH / CONSOLE</span></div></div>
    {localAuthBypass && <div className="dev-badge"><CircleDot size={12} /> LOCAL DEV · AUTH BYPASS</div>}
    <Link className="new-run-button" href="/research/new"><Plus size={17} /> 新建研究</Link>
    <div className="sidebar-section-title"><span>运行档案</span><span>{data?.items.length ?? 0}</span></div>
    <nav className="run-list" aria-label="研究历史">
      {data?.items.map((item) => { const id = String(item.run_id); const active = pathname.endsWith(id); return <Link key={id} className={`run-item ${active ? "active" : ""}`} href={`/research/${id}`}><i data-status={String(item.status)} /><span><b>{String(item.title ?? id)}</b><small>{statusLabel[String(item.status)] ?? String(item.status)}</small></span><ChevronRight size={14} /></Link>; })}
      {!data?.items.length && <p className="empty-note">还没有研究记录。发起第一个问题，事件轨道会在这里留下可回放档案。</p>}
    </nav>
    <div className="sidebar-footer"><Link href="/settings"><Settings size={16} /> 配置中心</Link><button onClick={switchLocale}><Languages size={16} /> 中 / EN</button></div>
  </>;
  return <div className="command-shell">
    <aside className={`left-rail ${leftOpen ? "drawer-open" : ""}`}>{sidebar}<button className="drawer-close" onClick={() => setLeftOpen(false)} aria-label="关闭导航"><X /></button></aside>
    <main className="workbench"><header className="mobile-bar"><button onClick={() => setLeftOpen(true)} aria-label="打开导航"><Menu /></button><span>ODR / COMMAND</span>{inspector ? <button onClick={() => setRightOpen(true)} aria-label="打开研究信息"><PanelRight /></button> : <i />}</header>{children}</main>
    {inspector && <aside className={`right-rail ${rightOpen ? "drawer-open" : ""}`}>{inspector}<button className="drawer-close" onClick={() => setRightOpen(false)} aria-label="关闭信息栏"><X /></button></aside>}
    {(leftOpen || rightOpen) && <button className="drawer-scrim" onClick={() => { setLeftOpen(false); setRightOpen(false); }} aria-label="关闭抽屉" />}
  </div>;
}
