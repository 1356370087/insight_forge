"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { KeyRound, MonitorSmartphone, ShieldCheck, Trash2 } from "lucide-react";
import { useState } from "react";
import { AppShell } from "@/components/app-shell";
import { iamApi } from "@/lib/auth";

export default function AccountSecurityPage() {
  const client = useQueryClient(); const [message, setMessage] = useState("");
  const identity = useQuery({ queryKey: ["identity"], queryFn: iamApi.me });
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: iamApi.sessions });
  const revoke = useMutation({ mutationFn: iamApi.revokeSession, onSuccess: () => client.invalidateQueries({ queryKey: ["sessions"] }) });
  async function change(form: FormData) { setMessage(""); try { await iamApi.changePassword(String(form.get("current_password")), String(form.get("new_password"))); setMessage("密码已更新，全部会话已撤销。请重新登录。"); } catch (cause) { setMessage(cause instanceof Error ? cause.message : "更新失败"); } }
  return <AppShell><div className="page account-page"><header className="page-header"><div><span className="eyebrow">IDENTITY / SECURITY</span><h1>账户与会话</h1><p>检查有效设备、撤销会话，并更新你的登录凭据。</p></div><ShieldCheck size={34} color="var(--cyan)" /></header><section className="account-grid"><article className="panel"><div className="panel-header"><h2>身份档案</h2></div><div className="panel-body identity-profile"><b>{identity.data?.display_name || "未命名研究员"}</b><span>{identity.data?.email}</span><div>{identity.data?.roles.map((role) => <i key={role}>{role}</i>)}</div></div></article><article className="panel"><div className="panel-header"><h2><KeyRound size={15} /> 更新密码</h2></div><form action={change} className="panel-body auth-form"><div className="field"><label htmlFor="current_password">当前密码</label><input id="current_password" name="current_password" type="password" required autoComplete="current-password" /></div><div className="field"><label htmlFor="new_password">新密码</label><input id="new_password" name="new_password" type="password" minLength={15} maxLength={128} required autoComplete="new-password" /></div>{message && <p className="form-alert">{message}</p>}<button className="primary">更新并撤销会话</button></form></article></section><section className="panel session-panel"><div className="panel-header"><h2><MonitorSmartphone size={15} /> 设备会话</h2><button className="secondary" onClick={() => iamApi.logoutAll().then(() => sessions.refetch())}>撤销其他会话</button></div><div className="session-list">{sessions.data?.map((item) => <article key={String(item.id)}><div><b>{String(item.user_agent || "未知客户端")}</b><span className="mono">{String(item.ip_address || "未知地址")} · {new Date(String(item.last_activity_at)).toLocaleString()}</span></div><span className={`session-state ${item.is_current ? "current" : ""}`}>{item.is_revoked ? "已撤销" : item.is_current ? "当前会话" : "有效"}</span>{!item.is_current && !item.is_revoked && <button onClick={() => revoke.mutate(String(item.id))} aria-label="撤销会话"><Trash2 size={15} /></button>}</article>)}</div></section></div></AppShell>;
}
