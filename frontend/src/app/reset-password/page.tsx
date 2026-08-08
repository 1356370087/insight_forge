"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import { AuthShell } from "@/components/auth-shell";
import { authFetch } from "@/lib/auth";

export default function ResetPasswordPage() {
  const token = useSearchParams().get("token") ?? ""; const [done, setDone] = useState(false); const [error, setError] = useState("");
  async function submit(form: FormData) { const password = String(form.get("password")); if (password !== form.get("confirm")) { setError("两次输入的密码不一致"); return; } try { await authFetch("/api/auth/reset-password", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token, password }) }); setDone(true); } catch (cause) { setError(cause instanceof Error ? cause.message : "重置失败"); } }
  return <AuthShell eyebrow="NEW CREDENTIAL" title="设置新密码" copy="设置成功会撤销该账户的全部现有会话。" footer={<Link href="/login">返回登录</Link>}>
    {done ? <div className="auth-success"><b>密码已更新</b><Link className="primary" href="/login">重新登录</Link></div> : <form action={submit} className="auth-form"><div className="field"><label htmlFor="password">新密码</label><input id="password" name="password" type="password" minLength={15} maxLength={128} required autoComplete="new-password" /></div><div className="field"><label htmlFor="confirm">确认新密码</label><input id="confirm" name="confirm" type="password" minLength={15} maxLength={128} required autoComplete="new-password" /></div>{error && <p className="form-alert error">{error}</p>}<button className="primary auth-submit" disabled={!token}>更新密码</button></form>}
  </AuthShell>;
}
