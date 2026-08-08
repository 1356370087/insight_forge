"use client";

import Link from "next/link";
import { useState } from "react";
import { AuthShell } from "@/components/auth-shell";
import { authFetch } from "@/lib/auth";

export default function ForgotPasswordPage() {
  const [sent, setSent] = useState(false); const [error, setError] = useState("");
  async function submit(form: FormData) {
    try { await authFetch("/api/auth/forgot-password", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email: form.get("email") }) }); setSent(true); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "请求失败"); }
  }
  return <AuthShell eyebrow="CREDENTIAL RECOVERY" title="重置访问凭据" copy="若邮箱对应有效账户，系统会发送一条 30 分钟内有效的重置链接。" footer={<Link href="/login">返回登录</Link>}>
    {sent ? <div className="auth-success"><b>检查你的收件箱</b><p>为避免泄露账户是否存在，本页面不会显示匹配结果。</p></div> : <form action={submit} className="auth-form"><div className="field"><label htmlFor="email">工作邮箱</label><input id="email" name="email" type="email" required autoComplete="email" /></div>{error && <p className="form-alert error">{error}</p>}<button className="primary auth-submit">发送重置链接</button></form>}
  </AuthShell>;
}
