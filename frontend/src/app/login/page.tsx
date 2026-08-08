"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { AuthShell } from "@/components/auth-shell";
import { authFetch, iamApi, localAuthBypass } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function submit(form: FormData) {
    if (localAuthBypass) { router.replace("/research/new"); return; }
    setBusy(true); setError("");
    try {
      await authFetch("/api/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email: form.get("email"), password: form.get("password") }) });
      const user = await iamApi.me();
      router.replace(user.status === "pending_approval" ? "/pending" : "/research/new");
      router.refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : "登录失败"); }
    finally { setBusy(false); }
  }
  return <AuthShell eyebrow="AUTHORIZED PERSONNEL" title="研究员身份验证" copy="使用由本项目签发的身份凭据进入研究工作台。访问令牌只保存在安全 Cookie 中。" footer={<><span>尚未注册？</span><Link href="/register">申请研究席位</Link></>}>
    <form action={submit} className="auth-form">
      <div className="field"><label htmlFor="email">工作邮箱</label><input id="email" name="email" type="email" required autoComplete="email" /></div>
      <div className="field"><label htmlFor="password">密码</label><input id="password" name="password" type="password" required autoComplete="current-password" /></div>
      <div className="auth-form-row"><Link href="/forgot-password">忘记密码</Link><span className="mono">ACCESS / 15 MIN</span></div>
      {error && <p className="form-alert error" role="alert">{error}</p>}
      <button className="primary auth-submit" disabled={busy}>{busy ? "正在验证…" : "进入研究台"}</button>
      {localAuthBypass && <p className="dev-badge">LOCAL DEV 已启用，可直接进入</p>}
    </form>
  </AuthShell>;
}
