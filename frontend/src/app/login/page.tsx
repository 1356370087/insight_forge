"use client";

import { FlaskConical } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { getSupabaseBrowserClient, localAuthBypass } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter(); const [error, setError] = useState("");
  async function submit(form: FormData) {
    if (localAuthBypass) { router.replace("/research/new"); return; }
    const client = getSupabaseBrowserClient(); if (!client) { setError("Supabase 环境变量尚未配置"); return; }
    const result = await client.auth.signInWithPassword({ email: String(form.get("email")), password: String(form.get("password")) });
    if (result.error) setError(result.error.message); else router.replace("/research/new");
  }
  return <main className="login-page"><section className="login-art"><div className="brand-lockup"><div className="brand-mark"><FlaskConical /></div><strong>OPEN DEEP RESEARCH</strong></div><h1>EVIDENCE<br />BEFORE<br />CONFIDENCE.</h1><span className="eyebrow">SECURE RESEARCH WORKSPACE / 2026</span></section><section className="login-card panel"><div className="panel-header"><h2>研究员身份验证</h2></div><form action={submit} className="panel-body"><div className="field"><label htmlFor="email">邮箱</label><input id="email" name="email" type="email" required autoComplete="email" /></div><div className="field"><label htmlFor="password">密码</label><input id="password" name="password" type="password" required autoComplete="current-password" /></div>{error && <p role="alert" style={{ color: "var(--red)" }}>{error}</p>}<button className="primary">进入研究台</button>{localAuthBypass && <p className="dev-badge">LOCAL DEV 已启用，可直接进入</p>}</form></section></main>;
}
