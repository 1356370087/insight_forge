"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import { AuthShell } from "@/components/auth-shell";
import { authFetch } from "@/lib/auth";

export default function VerifyEmailPage() {
  const token = useSearchParams().get("token") ?? ""; const [state, setState] = useState<"idle" | "ok" | "error">("idle"); const [error, setError] = useState("");
  async function verify() { try { await authFetch("/api/auth/verify-email", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token }) }); setState("ok"); } catch (cause) { setError(cause instanceof Error ? cause.message : "验证失败"); setState("error"); } }
  return <AuthShell eyebrow="EMAIL VERIFICATION" title="确认邮箱所有权" copy="验证成功后，账户会进入管理员审批队列。一次性链接使用后立即失效。" footer={<Link href="/login">返回登录</Link>}>
    {state === "ok" ? <div className="auth-success"><b>邮箱已确认</b><p>你的申请正在等待管理员审批。</p><Link className="primary" href="/login">继续登录</Link></div> : <div className="auth-form">{!token && <p className="form-alert error">链接缺少验证令牌。</p>}{state === "error" && <p className="form-alert error">{error}</p>}<button className="primary auth-submit" onClick={verify} disabled={!token}>验证邮箱</button></div>}
  </AuthShell>;
}
