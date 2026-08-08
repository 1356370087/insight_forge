"use client";

import { useQuery } from "@tanstack/react-query";
import { Clock3, LogOut, RefreshCw } from "lucide-react";
import { useRouter } from "next/navigation";
import { authFetch, iamApi } from "@/lib/auth";

export default function PendingPage() {
  const router = useRouter();
  const query = useQuery({ queryKey: ["identity"], queryFn: iamApi.me, refetchInterval: 15_000 });
  if (query.data?.status === "active") { router.replace("/research/new"); return null; }
  const logout = async () => { await authFetch("/api/auth/logout", { method: "POST" }); router.replace("/login"); };
  return <main className="pending-page"><section className="pending-card panel"><div className="pending-icon"><Clock3 /></div><span className="eyebrow">ACCESS REQUEST / PENDING</span><h1>邮箱已确认，研究权限正在审批</h1><p>你的身份记录已建立，但尚未获得研究工作台权限。管理员批准并分配角色后，此页面会自动更新。</p><dl><div><dt>账户</dt><dd>{query.data?.email ?? "正在读取…"}</dd></div><div><dt>状态</dt><dd>等待管理员批准</dd></div><div><dt>刷新</dt><dd>每 15 秒自动检查</dd></div></dl>{query.error && <p className="form-alert error">无法读取账户状态，请重新登录。</p>}<div className="pending-actions"><button className="secondary" onClick={() => query.refetch()}><RefreshCw size={15} /> 立即检查</button><button className="secondary" onClick={logout}><LogOut size={15} /> 退出登录</button></div></section></main>;
}
