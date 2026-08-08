import { FlaskConical, ShieldCheck } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

export function AuthShell({ eyebrow, title, copy, children, footer }: {
  eyebrow: string; title: string; copy: string; children: ReactNode; footer?: ReactNode;
}) {
  return <main className="login-page auth-page">
    <section className="login-art">
      <div className="brand-lockup"><div className="brand-mark"><FlaskConical /></div><strong>OPEN DEEP RESEARCH</strong></div>
      <div><span className="eyebrow">IDENTITY / EVIDENCE CONTROL</span><h1>EVIDENCE<br />BEFORE<br />CONFIDENCE.</h1></div>
      <span className="eyebrow"><ShieldCheck size={13} /> SELF-HOSTED IAM · ARGON2ID · EDDSA</span>
    </section>
    <section className="auth-stage">
      <div className="auth-card panel">
        <span className="eyebrow">{eyebrow}</span>
        <h2>{title}</h2>
        <p className="auth-copy">{copy}</p>
        {children}
        {footer && <div className="auth-footer">{footer}</div>}
      </div>
      <Link className="auth-home" href="/">ODR / COMMAND CONSOLE</Link>
    </section>
  </main>;
}
