import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

const buttonVariants = cva("inline-flex items-center justify-center gap-2 rounded-[5px] border px-3 py-2 text-sm transition-all active:translate-y-px disabled:pointer-events-none disabled:opacity-45", {
  variants: {
    variant: {
      default: "border-[#7dede0] bg-gradient-to-b from-[#5fe7d5] to-[var(--cyan)] font-semibold text-[#05110f] shadow-[0_10px_28px_rgba(66,220,199,.18),inset_0_1px_0_rgba(255,255,255,.22)] hover:-translate-y-px hover:from-[#74ecdc] hover:to-[#4fe0cd] hover:shadow-[0_14px_34px_rgba(66,220,199,.26),inset_0_1px_0_rgba(255,255,255,.26)]",
      secondary: "border-[var(--line-hot)] bg-[var(--surface-3)] text-[var(--text)] shadow-[0_1px_2px_rgba(0,0,0,.4)] hover:-translate-y-px hover:bg-[#1c2833]",
      danger: "border-[#693132] bg-[#2a1516] text-[#ffaaa5] hover:bg-[#341a1b]",
    },
  },
  defaultVariants: { variant: "default" },
});

export function Button({ className, variant, asChild, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof buttonVariants> & { asChild?: boolean }) {
  const Component = asChild ? Slot : "button";
  return <Component className={cn(buttonVariants({ variant }), className)} {...props} />;
}
