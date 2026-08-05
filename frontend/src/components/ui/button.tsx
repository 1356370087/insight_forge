import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

const buttonVariants = cva("inline-flex items-center justify-center gap-2 border px-3 py-2 text-sm transition-colors disabled:pointer-events-none disabled:opacity-45", {
  variants: {
    variant: {
      default: "border-[var(--cyan)] bg-[var(--cyan)] font-semibold text-[#05110f] hover:bg-[#69e6d4]",
      secondary: "border-[var(--line-hot)] bg-[var(--surface-3)] text-[var(--text)]",
      danger: "border-[#693132] bg-[#2a1516] text-[#ffaaa5]",
    },
  },
  defaultVariants: { variant: "default" },
});

export function Button({ className, variant, asChild, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof buttonVariants> & { asChild?: boolean }) {
  const Component = asChild ? Slot : "button";
  return <Component className={cn(buttonVariants({ variant }), className)} {...props} />;
}
