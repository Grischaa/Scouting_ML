import { cn } from "@/lib/utils";
import type { Tone } from "@/lib/types";

const tones = {
  neutral: "border-white/10 bg-white/[0.05] text-slate-300",
  green: "border-green/25 bg-green/10 text-green",
  blue: "border-blue/25 bg-blue/10 text-blue",
  amber: "border-amber/25 bg-amber/10 text-amber",
  red: "border-red/25 bg-red/10 text-red",
};

const sizes = {
  sm: "px-2.5 py-1 text-[10px]",
  md: "px-3 py-1.5 text-[11px]",
};

export function Badge({
  children,
  tone = "neutral",
  size = "sm",
  caps = true,
  className,
}: {
  children: React.ReactNode;
  tone?: Tone;
  size?: keyof typeof sizes;
  caps?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border font-semibold",
        tones[tone],
        sizes[size],
        caps ? "uppercase tracking-[0.18em]" : "tracking-normal",
        className,
      )}
    >
      {children}
    </span>
  );
}
