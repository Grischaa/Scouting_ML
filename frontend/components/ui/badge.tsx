import { cn } from "@/lib/utils";

const tones = {
  neutral: "border-white/10 bg-white/5 text-slate-300",
  green: "border-green/25 bg-green/10 text-green",
  blue: "border-blue/25 bg-blue/10 text-blue",
  amber: "border-amber/25 bg-amber/10 text-amber",
  red: "border-red/25 bg-red/10 text-red",
};

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: React.ReactNode;
  tone?: keyof typeof tones;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
