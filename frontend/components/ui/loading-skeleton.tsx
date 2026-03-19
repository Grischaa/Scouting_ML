import { cn } from "@/lib/utils";

export function LoadingSkeleton({ className = "h-20 w-full" }: { className?: string }) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-[24px] border border-white/[0.05] bg-gradient-to-r from-white/[0.04] via-white/[0.08] to-white/[0.04]",
        className,
      )}
    />
  );
}
