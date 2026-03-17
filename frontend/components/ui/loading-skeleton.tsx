export function LoadingSkeleton({ className = "h-20 w-full" }: { className?: string }) {
  return <div className={`animate-pulse rounded-2xl bg-white/6 ${className}`} />;
}
