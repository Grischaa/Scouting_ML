import { cn } from "@/lib/utils";
import type { PercentileMetric } from "@/lib/types";

const toneClasses = {
  green: "bg-green",
  blue: "bg-blue",
  amber: "bg-amber",
};

export function PercentileBarList({ metrics }: { metrics: PercentileMetric[] }) {
  return (
    <div className="space-y-4">
      {metrics.map((metric) => (
        <div key={metric.label} className="space-y-2 rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-200">{metric.label}</span>
            <span className="font-medium text-text">{metric.value}th</span>
          </div>
          <div className="relative h-2.5 overflow-hidden rounded-full bg-white/6">
            <div className="absolute inset-y-0 left-3/4 w-px bg-white/15" />
            <div className={cn("h-full rounded-full", toneClasses[metric.tone || "green"])} style={{ width: `${metric.value}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
