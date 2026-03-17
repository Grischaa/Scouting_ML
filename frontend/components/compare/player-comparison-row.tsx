import { cn } from "@/lib/utils";

export function PlayerComparisonRow({
  label,
  values,
}: {
  label: string;
  values: Array<{ name: string; value: number | string; highlight?: boolean }>;
}) {
  return (
    <div className="grid gap-3 rounded-[20px] border border-white/8 bg-white/[0.03] p-4 lg:grid-cols-[180px_repeat(3,minmax(0,1fr))]">
      <div className="text-sm font-medium text-muted">{label}</div>
      {values.map((entry) => (
        <div key={entry.name} className={cn("rounded-2xl border border-white/8 px-4 py-3 text-sm text-slate-300", entry.highlight && "border-green/30 bg-green/10 text-text")}>
          <p className="text-xs uppercase tracking-[0.18em] text-muted">{entry.name}</p>
          <p className="mt-2 text-lg font-semibold">{entry.value}</p>
        </div>
      ))}
    </div>
  );
}
