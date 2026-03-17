import { ArrowRight, Clock4 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatCurrencyMillions } from "@/lib/utils";
import type { Shortlist } from "@/lib/types";

const toneMap = {
  Critical: "green",
  Live: "blue",
  Exploratory: "amber",
} as const;

export function ShortlistCard({ shortlist }: { shortlist: Shortlist }) {
  return (
    <Card className="h-full overflow-hidden">
      <CardContent className="space-y-5 p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-text">{shortlist.name}</h3>
            <p className="mt-2 max-w-sm text-sm leading-6 text-muted">{shortlist.note}</p>
          </div>
          <Badge tone={toneMap[shortlist.priority]}>{shortlist.priority}</Badge>
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-3">
            <p className="text-xs uppercase tracking-[0.18em] text-muted">Targets</p>
            <p className="mt-2 text-2xl font-semibold text-text">{shortlist.count}</p>
          </div>
          <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-3">
            <p className="text-xs uppercase tracking-[0.18em] text-muted">Avg age</p>
            <p className="mt-2 text-2xl font-semibold text-text">{shortlist.averageAge.toFixed(1)}</p>
          </div>
          <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-3">
            <p className="text-xs uppercase tracking-[0.18em] text-muted">Avg value</p>
            <p className="mt-2 text-2xl font-semibold text-text">{formatCurrencyMillions(shortlist.averageValueM)}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {shortlist.tags.map((tag) => (
            <Badge key={tag}>{tag}</Badge>
          ))}
        </div>
        <div className="flex items-center justify-between text-sm text-muted">
          <div className="inline-flex items-center gap-2">
            <Clock4 className="size-4 text-blue" />
            Updated {shortlist.updatedAt}
          </div>
          <div className="inline-flex items-center gap-2 text-slate-200">
            Open shortlist
            <ArrowRight className="size-4" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
