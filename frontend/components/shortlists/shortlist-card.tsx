import { ArrowRight, Clock4, UserRound } from "lucide-react";
import { StatusTag } from "@/components/recruitment/status-tag";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatCurrencyMillions } from "@/lib/utils";
import type { PlayerProfile, Shortlist } from "@/lib/types";

export function ShortlistCard({
  shortlist,
  profiles = [],
}: {
  shortlist: Shortlist;
  profiles?: PlayerProfile[];
}) {
  const pursueCount = profiles.filter((profile) => profile.intel.decisionStatus === "Pursue").length;
  const reviewCount = profiles.filter((profile) => profile.intel.decisionStatus === "Review").length;
  const watchCount = profiles.filter((profile) => ["Watch", "Price Check"].includes(profile.intel.decisionStatus)).length;

  return (
    <Card className="h-full overflow-hidden">
      <CardContent className="space-y-5 p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <StatusTag label={shortlist.priority} />
              <Badge tone="neutral" size="sm" caps={false}>
                {shortlist.owner}
              </Badge>
            </div>
            <h3 className="mt-3 text-lg font-semibold text-text">{shortlist.name}</h3>
            <p className="mt-2 max-w-sm text-sm leading-6 text-muted">{shortlist.note}</p>
          </div>
          <ArrowRight className="size-4 text-muted" />
        </div>

        <div className="grid overflow-hidden rounded-[22px] bg-white/[0.03] sm:grid-cols-3">
          <div className="p-4">
            <p className="text-label">Pursue</p>
            <p className="mt-2 text-2xl font-semibold text-text">{pursueCount}</p>
          </div>
          <div className="border-t border-white/[0.06] p-4 sm:border-l sm:border-t-0">
            <p className="text-label">Review</p>
            <p className="mt-2 text-2xl font-semibold text-text">{reviewCount}</p>
          </div>
          <div className="border-t border-white/[0.06] p-4 sm:border-l sm:border-t-0">
            <p className="text-label">Watch / price</p>
            <p className="mt-2 text-2xl font-semibold text-text">{watchCount}</p>
            <p className="mt-1 text-xs text-muted">Avg value {formatCurrencyMillions(shortlist.averageValueM)}</p>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {shortlist.tags.map((tag) => (
            <Badge key={tag} size="sm" caps={false}>
              {tag}
            </Badge>
          ))}
        </div>

        <div className="rounded-[22px] bg-panel-2/72 p-4">
          <div className="flex items-center gap-2 text-sm text-slate-200">
            <UserRound className="size-4 text-blue" />
            Assignment owner: {shortlist.owner}
          </div>
          <div className="mt-2 flex items-center gap-2 text-sm text-muted">
            <Clock4 className="size-4 text-blue" />
            Updated {shortlist.updatedAt}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
