"use client";

import { useMemo, useState } from "react";
import { ClipboardCheck, Target, UserRound } from "lucide-react";
import { ConfidenceBadge } from "@/components/recruitment/confidence-badge";
import { DecisionBadge } from "@/components/recruitment/decision-badge";
import { StatusTag } from "@/components/recruitment/status-tag";
import { ValueGapBadge } from "@/components/recruitment/value-gap-badge";
import { ShortlistCard } from "@/components/shortlists/shortlist-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { DataTable, type TableColumn } from "@/components/ui/data-table";
import { SectionHeader } from "@/components/ui/section-header";
import { decisionPriorityMap, shortlistProfiles } from "@/lib/platform-data";
import type { PlayerProfile } from "@/lib/types";
import { formatCurrencyMillions } from "@/lib/utils";

export default function ShortlistsPage() {
  const [activeShortlistId, setActiveShortlistId] = useState(shortlistProfiles[0]?.shortlist.id ?? "");
  const active = shortlistProfiles.find((item) => item.shortlist.id === activeShortlistId) ?? shortlistProfiles[0];

  const columns: TableColumn<PlayerProfile>[] = [
    {
      key: "player",
      header: "Player",
      className: "min-w-[260px]",
      render: ({ player, intel }) => (
        <div>
          <p className="font-semibold text-text">{player.name}</p>
          <p className="mt-1 text-xs text-muted">
            {player.club} · {player.position}
          </p>
          <p className="mt-1 text-xs text-slate-300">{intel.roleFitLabel}</p>
        </div>
      ),
      sortAccessor: ({ player }) => player.name,
    },
    {
      key: "decision",
      header: "Decision",
      render: ({ intel }) => <DecisionBadge status={intel.decisionStatus} />,
      sortAccessor: ({ intel }) => decisionPriorityMap[intel.decisionStatus],
    },
    {
      key: "confidence",
      header: "Confidence",
      render: ({ intel }) => <ConfidenceBadge level={intel.confidenceLevel} compact />,
      sortAccessor: ({ intel }) => intel.confidenceScore,
    },
    {
      key: "gap",
      header: "Value gap",
      render: ({ intel }) => <ValueGapBadge valueGapM={intel.valueGapM} valueGapPct={intel.valueGapPct} compact />,
      sortAccessor: ({ intel }) => intel.valueGapM,
    },
    {
      key: "nextAction",
      header: "Next action",
      className: "min-w-[220px]",
      render: ({ intel }) => <span className="font-medium text-slate-100">{intel.nextAction}</span>,
      sortAccessor: ({ intel }) => intel.nextAction,
    },
    {
      key: "owner",
      header: "Owner",
      render: ({ intel }) => intel.watchlistOwner,
      sortAccessor: ({ intel }) => intel.watchlistOwner,
    },
  ];

  const summary = useMemo(
    () => ({
      pursue: active.profiles.filter((profile) => profile.intel.decisionStatus === "Pursue").length,
      review: active.profiles.filter((profile) => profile.intel.decisionStatus === "Review").length,
      liveActions: active.profiles.filter((profile) => profile.intel.nextAction === "Assign live scouting").length,
      boardReview: active.profiles.filter((profile) => profile.intel.nextAction === "Add to sporting director review").length,
    }),
    [active],
  );

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Shortlists"
        title="Operational recruitment boards"
        description="Boards with ownership, urgency, and explicit action lanes so discovery turns into real club work."
        action={<Button>Create shortlist</Button>}
      />

      <div className="grid gap-5 xl:grid-cols-3">
        {shortlistProfiles.map(({ shortlist, profiles }) => (
          <button key={shortlist.id} type="button" className="text-left" onClick={() => setActiveShortlistId(shortlist.id)}>
            <div className={activeShortlistId === shortlist.id ? "rounded-[28px] ring-2 ring-blue/60" : ""}>
              <ShortlistCard shortlist={shortlist} profiles={profiles} />
            </div>
          </button>
        ))}
      </div>

      <Card>
        <CardContent className="space-y-6 p-6">
          <SectionHeader
            eyebrow="Board detail"
            title={active.shortlist.name}
            description={active.shortlist.note}
            action={
              <div className="flex flex-wrap gap-2">
                <StatusTag label={active.shortlist.priority} />
                <Badge size="sm" caps={false}>
                  {active.shortlist.owner}
                </Badge>
                <Badge tone="neutral" size="sm" caps={false}>
                  Updated {active.shortlist.updatedAt}
                </Badge>
              </div>
            }
          />

          <div className="grid overflow-hidden rounded-[24px] bg-panel-2/65 md:grid-cols-4">
            <div className="p-5">
              <div className="flex items-center gap-2 text-sm text-slate-200">
                <Target className="size-4 text-green" />
                Pursue now
              </div>
              <p className="mt-4 text-2xl font-semibold text-text">{summary.pursue}</p>
            </div>
            <div className="border-t border-white/[0.06] p-5 md:border-l md:border-t-0">
              <div className="flex items-center gap-2 text-sm text-slate-200">
                <ClipboardCheck className="size-4 text-blue" />
                Review this week
              </div>
              <p className="mt-4 text-2xl font-semibold text-text">{summary.review}</p>
            </div>
            <div className="border-t border-white/[0.06] p-5 md:border-l md:border-t-0">
              <div className="flex items-center gap-2 text-sm text-slate-200">
                <UserRound className="size-4 text-amber" />
                Field checks
              </div>
              <p className="mt-4 text-2xl font-semibold text-text">{summary.liveActions}</p>
              <p className="mt-1 text-sm text-muted">Board review {summary.boardReview}</p>
            </div>
            <div className="border-t border-white/[0.06] p-5 md:border-l md:border-t-0">
              <p className="text-label">Average value</p>
              <p className="mt-4 text-2xl font-semibold text-text">{formatCurrencyMillions(active.shortlist.averageValueM)}</p>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {active.shortlist.tags.map((tag) => (
              <Badge key={tag} size="sm" caps={false}>
                {tag}
              </Badge>
            ))}
          </div>

          <DataTable
            columns={columns}
            data={[...active.profiles].sort((left, right) => decisionPriorityMap[right.intel.decisionStatus] - decisionPriorityMap[left.intel.decisionStatus])}
            rowKey={(profile) => profile.player.id}
            className="rounded-none border-0 bg-transparent"
          />
        </CardContent>
      </Card>
    </div>
  );
}
