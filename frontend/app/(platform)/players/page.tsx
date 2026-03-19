import Link from "next/link";
import { ArrowRight, FileDown, NotebookPen, Radar } from "lucide-react";
import { ConfidenceBadge } from "@/components/recruitment/confidence-badge";
import { DecisionBadge } from "@/components/recruitment/decision-badge";
import { StatusTag } from "@/components/recruitment/status-tag";
import { ValueGapBadge } from "@/components/recruitment/value-gap-badge";
import { PlayerCard } from "@/components/players/player-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";
import { trackedProfiles } from "@/lib/platform-data";

export default function PlayersPage() {
  const featured = trackedProfiles[0];
  const groups = [
    {
      label: "Pursue now",
      description: "The strongest dossiers with clear next steps attached.",
      profiles: trackedProfiles.filter((profile) => profile.intel.decisionStatus === "Pursue").slice(0, 3),
    },
    {
      label: "Review this week",
      description: "Serious cases that merit board discussion or a final validation step.",
      profiles: trackedProfiles.filter((profile) => profile.intel.decisionStatus === "Review").slice(0, 3),
    },
    {
      label: "Watch and price check",
      description: "Hold these names for timing, leverage, or benchmark discipline.",
      profiles: trackedProfiles.filter((profile) => ["Watch", "Price Check"].includes(profile.intel.decisionStatus)).slice(0, 6),
    },
  ];

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Players"
        title="Tracked player dossiers"
        description="A live library of active recruitment cases, organised by what the club should do next rather than by database storage."
        action={
          <div className="flex flex-wrap gap-3">
            <Link href="/discovery">
              <Button variant="outline" className="gap-2">
                <Radar className="size-4" />
                Open discovery
              </Button>
            </Link>
            <Button className="gap-2">
              <FileDown className="size-4" />
              Export dossier pack
            </Button>
          </div>
        }
      />

      <Card className="overflow-hidden bg-hero">
        <CardContent className="grid gap-6 p-6 xl:grid-cols-[1.2fr_0.8fr] xl:p-8">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <DecisionBadge status={featured.intel.decisionStatus} size="md" />
              <ConfidenceBadge level={featured.intel.confidenceLevel} />
              <ValueGapBadge valueGapM={featured.intel.valueGapM} valueGapPct={featured.intel.valueGapPct} />
              <StatusTag label={featured.intel.priceRealism} />
            </div>
            <h1 className="mt-4 text-4xl font-semibold text-text">{featured.player.name}</h1>
            <p className="mt-2 text-base text-slate-300">
              {featured.player.club} · {featured.player.league} · {featured.intel.roleFitLabel}
            </p>
            <div className="mt-5 rounded-[24px] bg-white/[0.03] p-5">
              <p className="text-label">Next action</p>
              <p className="mt-2 text-2xl font-semibold text-text">{featured.intel.nextAction}</p>
              <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-300">{featured.intel.decisionReason}</p>
            </div>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link href={`/players/${featured.player.slug}`}>
                <Button className="gap-2">
                  Open dossier
                  <ArrowRight className="size-4" />
                </Button>
              </Link>
              <Button variant="panel" className="gap-2">
                <NotebookPen className="size-4" />
                Add board note
              </Button>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-5">
              <p className="text-label">Why this case is live</p>
              <p className="mt-3 text-sm leading-7 text-slate-300">{featured.intel.reliabilityNote}</p>
            </div>
            <div className="rounded-[24px] border border-white/10 bg-panel-2/70 p-5">
              <p className="text-label">Commercial posture</p>
              <p className="mt-3 text-sm leading-7 text-slate-300">{featured.intel.contractUrgency}</p>
              <p className="mt-3 text-sm leading-7 text-muted">{featured.intel.availability}</p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              {featured.intel.modelFlags.map((flag) => (
                <div key={flag} className="rounded-[20px] border border-white/8 bg-panel-2/70 p-4 text-sm text-slate-200">
                  {flag}
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="space-y-6">
        {groups.map((group) => (
          <section key={group.label} className="space-y-4">
            <SectionHeader title={group.label} description={group.description} />
            <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
              {group.profiles.map(({ player }) => (
                <PlayerCard key={player.id} player={player} />
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
