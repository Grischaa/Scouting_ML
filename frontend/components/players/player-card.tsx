import Link from "next/link";
import { ArrowUpRight, Clock3, ShieldCheck, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import { ConfidenceBadge } from "@/components/recruitment/confidence-badge";
import { DecisionBadge } from "@/components/recruitment/decision-badge";
import { StatusTag } from "@/components/recruitment/status-tag";
import { ValueGapBadge } from "@/components/recruitment/value-gap-badge";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { getPlayerIntel } from "@/lib/platform-data";
import type { Player } from "@/lib/types";
import { formatCurrencyMillions, initials } from "@/lib/utils";

export function PlayerCard({
  player,
  linked = true,
}: {
  player: Player;
  linked?: boolean;
}) {
  const intel = getPlayerIntel(player);
  const card = (
    <Card className="h-full overflow-hidden">
      <CardContent className="space-y-5 p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex size-14 items-center justify-center rounded-[20px] bg-gradient-to-br from-green/18 via-blue/18 to-white/[0.08] text-sm font-semibold text-text">
              {initials(player.name)}
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <DecisionBadge status={intel.decisionStatus} />
                <ConfidenceBadge level={intel.confidenceLevel} compact />
                <Badge tone="neutral" size="sm" caps={false}>
                  {player.position}
                </Badge>
              </div>
              <h3 className="mt-2 text-lg font-semibold text-text">{player.name}</h3>
              <p className="text-sm text-muted">
                {player.nationality} · {player.club} · {player.league}
              </p>
            </div>
          </div>
          <ArrowUpRight className="size-4 text-muted" />
        </div>

        <div className="flex flex-wrap gap-2">
          <ValueGapBadge valueGapM={intel.valueGapM} valueGapPct={intel.valueGapPct} compact />
          <StatusTag label={intel.priceRealism} />
          <StatusTag label={intel.readiness} />
        </div>

        <div className="grid grid-cols-2 gap-3 rounded-[22px] bg-white/[0.03] p-3.5">
          <div>
            <p className="text-label">Market</p>
            <p className="mt-2 text-xl font-semibold text-text">{formatCurrencyMillions(player.marketValueM)}</p>
            <p className="mt-1 text-xs text-muted">Pred. {formatCurrencyMillions(intel.predictedValueM)}</p>
          </div>
          <div>
            <p className="text-label">Role fit</p>
            <p className="mt-2 text-xl font-semibold text-text">{intel.roleFitScore}</p>
            <p className="mt-1 text-xs text-muted">{intel.roleFitLabel}</p>
          </div>
        </div>

        <div className="rounded-[22px] bg-panel-2/72 p-4">
          <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-muted">
            <Sparkles className="size-3.5 text-green" />
            Next action
          </div>
          <p className="mt-2 text-base font-semibold text-text">{intel.nextAction}</p>
          <p className="mt-2 text-sm leading-6 text-slate-300">{intel.decisionReason}</p>
        </div>

        <p className="text-sm leading-6 text-muted">{player.summary}</p>

        <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted">
          <div className="inline-flex items-center gap-2">
            <Clock3 className="size-4 text-blue" />
            {player.minutes.toLocaleString("en-GB")} mins
          </div>
          <div className="inline-flex items-center gap-2">
            <ShieldCheck className="size-4 text-green" />
            {intel.contractUrgency}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <motion.div whileHover={{ y: -5 }} transition={{ duration: 0.22 }}>
      {linked ? <Link href={`/players/${player.slug}`}>{card}</Link> : card}
    </motion.div>
  );
}
