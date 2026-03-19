"use client";

import { useMemo, useState } from "react";
import { GripVertical, LayoutTemplate, Plus } from "lucide-react";
import { motion } from "framer-motion";
import { ConfidenceBadge } from "@/components/recruitment/confidence-badge";
import { DecisionBadge } from "@/components/recruitment/decision-badge";
import { StatusTag } from "@/components/recruitment/status-tag";
import { ValueGapBadge } from "@/components/recruitment/value-gap-badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";
import { getPlayerIntel } from "@/lib/platform-data";
import type { Player, ReportBlock } from "@/lib/types";
import { formatCurrencyMillions } from "@/lib/utils";

export function ReportBlockEditor({ blocks, player }: { blocks: ReportBlock[]; player: Player }) {
  const [selectedBlocks, setSelectedBlocks] = useState<ReportBlock[]>(blocks.slice(0, 6));
  const intel = getPlayerIntel(player);

  const preview = useMemo(
    () => ({
      summary: `${player.name} projects as a ${intel.roleFitLabel.toLowerCase()} with a current market value of ${formatCurrencyMillions(player.marketValueM)} and a predicted value of ${formatCurrencyMillions(intel.predictedValueM)}.`,
      recommendation: `${intel.decisionStatus} with ${intel.confidenceLevel.toLowerCase()} conviction. ${intel.nextAction} if the club accepts ${intel.reportFocus.toLowerCase()}`,
    }),
    [intel, player],
  );

  return (
    <div className="grid gap-6 xl:grid-cols-[1.02fr_0.98fr]">
      <Card>
        <CardHeader>
          <SectionHeader
            eyebrow="Composer"
            title="Reusable report blocks"
            description="Assemble board-facing or consultant-facing documents from the same repeatable recruitment modules."
            action={
              <Button variant="secondary">
                <Plus className="mr-2 size-4" />
                Add block
              </Button>
            }
          />
        </CardHeader>
        <CardContent className="space-y-3 pt-4">
          {selectedBlocks.map((block) => (
            <motion.button
              key={block.id}
              layout
              type="button"
              className="flex w-full items-center gap-4 rounded-[22px] bg-white/[0.03] p-4 text-left transition hover:bg-white/[0.05]"
              onClick={() =>
                setSelectedBlocks((current) => {
                  const exists = current.find((item) => item.id === block.id);
                  return exists ? current : [...current, block];
                })
              }
            >
              <div className="flex size-10 items-center justify-center rounded-2xl bg-panel-2/80 text-muted">
                <GripVertical className="size-4" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-text">{block.label}</p>
                <p className="mt-1 text-sm leading-6 text-muted">{block.description}</p>
              </div>
            </motion.button>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <SectionHeader
            eyebrow="Preview"
            title="Live board memo"
            description="Editorial layout designed to feel export-ready for sporting directors and consultant clients."
            action={
              <Button variant="outline">
                <LayoutTemplate className="mr-2 size-4" />
                Export PDF
              </Button>
            }
          />
        </CardHeader>
        <CardContent className="space-y-5 pt-4">
          <div className="rounded-[26px] bg-[#101728] p-6">
            <div className="flex flex-wrap items-center gap-2">
              <DecisionBadge status={intel.decisionStatus} size="md" />
              <ConfidenceBadge level={intel.confidenceLevel} />
              <StatusTag label={intel.priceRealism} />
              <ValueGapBadge valueGapM={intel.valueGapM} valueGapPct={intel.valueGapPct} />
            </div>
            <h3 className="mt-4 text-3xl font-semibold text-text">{player.name}</h3>
            <p className="mt-2 text-sm text-muted">
              {player.club} · {player.league} · {intel.roleFitLabel}
            </p>
            <p className="mt-4 text-sm leading-7 text-slate-300">{preview.summary}</p>
            <div className="mt-4 rounded-[22px] bg-white/[0.03] p-4">
              <p className="text-label">Next action</p>
              <p className="mt-2 text-lg font-semibold text-text">{intel.nextAction}</p>
              <p className="mt-2 text-sm leading-6 text-muted">{intel.decisionReason}</p>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-[22px] bg-white/[0.03] p-5">
              <p className="text-label">Stats snapshot</p>
              <div className="mt-4 space-y-3 text-sm text-slate-300">
                <div className="flex items-center justify-between">
                  <span>Scouting score</span>
                  <span>{player.scoutingScore}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Market value</span>
                  <span>{formatCurrencyMillions(player.marketValueM)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Predicted value</span>
                  <span>{formatCurrencyMillions(intel.predictedValueM)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Role fit</span>
                  <span>{intel.roleFitScore}</span>
                </div>
              </div>
            </div>

            <div className="rounded-[22px] bg-white/[0.03] p-5">
              <p className="text-label">Decision call</p>
              <p className="mt-4 text-lg font-semibold text-text">{intel.decisionStatus}</p>
              <p className="mt-2 text-sm leading-6 text-muted">{preview.recommendation}</p>
            </div>
          </div>

          <div className="rounded-[22px] bg-white/[0.03] p-5">
            <p className="text-label">Strengths / concerns</p>
            <div className="mt-4 grid gap-5 md:grid-cols-2">
              <div>
                <p className="text-sm font-semibold text-text">Strengths</p>
                <ul className="mt-3 space-y-2 text-sm text-slate-300">
                  {player.strengths.map((strength) => (
                    <li key={strength}>• {strength}</li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="text-sm font-semibold text-text">Concerns</p>
                <ul className="mt-3 space-y-2 text-sm text-slate-300">
                  {player.concerns.map((concern) => (
                    <li key={concern}>• {concern}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
