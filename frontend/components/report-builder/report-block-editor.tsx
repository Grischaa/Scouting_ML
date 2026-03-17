"use client";

import { useMemo, useState } from "react";
import { GripVertical, LayoutTemplate, Plus } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";
import type { Player, ReportBlock } from "@/lib/types";
import { formatCurrencyMillions } from "@/lib/utils";

export function ReportBlockEditor({ blocks, player }: { blocks: ReportBlock[]; player: Player }) {
  const [selectedBlocks, setSelectedBlocks] = useState<ReportBlock[]>(blocks.slice(0, 4));

  const preview = useMemo(
    () => ({
      summary: `${player.name} projects as a ${player.archetype.toLowerCase()} with a scouting score of ${player.scoutingScore} and current market value of ${formatCurrencyMillions(player.marketValueM)}.`,
      recommendation: player.report.recommendation,
    }),
    [player],
  );

  return (
    <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
      <Card>
        <CardHeader>
          <SectionHeader
            eyebrow="Composer"
            title="Report blocks"
            description="Assemble a reusable internal report using the same modules scouts already trust."
            action={<Button variant="secondary"><Plus className="mr-2 size-4" />Add section</Button>}
          />
        </CardHeader>
        <CardContent className="space-y-3 pt-4">
          {selectedBlocks.map((block) => (
            <motion.div key={block.id} layout className="flex items-center gap-4 rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
              <div className="flex size-10 items-center justify-center rounded-2xl border border-white/8 bg-panel-2/80 text-muted">
                <GripVertical className="size-4" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-text">{block.label}</p>
                <p className="mt-1 text-sm leading-6 text-muted">{block.description}</p>
              </div>
            </motion.div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <SectionHeader eyebrow="Preview" title="Live document" description="Internal club-report styling with editable narrative blocks." action={<Button variant="outline"><LayoutTemplate className="mr-2 size-4" />Export PDF</Button>} />
        </CardHeader>
        <CardContent className="space-y-5 pt-4">
          <div className="rounded-[22px] border border-white/8 bg-[#101728] p-5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-green">Executive summary</p>
            <h3 className="mt-3 text-2xl font-semibold text-text">{player.name}</h3>
            <p className="mt-3 text-sm leading-7 text-slate-300">{preview.summary}</p>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-muted">Strengths</p>
              <ul className="mt-3 space-y-2 text-sm text-slate-300">
                {player.strengths.map((strength) => (
                  <li key={strength}>• {strength}</li>
                ))}
              </ul>
            </div>
            <div className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-muted">Recommendation</p>
              <p className="mt-3 text-lg font-semibold text-text">{preview.recommendation}</p>
              <p className="mt-2 text-sm leading-6 text-muted">{player.report.tactical}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
