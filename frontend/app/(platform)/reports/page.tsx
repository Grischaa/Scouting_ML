import { FileDown, LayoutPanelTop } from "lucide-react";
import { ConfidenceBadge } from "@/components/recruitment/confidence-badge";
import { DecisionBadge } from "@/components/recruitment/decision-badge";
import { StatusTag } from "@/components/recruitment/status-tag";
import { ValueGapBadge } from "@/components/recruitment/value-gap-badge";
import { ReportBlockEditor } from "@/components/report-builder/report-block-editor";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";
import { reportFocusPlayer, reportTemplates } from "@/lib/platform-data";

export default function ReportsPage() {
  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Reports"
        title="Report builder"
        description="Build one clean decision memo from the same evidence base used across discovery, shortlists, and player dossiers."
        action={
          <div className="flex flex-wrap gap-3">
            <Button variant="outline" className="gap-2">
              <LayoutPanelTop className="size-4" />
              Use template
            </Button>
            <Button variant="secondary" className="gap-2">
              <FileDown className="size-4" />
              Create report pack
            </Button>
          </div>
        }
      />

      <Card className="overflow-hidden">
        <CardContent className="grid gap-4 p-6 xl:grid-cols-[1.2fr_0.8fr]">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <DecisionBadge status={reportFocusPlayer.intel.decisionStatus} size="md" />
              <ConfidenceBadge level={reportFocusPlayer.intel.confidenceLevel} />
              <StatusTag label={reportFocusPlayer.intel.priceRealism} />
              <ValueGapBadge valueGapM={reportFocusPlayer.intel.valueGapM} valueGapPct={reportFocusPlayer.intel.valueGapPct} />
            </div>
            <h2 className="mt-4 text-3xl font-semibold text-text">{reportFocusPlayer.player.name}</h2>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-300">
              The report should explain the call, the conviction behind it, and the exact next step without forcing the reader to reconstruct the model.
            </p>
          </div>
          <div className="rounded-[24px] bg-panel-2/65 p-5">
            <p className="text-label">Current report intent</p>
            <p className="mt-3 text-lg font-semibold text-text">{reportFocusPlayer.intel.nextAction}</p>
            <p className="mt-2 text-sm leading-6 text-muted">{reportFocusPlayer.intel.decisionReason}</p>
          </div>
        </CardContent>
      </Card>

      <ReportBlockEditor blocks={reportTemplates} player={reportFocusPlayer.player} />
    </div>
  );
}
