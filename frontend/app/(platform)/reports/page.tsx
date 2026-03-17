import { reportBlocks, players } from "@/lib/mock-data";
import { ReportBlockEditor } from "@/components/report-builder/report-block-editor";
import { SectionHeader } from "@/components/ui/section-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function ReportsPage() {
  const featuredPlayer = players[1];

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Reports"
        title="Report builder"
        description="Assemble internal reporting packs from reusable modules so club decision-makers receive consistent, high-quality documents."
        action={<Button variant="secondary">Create report pack</Button>}
      />

      <Card>
        <CardContent className="grid gap-4 p-6 lg:grid-cols-[1.2fr_0.8fr]">
          <div>
            <p className="text-label">Current focus player</p>
            <h2 className="mt-3 text-3xl font-semibold text-text">{featuredPlayer.name}</h2>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-muted">This page is designed to feel like internal club software: structured, modular, and editorial rather than like a generic export menu.</p>
          </div>
          <div className="rounded-[22px] border border-white/8 bg-white/[0.03] p-5">
            <p className="text-label">Current report intent</p>
            <p className="mt-3 text-lg font-semibold text-text">Board review for sporting director meeting</p>
            <p className="mt-2 text-sm leading-6 text-muted">Includes summary, stats snapshot, scout notes, and final recommendation blocks.</p>
          </div>
        </CardContent>
      </Card>

      <ReportBlockEditor blocks={reportBlocks} player={featuredPlayer} />
    </div>
  );
}
