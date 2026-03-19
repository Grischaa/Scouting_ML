import { BellRing, ShieldCheck, SlidersHorizontal, UserCircle2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";
import { confidenceSummary, executiveKpis } from "@/lib/platform-data";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Settings"
        title="Workspace rules and admin diagnostics"
        description="Keep raw score thresholds, decision defaults, and workflow alerts here instead of leaking them into the core scouting surfaces."
        action={<Button>Save changes</Button>}
      />

      <div className="grid gap-5 xl:grid-cols-3">
        <Card>
          <CardContent className="space-y-5 p-6">
            <div className="flex items-center gap-3">
              <UserCircle2 className="size-5 text-blue" />
              <h3 className="text-lg font-semibold text-text">Profile and workspace</h3>
            </div>
            <div className="space-y-3 text-sm text-muted">
              <div className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">Lead scout: Louis De Smet</div>
              <div className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">Workspace: First Team Recruitment</div>
              <div className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">Default report style: Sporting director brief</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-5 p-6">
            <div className="flex items-center gap-3">
              <BellRing className="size-5 text-green" />
              <h3 className="text-lg font-semibold text-text">Notifications</h3>
            </div>
            <div className="space-y-4 text-sm text-slate-300">
              {[
                "Alert when a pursue-now target needs live scouting",
                "Notify when a review case gets a new report or price change",
                "Send a weekly board memo to the sporting director",
              ].map((label) => (
                <label key={label} className="flex items-start gap-3 rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
                  <input type="checkbox" defaultChecked className="mt-1 size-4 rounded border-white/10 bg-panel-2" />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-5 p-6">
            <div className="flex items-center gap-3">
              <SlidersHorizontal className="size-5 text-amber" />
              <h3 className="text-lg font-semibold text-text">Decision defaults</h3>
            </div>
            <div className="space-y-4">
              {[
                ["Pursue threshold", "High confidence + strong value gap + live action"],
                ["Review threshold", "Board-ready cases with caution still resolved"],
                ["Price-check rule", "Fair-value profiles stay out of the active lane"],
                ["Default market lens", "Outside Big Five"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
                  <p className="text-label">{label}</p>
                  <div className="mt-3 flex items-center justify-between gap-3">
                    <p className="text-sm font-medium text-text">{value}</p>
                    <Badge size="sm" caps={false}>
                      Active
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.05fr_0.95fr]">
        <Card>
          <CardContent className="space-y-5 p-6">
            <div className="flex items-center gap-3">
              <ShieldCheck className="size-5 text-blue" />
              <h3 className="text-lg font-semibold text-text">Confidence diagnostics</h3>
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              {confidenceSummary.map((item) => (
                <div key={item.label} className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
                  <p className="text-label">{item.label}</p>
                  <p className="mt-3 text-2xl font-semibold text-text">{item.value}</p>
                  <p className="mt-2 text-sm leading-6 text-muted">{item.note}</p>
                </div>
              ))}
            </div>
            <div className="rounded-[22px] border border-white/8 bg-panel-2/70 p-4 text-sm text-slate-300">
              <p className="font-semibold text-text">Admin thresholds</p>
              <p className="mt-3">High confidence: score 84 and above.</p>
              <p className="mt-2">Caution: score 74 to 83.</p>
              <p className="mt-2">Thin evidence: score below 74.</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-5 p-6">
            <div className="flex items-center gap-3">
              <ShieldCheck className="size-5 text-amber" />
              <h3 className="text-lg font-semibold text-text">Action-lane diagnostics</h3>
            </div>
            <div className="space-y-4">
              {executiveKpis.map((item) => (
                <div key={item.label} className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
                  <div className="flex items-center justify-between gap-3">
                    <p className="font-medium text-text">{item.label}</p>
                    <span className="text-xl font-semibold text-text">{item.value}</span>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-muted">{item.context}</p>
                  <p className="mt-2 text-xs uppercase tracking-[0.18em] text-slate-400">{item.delta}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
