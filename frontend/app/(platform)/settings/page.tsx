import { BellRing, ShieldCheck, UserCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Settings"
        title="Workspace settings"
        description="Fine-tune notifications, report defaults, and shared scouting preferences for the club workspace."
        action={<Button>Save changes</Button>}
      />

      <div className="grid gap-5 xl:grid-cols-3">
        <Card>
          <CardContent className="space-y-5 p-6">
            <div className="flex items-center gap-3"><UserCircle2 className="size-5 text-blue" /><h3 className="text-lg font-semibold text-text">Profile</h3></div>
            <div className="space-y-3 text-sm text-muted">
              <div className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">Lead scout: Louis De Smet</div>
              <div className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">Workspace: Asteria Recruitment</div>
              <div className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">Default export style: Sporting director brief</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-5 p-6">
            <div className="flex items-center gap-3"><BellRing className="size-5 text-green" /><h3 className="text-lg font-semibold text-text">Notifications</h3></div>
            <div className="space-y-4 text-sm text-slate-300">
              {[
                "Alert when priority targets move above internal value ceiling",
                "Notify when a shortlist player receives new live report",
                "Send weekly memo pack to sporting director",
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
            <div className="flex items-center gap-3"><ShieldCheck className="size-5 text-amber" /><h3 className="text-lg font-semibold text-text">Decision defaults</h3></div>
            <div className="space-y-4">
              {[
                ["Preferred value tolerance", "Conservative gap only"],
                ["Default age corridor", "18 - 23"],
                ["Primary market lens", "Outside Big Five"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
                  <p className="text-label">{label}</p>
                  <p className="mt-3 text-sm font-medium text-text">{value}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
