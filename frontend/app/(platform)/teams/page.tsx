import { Gauge, Shield, Waypoints } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";
import { teamContextCards, teamNeeds } from "@/lib/platform-data";

export default function TeamsPage() {
  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Teams"
        title="Club context and formation fit"
        description="Translate player discovery into club-context decisions by matching tactical style, squad need, and value strategy."
      />

      <div className="grid gap-5 xl:grid-cols-3">
        {teamNeeds.map((team) => (
          <Card key={team.id}>
            <CardContent className="space-y-5 p-6">
              <div>
                <p className="text-label">{team.league}</p>
                <h3 className="mt-3 text-2xl font-semibold text-text">{team.team}</h3>
                <p className="mt-2 text-sm leading-6 text-muted">{team.formation} · {team.need}</p>
              </div>
              <div className="rounded-[22px] border border-white/8 bg-white/[0.03] p-4">
                <p className="text-sm leading-6 text-slate-300">{team.fitSummary}</p>
              </div>
              <div className="rounded-[22px] border border-white/8 bg-panel-2/70 p-4">
                <p className="text-label">Value focus</p>
                <p className="mt-3 text-sm leading-6 text-slate-300">{team.valueFocus}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-5 xl:grid-cols-3">
        {teamContextCards.map((team) => (
          <Card key={team.name}>
            <CardContent className="space-y-5 p-6">
              <div>
                <p className="text-label">{team.league}</p>
                <h3 className="mt-3 text-2xl font-semibold text-text">{team.name}</h3>
                <p className="mt-2 text-sm leading-6 text-muted">{team.style}</p>
              </div>
              <div className="space-y-4">
                <div>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="inline-flex items-center gap-2 text-slate-300">
                      <Shield className="size-4 text-green" />
                      Pressing
                    </span>
                    <span>{team.pressing}</span>
                  </div>
                  <div className="h-2.5 rounded-full bg-white/6">
                    <div className="h-full rounded-full bg-green" style={{ width: `${team.pressing}%` }} />
                  </div>
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="inline-flex items-center gap-2 text-slate-300">
                      <Waypoints className="size-4 text-blue" />
                      Possession
                    </span>
                    <span>{team.possession}</span>
                  </div>
                  <div className="h-2.5 rounded-full bg-white/6">
                    <div className="h-full rounded-full bg-blue" style={{ width: `${team.possession}%` }} />
                  </div>
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="inline-flex items-center gap-2 text-slate-300">
                      <Gauge className="size-4 text-amber" />
                      Average age
                    </span>
                    <span>{team.averageAge}</span>
                  </div>
                  <div className="h-2.5 rounded-full bg-white/6">
                    <div className="h-full rounded-full bg-amber" style={{ width: `${team.averageAge * 3}%` }} />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
