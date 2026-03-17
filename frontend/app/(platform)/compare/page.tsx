"use client";

import { useMemo, useState } from "react";
import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer } from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";
import { PlayerComparisonRow } from "@/components/compare/player-comparison-row";
import { players } from "@/lib/mock-data";
import type { Player } from "@/lib/types";
import { formatCurrencyMillions } from "@/lib/utils";

const defaultIds = ["p1", "p2", "p3"];
const colors = ["#2EC27E", "#4EA1FF", "#F4B740"];

export default function ComparePage() {
  const [selectedIds, setSelectedIds] = useState(defaultIds);

  const selectedPlayers = useMemo(
    () => selectedIds.map((id) => players.find((player) => player.id === id)).filter(Boolean) as Player[],
    [selectedIds],
  );

  const radarData = useMemo(() => {
    if (!selectedPlayers.length) return [];
    const labels = selectedPlayers[0].radar.map((item) => item.subject);
    return labels.map((label) => {
      const row: Record<string, string | number> = { subject: label };
      selectedPlayers.forEach((player) => {
        row[player.name] = player.radar.find((item) => item.subject === label)?.value ?? 0;
      });
      return row;
    });
  }, [selectedPlayers]);

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Compare"
        title="Player comparison room"
        description="Compare two or three targets side by side to make trade-offs obvious before the final shortlist discussion."
        action={<Button variant="secondary">Export comparison</Button>}
      />

      <Card>
        <CardHeader>
          <SectionHeader title="Selected players" description="Use this sticky comparison header to swap targets without losing the page context." />
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          {[0, 1, 2].map((index) => (
            <label key={index} className="space-y-2">
              <span className="text-label">Player {index + 1}</span>
              <select
                value={selectedIds[index]}
                onChange={(event) => setSelectedIds((current) => current.map((item, i) => (i === index ? event.target.value : item)))}
                className="h-12 w-full rounded-2xl border border-white/10 bg-panel-2/80 px-4 text-sm text-text outline-none focus:border-blue/60"
              >
                {players.map((player) => (
                  <option key={player.id} value={player.id}>{player.name}</option>
                ))}
              </select>
            </label>
          ))}
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <Card>
          <CardHeader>
            <SectionHeader title="Radar comparison" description="Role-shape differences displayed on one tactical profile chart." />
          </CardHeader>
          <CardContent>
            <div className="h-[360px]">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} outerRadius="70%">
                  <PolarGrid stroke="rgba(255,255,255,0.08)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: "#94A3B8", fontSize: 11 }} />
                  {selectedPlayers.map((player, index) => (
                    <Radar
                      key={player.id}
                      name={player.name}
                      dataKey={player.name}
                      stroke={colors[index]}
                      fill={colors[index]}
                      fillOpacity={0.16}
                      strokeWidth={2.4}
                    />
                  ))}
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <SectionHeader title="Snapshot" description="Immediate pricing, age, and role context before diving into detailed rows." />
          </CardHeader>
          <CardContent className="space-y-3">
            {selectedPlayers.map((player, index) => (
              <div key={player.id} className="rounded-[22px] border border-white/8 bg-white/[0.03] p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-base font-semibold text-text">{player.name}</p>
                    <p className="mt-1 text-sm text-muted">{player.club} · {player.position}</p>
                  </div>
                  <span className="size-3 rounded-full" style={{ backgroundColor: colors[index] }} />
                </div>
                <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-label">Score</p>
                    <p className="mt-2 text-lg font-semibold text-text">{player.scoutingScore}</p>
                  </div>
                  <div>
                    <p className="text-label">Value</p>
                    <p className="mt-2 text-lg font-semibold text-text">{formatCurrencyMillions(player.marketValueM)}</p>
                  </div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <div className="space-y-4">
        <PlayerComparisonRow label="Scouting score" values={selectedPlayers.map((player, index) => ({ name: player.name, value: player.scoutingScore, highlight: index === 0 }))} />
        <PlayerComparisonRow label="Market value" values={selectedPlayers.map((player) => ({ name: player.name, value: formatCurrencyMillions(player.marketValueM) }))} />
        <PlayerComparisonRow label="Minutes played" values={selectedPlayers.map((player) => ({ name: player.name, value: player.minutes.toLocaleString("en-GB") }))} />
        <PlayerComparisonRow label="Preferred foot" values={selectedPlayers.map((player) => ({ name: player.name, value: player.preferredFoot }))} />
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        {selectedPlayers.map((player) => (
          <Card key={player.id}>
            <CardHeader>
              <SectionHeader title={player.name} description={`${player.archetype} · ${player.form} form`} />
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <p className="text-label">Strengths</p>
                <ul className="mt-3 space-y-2 text-sm text-slate-300">
                  {player.strengths.map((item) => <li key={item}>• {item}</li>)}
                </ul>
              </div>
              <div>
                <p className="text-label">Concerns</p>
                <ul className="mt-3 space-y-2 text-sm text-slate-300">
                  {player.concerns.map((item) => <li key={item}>• {item}</li>)}
                </ul>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
