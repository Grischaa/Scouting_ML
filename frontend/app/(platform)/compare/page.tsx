"use client";

import { useMemo, useState } from "react";
import { Line, LineChart, PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer } from "recharts";
import { ArrowRightLeft, FileDown } from "lucide-react";
import { ConfidenceBadge } from "@/components/recruitment/confidence-badge";
import { DecisionBadge } from "@/components/recruitment/decision-badge";
import { StatusTag } from "@/components/recruitment/status-tag";
import { ValueGapBadge } from "@/components/recruitment/value-gap-badge";
import { PlayerComparisonRow } from "@/components/compare/player-comparison-row";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";
import { compareDefaultIds, playerProfiles } from "@/lib/platform-data";
import type { PlayerProfile } from "@/lib/types";
import { formatCurrencyMillions, formatDateLabel } from "@/lib/utils";

const colors = ["#2EC27E", "#4EA1FF", "#F4B740"];

export default function ComparePage() {
  const [selectedIds, setSelectedIds] = useState(compareDefaultIds);

  const selectedProfiles = useMemo(
    () =>
      selectedIds
        .map((id) => playerProfiles.find((profile) => profile.player.id === id))
        .filter(Boolean) as PlayerProfile[],
    [selectedIds],
  );

  const radarData = useMemo(() => {
    if (!selectedProfiles.length) return [];
    const labels = selectedProfiles[0].player.radar.map((item) => item.subject);
    return labels.map((label) => {
      const row: Record<string, string | number> = { subject: label };
      selectedProfiles.forEach(({ player }) => {
        row[player.name] = player.radar.find((item) => item.subject === label)?.value ?? 0;
      });
      return row;
    });
  }, [selectedProfiles]);

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Compare"
        title="Player comparison room"
        description="Make the decision trade-offs obvious before the final board discussion: conviction, price stance, and the exact next action."
        action={
          <Button variant="secondary" className="gap-2">
            <FileDown className="size-4" />
            Export comparison
          </Button>
        }
      />

      <Card className="sticky top-[104px] z-20">
        <CardHeader>
          <SectionHeader
            eyebrow="Selectors"
            title="Comparison header"
            description="Swap targets without losing the decision context beneath."
            action={<ArrowRightLeft className="size-4 text-blue" />}
          />
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          {[0, 1, 2].map((index) => (
            <label key={index} className="space-y-2">
              <span className="text-label">Player {index + 1}</span>
              <select
                value={selectedIds[index]}
                onChange={(event) =>
                  setSelectedIds((current) => current.map((item, i) => (i === index ? event.target.value : item)))
                }
                className="h-12 w-full rounded-2xl border border-white/10 bg-panel-2/80 px-4 text-sm text-text outline-none focus:border-blue/60"
              >
                {playerProfiles.map(({ player }) => (
                  <option key={player.id} value={player.id}>
                    {player.name}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
        <Card>
          <CardHeader>
            <SectionHeader title="Radar comparison" description="How the players differ in tactical profile and all-phase shape." />
          </CardHeader>
          <CardContent>
            <div className="h-[360px]">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} outerRadius="70%">
                  <PolarGrid stroke="rgba(255,255,255,0.08)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: "#94A3B8", fontSize: 11 }} />
                  {selectedProfiles.map(({ player }, index) => (
                    <Radar
                      key={player.id}
                      name={player.name}
                      dataKey={player.name}
                      stroke={colors[index]}
                      fill={colors[index]}
                      fillOpacity={0.12}
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
            <SectionHeader title="Decision snapshot" description="The call, the conviction, and the next action before deeper comparison." />
          </CardHeader>
          <CardContent className="space-y-4">
            {selectedProfiles.map(({ player, intel }, index) => (
              <div key={player.id} className="rounded-[24px] border border-white/8 bg-white/[0.03] p-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-base font-semibold text-text">{player.name}</p>
                    <p className="mt-1 text-sm text-muted">
                      {player.club} · {player.position}
                    </p>
                  </div>
                  <span className="size-3 rounded-full" style={{ backgroundColor: colors[index] }} />
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  <DecisionBadge status={intel.decisionStatus} />
                  <ConfidenceBadge level={intel.confidenceLevel} compact />
                  <StatusTag label={intel.priceRealism} />
                </div>
                <p className="mt-3 text-sm font-medium text-slate-100">{intel.nextAction}</p>
                <div className="mt-4 h-20">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={player.trend}>
                      <Line type="monotone" dataKey="value" stroke={colors[index]} strokeWidth={2.2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <div className="space-y-4">
        <PlayerComparisonRow label="Decision" values={selectedProfiles.map(({ player, intel }) => ({ name: player.name, value: intel.decisionStatus }))} />
        <PlayerComparisonRow label="Next action" values={selectedProfiles.map(({ player, intel }) => ({ name: player.name, value: intel.nextAction }))} />
        <PlayerComparisonRow label="Confidence" values={selectedProfiles.map(({ player, intel }) => ({ name: player.name, value: intel.confidenceLevel }))} />
        <PlayerComparisonRow
          label="Value gap"
          values={selectedProfiles.map(({ player, intel }) => ({
            name: player.name,
            value: `${formatCurrencyMillions(intel.valueGapM)} (${intel.valueGapPct}%)`,
            highlight: intel.valueGapM === Math.max(...selectedProfiles.map((item) => item.intel.valueGapM)),
          }))}
        />
        <PlayerComparisonRow label="Price stance" values={selectedProfiles.map(({ player, intel }) => ({ name: player.name, value: intel.priceRealism }))} />
        <PlayerComparisonRow label="Contract expiry" values={selectedProfiles.map(({ player }) => ({ name: player.name, value: formatDateLabel(player.contractExpiry) }))} />
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        {selectedProfiles.map(({ player, intel }) => (
          <Card key={player.id}>
            <CardHeader>
              <SectionHeader title={player.name} description={`${intel.roleFitLabel} · ${intel.contractUrgency}`} />
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <DecisionBadge status={intel.decisionStatus} />
                <StatusTag label={intel.readiness} />
                <StatusTag label={intel.priceRealism} />
              </div>
              <div className="rounded-[22px] border border-white/8 bg-panel-2/70 p-4">
                <p className="text-label">Decision reason</p>
                <p className="mt-3 text-sm leading-6 text-slate-300">{intel.decisionReason}</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <ValueGapBadge valueGapM={intel.valueGapM} valueGapPct={intel.valueGapPct} compact />
                  <StatusTag label={intel.priceRealism} />
                </div>
              </div>
              <div>
                <p className="text-label">Strengths</p>
                <ul className="mt-3 space-y-2 text-sm text-slate-300">
                  {player.strengths.map((item) => (
                    <li key={item}>• {item}</li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="text-label">Concerns</p>
                <ul className="mt-3 space-y-2 text-sm text-slate-300">
                  {player.concerns.map((item) => (
                    <li key={item}>• {item}</li>
                  ))}
                </ul>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
