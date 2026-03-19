"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Eye, FileDown, NotebookPen, PlusCircle, PlayCircle, Scale, ShieldAlert } from "lucide-react";
import { RadarChartCard } from "@/components/charts/radar-chart-card";
import { ConfidenceBadge } from "@/components/recruitment/confidence-badge";
import { DecisionBadge } from "@/components/recruitment/decision-badge";
import { StatusTag } from "@/components/recruitment/status-tag";
import { ValueGapBadge } from "@/components/recruitment/value-gap-badge";
import { PlayerCard } from "@/components/players/player-card";
import { PercentileBarList } from "@/components/players/percentile-bar-list";
import { ScoutNoteCard } from "@/components/players/scout-note-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { DataTable, type TableColumn } from "@/components/ui/data-table";
import { SectionHeader } from "@/components/ui/section-header";
import { getPlayerIntel } from "@/lib/platform-data";
import type { MatchLog, Player } from "@/lib/types";
import { formatCurrencyMillions, formatDateLabel, initials } from "@/lib/utils";

const tabs = ["Overview", "Performance", "Recruitment Fit", "Scouting Report", "Similar Players", "Video / Clips"] as const;
type TabKey = (typeof tabs)[number];

export function PlayerProfileView({ player, similarPlayers }: { player: Player; similarPlayers: Player[] }) {
  const [activeTab, setActiveTab] = useState<TabKey>("Overview");
  const intel = getPlayerIntel(player);

  const matchColumns: TableColumn<MatchLog>[] = [
    { key: "date", header: "Date", render: (row) => row.date, sortAccessor: (row) => row.date },
    { key: "opponent", header: "Opponent", render: (row) => row.opponent, sortAccessor: (row) => row.opponent },
    { key: "rating", header: "Rating", align: "right", render: (row) => row.rating.toFixed(1), sortAccessor: (row) => row.rating },
    { key: "minutes", header: "Min", align: "right", render: (row) => row.minutes, sortAccessor: (row) => row.minutes },
    { key: "goals", header: "Goals", align: "right", render: (row) => row.goals, sortAccessor: (row) => row.goals },
    { key: "assists", header: "Assists", align: "right", render: (row) => row.assists, sortAccessor: (row) => row.assists },
    {
      key: "progressivePasses",
      header: "Prog. passes",
      align: "right",
      render: (row) => row.progressivePasses,
      sortAccessor: (row) => row.progressivePasses,
    },
  ];

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden">
        <CardContent className="grid gap-6 p-6 xl:grid-cols-[1.08fr_0.92fr_0.7fr]">
          <div className="space-y-5">
            <div className="flex items-start gap-5">
              <div className="flex size-20 items-center justify-center rounded-[28px] bg-gradient-to-br from-green/18 via-blue/18 to-white/[0.08] text-lg font-semibold text-text">
                {initials(player.name)}
              </div>
              <div className="space-y-4">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <DecisionBadge status={intel.decisionStatus} size="md" />
                    <ConfidenceBadge level={intel.confidenceLevel} />
                    <StatusTag label={intel.priceRealism} />
                    <Badge tone="neutral" size="sm" caps={false}>
                      {player.position}
                    </Badge>
                  </div>
                  <h1 className="mt-4 text-3xl font-semibold tracking-tight text-text">{player.name}</h1>
                  <p className="mt-2 text-sm leading-6 text-muted">
                    {player.age} · {player.nationality} · {player.preferredFoot} foot
                  </p>
                  <p className="text-sm leading-6 text-muted">
                    {player.club} · {player.league} · {intel.roleFitLabel}
                  </p>
                </div>

                <div className="flex flex-wrap gap-2">
                  <ValueGapBadge valueGapM={intel.valueGapM} valueGapPct={intel.valueGapPct} />
                  <StatusTag label={intel.readiness} />
                </div>
              </div>
            </div>

            <div className="rounded-[24px] bg-panel-2/65 p-5">
              <p className="text-label">Scouting summary</p>
              <p className="mt-3 text-sm leading-7 text-slate-200">{player.summary}</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-[24px] bg-white/[0.03] p-5">
              <p className="text-label">Next action</p>
              <p className="mt-3 text-2xl font-semibold text-text">{intel.nextAction}</p>
              <p className="mt-3 text-sm leading-6 text-slate-200">{intel.decisionReason}</p>
              <p className="mt-3 text-sm leading-6 text-muted">{intel.reliabilityNote}</p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {[
                ["Market value", formatCurrencyMillions(player.marketValueM)],
                ["Predicted value", formatCurrencyMillions(intel.predictedValueM)],
                ["Contract expiry", formatDateLabel(player.contractExpiry)],
                ["Price stance", intel.priceRealism],
                ["Readiness", intel.readiness],
                ["Role fit", `${intel.roleFitScore}`],
              ].map(([label, value]) => (
                <div key={label} className="rounded-[20px] bg-white/[0.03] p-4">
                  <p className="text-label">{label}</p>
                  <p className="mt-2 text-lg font-semibold text-text">{value}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-3">
            <Button className="w-full gap-2">
              <PlusCircle className="size-4" />
              {intel.nextAction}
            </Button>
            <Button variant="secondary" className="w-full gap-2">
              <Scale className="size-4" />
              Compare
            </Button>
            <Button variant="outline" className="w-full gap-2">
              <FileDown className="size-4" />
              Export report
            </Button>
            <Button variant="panel" className="w-full gap-2">
              <NotebookPen className="size-4" />
              Add note
            </Button>

            <div className="rounded-[24px] bg-white/[0.03] p-4">
              <p className="text-label">Decision call</p>
              <div className="mt-3 flex flex-wrap gap-2">
                <DecisionBadge status={intel.decisionStatus} size="md" />
                <ConfidenceBadge level={intel.confidenceLevel} />
                <StatusTag label={intel.priceRealism} size="md" />
              </div>
              <p className="mt-3 text-sm leading-6 text-muted">{intel.availability}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex flex-wrap gap-2 rounded-[24px] bg-panel-2/60 p-2">
        {tabs.map((tab) => (
          <button
            key={tab}
            type="button"
            className={`rounded-2xl px-4 py-2.5 text-sm font-medium transition ${activeTab === tab ? "bg-white/10 text-text shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]" : "text-muted hover:text-text"}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      {activeTab === "Overview" && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
          <div className="space-y-6">
            <Card>
              <CardContent className="space-y-5 p-6">
                <SectionHeader eyebrow="Summary" title="Season snapshot" description="A compact executive read before deeper tactical or valuation analysis." />
                <div className="grid gap-4 md:grid-cols-4">
                  {player.performanceSplit.map((item) => (
                    <div key={item.category} className="rounded-[20px] bg-white/[0.03] p-4">
                      <p className="text-label">{item.category}</p>
                      <p className="mt-3 text-2xl font-semibold text-text">{item.current}</p>
                      <p className="mt-2 text-sm text-muted">Benchmark {item.benchmark}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="space-y-4 p-6">
                <SectionHeader eyebrow="Recruitment read" title="Role, strengths, and decision call" description="Immediate high-level read for a sporting director or consultant deck." />
                <div className="grid gap-5 md:grid-cols-2">
                  <div className="space-y-4">
                    <div>
                      <p className="text-label">Archetype</p>
                      <p className="mt-3 text-lg font-semibold text-text">{player.archetype}</p>
                      <p className="mt-2 text-sm text-muted">{intel.roleFitLabel}</p>
                    </div>
                    <div>
                      <p className="mb-3 text-label">Strengths</p>
                      <div className="flex flex-wrap gap-2">
                        {player.strengths.map((item) => (
                          <Badge key={item} tone="green" caps={false}>
                            {item}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <p className="text-label">Decision</p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        <DecisionBadge status={intel.decisionStatus} size="md" />
                        <ConfidenceBadge level={intel.confidenceLevel} />
                        <StatusTag label={intel.priceRealism} size="md" />
                      </div>
                      <p className="mt-3 text-sm leading-6 text-muted">{intel.nextAction}</p>
                    </div>
                    <div>
                      <p className="mb-3 text-label">Weaknesses</p>
                      <div className="flex flex-wrap gap-2">
                        {player.concerns.map((item) => (
                          <Badge key={item} tone="amber" caps={false}>
                            {item}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="space-y-4 p-6">
                <SectionHeader eyebrow="Notes" title="Scout notes" description="Current internal notes attached to this player profile." />
                <div className="grid gap-4 md:grid-cols-2">
                  {player.scoutNotes.map((note, index) => (
                    <ScoutNoteCard key={`${note.author}-${note.time}`} {...note} type={index % 2 === 0 ? "Live" : "Video"} />
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            <RadarChartCard title="Role fit radar" data={player.radar} />
            <Card>
              <CardContent className="space-y-4 p-6">
                <SectionHeader eyebrow="Percentiles" title="Percentile stack" description="High-signal percentiles relative to the current comparison set." />
                <PercentileBarList metrics={player.percentiles} />
              </CardContent>
            </Card>
          </div>
        </motion.div>
      )}

      {activeTab === "Performance" && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
            <RadarChartCard title="Attribute profile" data={player.radar} />
            <Card>
              <CardContent className="space-y-5 p-6">
                <SectionHeader eyebrow="Form" title="Trend lines" description="Score trend versus market value trend across the current sample window." />
                <div className="grid gap-6 lg:grid-cols-2">
                  <div className="h-[220px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={player.trend}>
                        <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
                        <XAxis dataKey="label" tick={{ fill: "#94A3B8", fontSize: 11 }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fill: "#94A3B8", fontSize: 11 }} axisLine={false} tickLine={false} />
                        <Tooltip contentStyle={{ background: "#12192B", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16 }} />
                        <Line type="monotone" dataKey="value" stroke="#2EC27E" strokeWidth={2.6} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="h-[220px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={player.marketTrend}>
                        <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
                        <XAxis dataKey="label" tick={{ fill: "#94A3B8", fontSize: 11 }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fill: "#94A3B8", fontSize: 11 }} axisLine={false} tickLine={false} />
                        <Tooltip contentStyle={{ background: "#12192B", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16 }} />
                        <Line type="monotone" dataKey="value" stroke="#4EA1FF" strokeWidth={2.6} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="space-y-4">
                  {player.performanceSplit.map((item) => (
                    <div key={item.category} className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-200">{item.category}</span>
                        <span className="text-text">
                          {item.current} / 100
                        </span>
                      </div>
                      <div className="h-3 overflow-hidden rounded-full bg-white/6">
                        <div className="h-full rounded-full bg-gradient-to-r from-green to-blue" style={{ width: `${item.current}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
          <Card>
            <CardContent className="space-y-4 p-6">
              <SectionHeader eyebrow="Recent form" title="Match log" description="Recent outputs to separate one-off spikes from consistent repeatability." />
              <DataTable columns={matchColumns} data={player.matchLogs} rowKey={(row) => `${row.opponent}-${row.date}`} defaultSortKey="date" defaultSortDirection="desc" />
            </CardContent>
          </Card>
        </motion.div>
      )}

      {activeTab === "Recruitment Fit" && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
          <Card>
            <CardContent className="space-y-5 p-6">
              <SectionHeader eyebrow="Fit" title="Formation and squad context" description="Why the player makes sense for some clubs and not for others." />
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-[22px] bg-white/[0.03] p-4">
                  <p className="text-label">Formation fit</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {intel.formationFits.map((fit) => (
                      <Badge key={fit} tone="blue" caps={false}>
                        {fit}
                      </Badge>
                    ))}
                  </div>
                </div>
                <div className="rounded-[22px] bg-white/[0.03] p-4">
                  <p className="text-label">Squad need alignment</p>
                  <p className="mt-3 text-sm leading-6 text-slate-300">{intel.squadNeed}</p>
                </div>
                <div className="rounded-[22px] bg-white/[0.03] p-4">
                  <p className="text-label">Price realism</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <StatusTag label={intel.priceRealism} />
                    <ValueGapBadge valueGapM={intel.valueGapM} valueGapPct={intel.valueGapPct} compact />
                  </div>
                </div>
                <div className="rounded-[22px] bg-white/[0.03] p-4">
                  <p className="text-label">Readiness</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <StatusTag label={intel.readiness} />
                    <Badge size="sm" caps={false}>
                      {intel.ageCurve}
                    </Badge>
                  </div>
                </div>
              </div>
              <div className="rounded-[24px] bg-panel-2/70 p-5">
                <p className="text-label">Tactical fit explanation</p>
                <p className="mt-3 text-sm leading-7 text-slate-300">{player.report.tactical}</p>
              </div>
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card>
              <CardContent className="space-y-4 p-6">
                <SectionHeader eyebrow="Risk" title="Flags to resolve" description="Risk should stay explicit before a player is pushed into negotiation." />
                <div className="space-y-3">
                  {[
                    `Injury risk: ${intel.injuryRisk}`,
                    `Adaptation risk: ${intel.adaptationRisk}`,
                    intel.reportFocus,
                  ].map((item) => (
                    <div key={item} className="flex items-start gap-3 rounded-[20px] bg-amber/10 p-4">
                      <ShieldAlert className="mt-0.5 size-4 text-amber" />
                      <p className="text-sm leading-6 text-slate-200">{item}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="space-y-4 p-6">
                <SectionHeader eyebrow="Availability" title="Commercial context" description="The recruitment case is only real if the player is actually attainable." />
                <div className="rounded-[22px] bg-white/[0.03] p-4">
                  <p className="text-sm leading-7 text-slate-300">{intel.availability}</p>
                </div>
                <div className="rounded-[22px] bg-white/[0.03] p-4">
                  <p className="text-label">Contract urgency</p>
                  <p className="mt-3 text-sm leading-6 text-slate-300">{intel.contractUrgency}</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </motion.div>
      )}

      {activeTab === "Scouting Report" && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <Card>
            <CardContent className="space-y-5 p-6">
              <SectionHeader eyebrow="Written report" title="Scout evaluation" description="Editorial scouting report designed for internal recruitment meetings." />
              <div className="grid gap-4 md:grid-cols-2">
                {[
                  ["Technical", player.report.technical],
                  ["Tactical", player.report.tactical],
                  ["Physical", player.report.physical],
                  ["Mental", player.report.mental],
                ].map(([title, copy]) => (
                  <div key={title} className="rounded-[22px] bg-white/[0.03] p-4">
                    <p className="text-label">{title}</p>
                    <p className="mt-3 text-sm leading-7 text-slate-300">{copy}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          <div className="space-y-6">
            <Card>
              <CardContent className="space-y-4 p-6">
                <SectionHeader eyebrow="Decision" title="Current call" description="Short, explicit, and ready for sporting director review." />
                <div className="rounded-[24px] bg-green/10 p-5">
                  <div className="flex flex-wrap gap-2">
                    <DecisionBadge status={intel.decisionStatus} size="md" />
                    <ConfidenceBadge level={intel.confidenceLevel} />
                    <StatusTag label={intel.priceRealism} />
                  </div>
                  <p className="mt-4 text-sm leading-7 text-slate-200">
                    {intel.decisionReason} The next operational move is to {intel.nextAction.toLowerCase()}.
                  </p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="space-y-4 p-6">
                <SectionHeader eyebrow="Concerns" title="Open questions" description="Issues to resolve before the club escalates this case." />
                <div className="space-y-3">
                  {player.concerns.map((concern) => (
                    <div key={concern} className="flex items-start gap-3 rounded-[20px] bg-amber/10 p-4">
                      <ShieldAlert className="mt-0.5 size-4 text-amber" />
                      <p className="text-sm leading-6 text-slate-200">{concern}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </motion.div>
      )}

      {activeTab === "Similar Players" && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <SectionHeader eyebrow="Comparables" title="Similar players" description="Alternative routes for price anchoring, stylistic comparison, or shortlist diversification." />
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {similarPlayers.map((item) => (
              <div key={item.id} className="space-y-3">
                <PlayerCard player={item} />
                <div className="flex items-center justify-between rounded-[22px] bg-white/[0.03] px-4 py-3 text-sm">
                  <span className="text-muted">Similarity score</span>
                  <span className="font-semibold text-text">{74 + (item.scoutingScore % 18)}%</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {activeTab === "Video / Clips" && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <Card>
            <CardContent className="p-6">
              <SectionHeader eyebrow="Future integration" title="Video and match clips" description="The layout is ready for clip providers even before a club wires in video infrastructure." />
              <div className="mt-6 grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
                <div className="flex min-h-[340px] items-center justify-center rounded-[28px] border border-dashed border-white/10 bg-gradient-to-br from-white/[0.04] to-transparent">
                  <div className="text-center">
                    <div className="mx-auto mb-4 flex size-16 items-center justify-center rounded-full border border-white/10 bg-panel-2/80 text-blue">
                      <PlayCircle className="size-8" />
                    </div>
                    <h3 className="text-xl font-semibold text-text">Clips will appear here</h3>
                    <p className="mt-3 max-w-md text-sm leading-7 text-muted">
                      Attach Wyscout, Hudl, or manual cut-ups later. The layout already supports clip reels, phase tags, and analyst comments.
                    </p>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="rounded-[22px] bg-white/[0.03] p-5">
                    <p className="text-label">Suggested clips</p>
                    <div className="mt-4 space-y-3 text-sm text-slate-300">
                      <div className="flex items-center justify-between">
                        <span>Build-up under pressure</span>
                        <span>04:12</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Transition defending</span>
                        <span>08:31</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Final-third creation</span>
                        <span>11:02</span>
                      </div>
                    </div>
                  </div>
                  <div className="rounded-[22px] bg-white/[0.03] p-5">
                    <p className="text-label">Workflow</p>
                    <p className="mt-4 text-sm leading-7 text-muted">
                      Use this area for internal cut-ups, tagged moments, and synced qualitative comments attached back into the report builder.
                    </p>
                    <div className="mt-4 inline-flex items-center gap-2 text-sm text-slate-200">
                      <Eye className="size-4 text-green" />
                      Ready for integration
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      <div className="flex items-center justify-between rounded-[24px] border border-white/8 bg-panel/80 px-5 py-4 text-sm text-muted">
        <span>Returning to discovery?</span>
        <Link href="/discovery" className="inline-flex items-center gap-2 font-medium text-slate-100">
          Back to search
          <PlusCircle className="size-4" />
        </Link>
      </div>
    </div>
  );
}
