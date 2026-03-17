"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Eye, FileDown, NotebookPen, PlusCircle, PlayCircle, Scale, ShieldAlert } from "lucide-react";
import { RadarChartCard } from "@/components/charts/radar-chart-card";
import { PlayerCard } from "@/components/players/player-card";
import { PercentileBarList } from "@/components/players/percentile-bar-list";
import { ScoutNoteCard } from "@/components/players/scout-note-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { DataTable, type TableColumn } from "@/components/ui/data-table";
import { SectionHeader } from "@/components/ui/section-header";
import type { MatchLog, Player } from "@/lib/types";
import { formatCurrencyMillions, formatPercent } from "@/lib/utils";

const tabs = ["Overview", "Performance", "Scouting Report", "Similar Players", "Video / Clips"] as const;

type TabKey = (typeof tabs)[number];

export function PlayerProfileView({ player, similarPlayers }: { player: Player; similarPlayers: Player[] }) {
  const [activeTab, setActiveTab] = useState<TabKey>("Overview");

  const matchColumns: TableColumn<MatchLog>[] = [
    { key: "date", header: "Date", render: (row) => row.date },
    { key: "opponent", header: "Opponent", render: (row) => row.opponent },
    { key: "rating", header: "Rating", align: "right", render: (row) => row.rating.toFixed(1) },
    { key: "minutes", header: "Min", align: "right", render: (row) => row.minutes },
    { key: "goals", header: "Goals", align: "right", render: (row) => row.goals },
    { key: "assists", header: "Assists", align: "right", render: (row) => row.assists },
    { key: "progressivePasses", header: "Prog. passes", align: "right", render: (row) => row.progressivePasses },
  ];

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden bg-hero">
        <CardContent className="space-y-8 p-6 lg:p-8">
          <div className="flex flex-col gap-6 xl:flex-row xl:items-start xl:justify-between">
            <div className="flex items-start gap-5">
              <div className="flex size-24 items-center justify-center rounded-[28px] bg-white/10 text-2xl font-semibold text-text">
                {player.name.split(" ").map((part) => part[0]).join("").slice(0, 2)}
              </div>
              <div className="space-y-4">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone={player.status === "priority" ? "green" : player.status === "shortlist" ? "blue" : "amber"}>{player.status}</Badge>
                    <Badge tone="neutral">{player.position}</Badge>
                    <Badge tone="blue">{player.archetype}</Badge>
                  </div>
                  <h1 className="mt-4 text-4xl font-semibold text-text">{player.name}</h1>
                  <p className="mt-3 text-base leading-7 text-slate-300">
                    {player.age} · {player.nationality} · {player.club} · {player.league}
                  </p>
                </div>
                <p className="max-w-3xl text-sm leading-7 text-slate-300">{player.summary}</p>
              </div>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
              <Button className="gap-2"><PlusCircle className="size-4" />Add to shortlist</Button>
              <Button variant="secondary" className="gap-2"><Scale className="size-4" />Compare</Button>
              <Button variant="outline" className="gap-2"><FileDown className="size-4" />Export report</Button>
              <Button variant="ghost" className="gap-2"><NotebookPen className="size-4" />Add note</Button>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-6">
            {[
              ["Market value", formatCurrencyMillions(player.marketValueM)],
              ["Contract expiry", player.contractExpiry],
              ["Preferred foot", player.preferredFoot],
              ["Height", `${player.heightCm} cm`],
              ["Minutes", player.minutes.toLocaleString("en-GB")],
              ["Scouting score", String(player.scoutingScore)],
            ].map(([label, value]) => (
              <div key={label} className="rounded-[22px] border border-white/10 bg-white/[0.04] p-4">
                <p className="text-label">{label}</p>
                <p className="mt-3 text-xl font-semibold text-text">{value}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="flex flex-wrap gap-2 rounded-[24px] border border-white/8 bg-panel/80 p-2">
        {tabs.map((tab) => (
          <button
            key={tab}
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
                <SectionHeader eyebrow="Summary" title="Season snapshot" description="A compact top-line read before deeper analysis." />
                <div className="grid gap-4 md:grid-cols-3">
                  {player.performanceSplit.map((item) => (
                    <div key={item.category} className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
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
                <SectionHeader eyebrow="Scouting read" title="Strengths and weaknesses" description="Quick read before opening the full qualitative report." />
                <div className="grid gap-5 md:grid-cols-2">
                  <div>
                    <p className="mb-3 text-label">Strengths</p>
                    <div className="flex flex-wrap gap-2">
                      {player.strengths.map((item) => <Badge key={item} tone="green" className="normal-case tracking-normal text-xs">{item}</Badge>)}
                    </div>
                  </div>
                  <div>
                    <p className="mb-3 text-label">Weaknesses</p>
                    <div className="flex flex-wrap gap-2">
                      {player.concerns.map((item) => <Badge key={item} tone="amber" className="normal-case tracking-normal text-xs">{item}</Badge>)}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="space-y-4 p-6">
                <SectionHeader eyebrow="Notes" title="Scout notes" description="Current internal notes attached to this player profile." />
                <div className="grid gap-4 md:grid-cols-2">
                  {player.scoutNotes.map((note) => (
                    <ScoutNoteCard key={`${note.author}-${note.time}`} {...note} />
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            <RadarChartCard title="Role fit radar" data={player.radar} />
            <Card>
              <CardContent className="space-y-4 p-6">
                <SectionHeader eyebrow="Percentiles" title="Percentile bar stack" description="High-signal percentiles relative to the current comparison set." />
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
                <SectionHeader eyebrow="Splits" title="Attacking / passing / defending / physical" description="How the player scores across the core scouting buckets." />
                <div className="space-y-4">
                  {player.performanceSplit.map((item) => (
                    <div key={item.category} className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-200">{item.category}</span>
                        <span className="text-text">{item.current} / 100</span>
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
              <SectionHeader eyebrow="Recent form" title="Match-by-match form" description="Recent match outputs to separate one-off spikes from consistent repeatability." />
              <DataTable columns={matchColumns} data={player.matchLogs} rowKey={(row) => `${row.opponent}-${row.date}`} />
            </CardContent>
          </Card>
        </motion.div>
      )}

      {activeTab === "Scouting Report" && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <Card>
            <CardContent className="space-y-5 p-6">
              <SectionHeader eyebrow="Written report" title="Scout evaluation" description="Editorial report format designed for internal recruitment decisions." />
              <div className="grid gap-4 md:grid-cols-2">
                {[
                  ["Technical", player.report.technical],
                  ["Tactical", player.report.tactical],
                  ["Physical", player.report.physical],
                  ["Mental", player.report.mental],
                ].map(([title, copy]) => (
                  <div key={title} className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
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
                <SectionHeader eyebrow="Recommendation" title="Current decision" description="This block is designed for sporting director review and sign-off." />
                <div className="rounded-[22px] border border-green/20 bg-green/10 p-5">
                  <p className="text-label">Recommendation</p>
                  <p className="mt-3 text-2xl font-semibold text-text">{player.report.recommendation}</p>
                  <p className="mt-3 text-sm leading-7 text-slate-300">The profile fits a recruitment case built around {player.archetype.toLowerCase()}, provided the club accepts the listed caution areas and intended tactical context.</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="space-y-4 p-6">
                <SectionHeader eyebrow="Risk" title="Concerns to resolve" description="Questions to answer before escalating from shortlist to negotiation." />
                <div className="space-y-3">
                  {player.concerns.map((concern) => (
                    <div key={concern} className="flex items-start gap-3 rounded-[20px] border border-amber/20 bg-amber/10 p-4">
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
          <SectionHeader eyebrow="Comparables" title="Similar players" description="Quick comparables for alternative recruitment routes or price anchoring." />
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {similarPlayers.map((item) => (
              <div key={item.id} className="space-y-3">
                <PlayerCard player={item} />
                <div className="flex items-center justify-between rounded-[20px] border border-white/8 bg-white/[0.03] px-4 py-3 text-sm">
                  <span className="text-muted">Similarity score</span>
                  <span className="font-semibold text-text">{formatPercent(74 + item.scoutingScore % 18)}</span>
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
              <SectionHeader eyebrow="Future integration" title="Video and match clips" description="The product is ready for video hooks even when a club has not wired in its clip provider yet." />
              <div className="mt-6 grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
                <div className="flex min-h-[340px] items-center justify-center rounded-[28px] border border-dashed border-white/10 bg-gradient-to-br from-white/[0.04] to-transparent">
                  <div className="text-center">
                    <div className="mx-auto mb-4 flex size-16 items-center justify-center rounded-full border border-white/10 bg-panel-2/80 text-blue">
                      <PlayCircle className="size-8" />
                    </div>
                    <h3 className="text-xl font-semibold text-text">Clips will appear here</h3>
                    <p className="mt-3 max-w-md text-sm leading-7 text-muted">Attach Wyscout, Hudl, or manual cut-ups later. The layout already supports clip reels, phase tags, and analyst comments.</p>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="rounded-[22px] border border-white/8 bg-white/[0.03] p-5">
                    <p className="text-label">Suggested clips</p>
                    <div className="mt-4 space-y-3 text-sm text-slate-300">
                      <div className="flex items-center justify-between"><span>Build-up under pressure</span><span>04:12</span></div>
                      <div className="flex items-center justify-between"><span>Transition defending</span><span>08:31</span></div>
                      <div className="flex items-center justify-between"><span>Final-third creation</span><span>11:02</span></div>
                    </div>
                  </div>
                  <div className="rounded-[22px] border border-white/8 bg-white/[0.03] p-5">
                    <p className="text-label">Workflow</p>
                    <p className="mt-4 text-sm leading-7 text-muted">Use this area for internal video cut-ups, tagging moments, and syncing qualitative comments with the report builder.</p>
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
        <Link href="/players" className="inline-flex items-center gap-2 font-medium text-slate-100">
          Back to search
          <PlusCircle className="size-4" />
        </Link>
      </div>
    </div>
  );
}
