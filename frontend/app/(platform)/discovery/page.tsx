"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, BookmarkPlus, FileDown, LayoutGrid, List, NotebookPen, Scale, SlidersHorizontal } from "lucide-react";
import { motion } from "framer-motion";
import { ConfidenceBadge } from "@/components/recruitment/confidence-badge";
import { DecisionBadge } from "@/components/recruitment/decision-badge";
import { StatusTag } from "@/components/recruitment/status-tag";
import { ValueGapBadge } from "@/components/recruitment/value-gap-badge";
import { FilterPanel, type PlayerFilters } from "@/components/players/filter-panel";
import { PlayerCard } from "@/components/players/player-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { DataTable, type TableColumn } from "@/components/ui/data-table";
import { EmptyState } from "@/components/ui/empty-state";
import { SectionHeader } from "@/components/ui/section-header";
import {
  confidencePriorityMap,
  decisionPriorityMap,
  discoveryFilterPresets,
  discoveryOptions,
  playerProfiles,
  sortProfilesByDecision,
} from "@/lib/platform-data";
import type { PlayerProfile } from "@/lib/types";
import { formatCurrencyMillions, formatDateLabel, initials } from "@/lib/utils";

const initialFilters: PlayerFilters = {
  ageMax: "",
  nationality: "",
  position: "",
  league: "",
  club: "",
  minMinutes: "",
  maxMarketValue: "",
  contractBefore: "",
  foot: "",
  minHeight: "",
  minScore: "",
  minValueGap: "",
  minConfidence: "",
  nonBig5Only: true,
  archetype: "",
  readiness: "",
  risk: "",
};

const riskRank = { Low: 1, Moderate: 2, Elevated: 3 } as const;

export default function DiscoveryPage() {
  const [filters, setFilters] = useState<PlayerFilters>(initialFilters);
  const [view, setView] = useState<"table" | "grid">("table");
  const [selectedProfileId, setSelectedProfileId] = useState("");

  const setFilter = <K extends keyof PlayerFilters>(key: K, value: PlayerFilters[K]) => {
    setFilters((current) => ({ ...current, [key]: value }));
  };

  const filteredProfiles = useMemo(() => {
    const nextProfiles = playerProfiles.filter(({ player, intel }) => {
      const checks = [
        !filters.ageMax || player.age <= Number(filters.ageMax),
        !filters.nationality || player.nationality.toLowerCase().includes(filters.nationality.toLowerCase()),
        !filters.position || player.position === filters.position,
        !filters.league || player.league === filters.league,
        !filters.club || player.club.toLowerCase().includes(filters.club.toLowerCase()),
        !filters.minMinutes || player.minutes >= Number(filters.minMinutes),
        !filters.maxMarketValue || player.marketValueM <= Number(filters.maxMarketValue),
        !filters.contractBefore || player.contractExpiry <= filters.contractBefore,
        !filters.foot || player.preferredFoot === filters.foot,
        !filters.minHeight || player.heightCm >= Number(filters.minHeight),
        !filters.minScore || player.scoutingScore >= Number(filters.minScore),
        !filters.minValueGap || intel.valueGapM >= Number(filters.minValueGap),
        !filters.minConfidence || confidencePriorityMap[intel.confidenceLevel] >= confidencePriorityMap[filters.minConfidence as keyof typeof confidencePriorityMap],
        !filters.nonBig5Only || !["Premier League", "LaLiga", "Bundesliga", "Serie A", "Ligue 1"].includes(player.league),
        !filters.archetype || player.archetype === filters.archetype,
        !filters.readiness || intel.readiness === filters.readiness,
        !filters.risk || Math.max(riskRank[intel.injuryRisk], riskRank[intel.adaptationRisk]) <= riskRank[filters.risk as keyof typeof riskRank],
      ];

      return checks.every(Boolean);
    });

    return [...nextProfiles].sort(sortProfilesByDecision);
  }, [filters]);

  useEffect(() => {
    if (!filteredProfiles.length) {
      if (selectedProfileId) {
        setSelectedProfileId("");
      }
      return;
    }

    const stillVisible = filteredProfiles.some(({ player }) => player.id === selectedProfileId);
    if (!stillVisible) {
      setSelectedProfileId(filteredProfiles[0].player.id);
    }
  }, [filteredProfiles, selectedProfileId]);

  const selectedProfile = filteredProfiles.find(({ player }) => player.id === selectedProfileId) ?? filteredProfiles[0] ?? null;

  const columns: TableColumn<PlayerProfile>[] = [
    {
      key: "player",
      header: "Player",
      className: "min-w-[320px]",
      render: ({ player, intel }) => (
        <div className="flex items-center gap-3">
          <div className="flex size-11 items-center justify-center rounded-[18px] bg-gradient-to-br from-green/16 via-blue/14 to-white/[0.07] text-xs font-semibold text-text">
            {initials(player.name)}
          </div>
          <div>
            <div className="font-semibold text-text">{player.name}</div>
            <div className="mt-1 text-xs text-muted">
              {player.club} · {player.league}
            </div>
            <div className="mt-1 text-xs text-slate-300">
              {player.age} · {player.position} · {intel.roleFitLabel}
            </div>
          </div>
        </div>
      ),
      sortAccessor: ({ player }) => player.name,
    },
    {
      key: "decision",
      header: "Decision",
      className: "min-w-[190px]",
      render: ({ intel }) => (
        <div className="space-y-2">
          <DecisionBadge status={intel.decisionStatus} />
          <p className="text-xs leading-5 text-muted">{intel.decisionReason}</p>
        </div>
      ),
      sortAccessor: ({ intel }) => decisionPriorityMap[intel.decisionStatus],
    },
    {
      key: "valueGap",
      header: "Value gap",
      render: ({ intel }) => <ValueGapBadge valueGapM={intel.valueGapM} valueGapPct={intel.valueGapPct} compact />,
      sortAccessor: ({ intel }) => intel.valueGapM,
    },
    {
      key: "confidence",
      header: "Confidence",
      render: ({ intel }) => <ConfidenceBadge level={intel.confidenceLevel} compact />,
      sortAccessor: ({ intel }) => confidencePriorityMap[intel.confidenceLevel],
    },
    {
      key: "price",
      header: "Price stance",
      render: ({ intel }) => <StatusTag label={intel.priceRealism} />,
      sortAccessor: ({ intel }) => intel.priceRealism,
    },
    {
      key: "nextAction",
      header: "Next action",
      className: "min-w-[220px]",
      render: ({ intel }) => <span className="font-medium text-slate-100">{intel.nextAction}</span>,
      sortAccessor: ({ intel }) => intel.nextAction,
    },
  ];

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Discovery"
        title="Decision-first discovery workspace"
        description="Filter the market, hold one player in focus, and decide what the club should do next without leaving the page."
        action={
          <div className="flex flex-wrap gap-3">
            <Button variant="outline" className="gap-2">
              <SlidersHorizontal className="size-4" />
              Saved filters
            </Button>
            <Button className="gap-2">Share board</Button>
          </div>
        }
      />

      <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)_420px]">
        <div className="xl:sticky xl:top-24 xl:self-start">
          <FilterPanel
            filters={filters}
            setFilter={setFilter}
            resetFilters={() => setFilters(initialFilters)}
            presets={discoveryFilterPresets}
            options={discoveryOptions}
          />
        </div>

        <Card className="overflow-hidden">
          <CardContent className="p-0">
            <div className="flex flex-col gap-4 border-b border-white/[0.06] px-6 py-5 xl:flex-row xl:items-end xl:justify-between">
              <div>
                <p className="text-label">Decision queue</p>
                <h3 className="mt-2 text-[1.8rem] font-semibold tracking-tight text-text">{filteredProfiles.length} live profiles</h3>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-muted">
                  The queue is ranked by decision priority first, then value gap and conviction. Work from the top until a player no longer deserves time.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                {selectedProfile ? (
                  <div className="rounded-[20px] bg-panel-2/70 px-4 py-3 text-sm">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted">Focused case</p>
                    <p className="mt-1 font-medium text-text">{selectedProfile.player.name}</p>
                    <p className="mt-1 text-xs text-muted">{selectedProfile.intel.nextAction}</p>
                  </div>
                ) : null}
                <div className="inline-flex rounded-2xl bg-white/[0.03] p-1">
                  <button
                    type="button"
                    onClick={() => setView("table")}
                    className={`rounded-xl px-3 py-2 text-sm ${view === "table" ? "bg-white/10 text-text" : "text-muted"}`}
                  >
                    <span className="inline-flex items-center gap-2">
                      <List className="size-4" />
                      Queue
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setView("grid")}
                    className={`rounded-xl px-3 py-2 text-sm ${view === "grid" ? "bg-white/10 text-text" : "text-muted"}`}
                  >
                    <span className="inline-flex items-center gap-2">
                      <LayoutGrid className="size-4" />
                      Cards
                    </span>
                  </button>
                </div>
              </div>
            </div>

            <div className="px-6 py-6">
              {filteredProfiles.length === 0 ? (
                <EmptyState
                  title="No profiles match this recruitment lens"
                  description="Relax the value, confidence, or contract constraints to reopen the active market."
                  action={<Button onClick={() => setFilters(initialFilters)}>Clear filters</Button>}
                />
              ) : view === "table" ? (
                <DataTable
                  columns={columns}
                  data={filteredProfiles}
                  rowKey={(profile) => profile.player.id}
                  selectedRowKey={selectedProfile?.player.id ?? null}
                  onRowClick={(profile) => setSelectedProfileId(profile.player.id)}
                  className="rounded-none border-0 bg-transparent"
                />
              ) : (
                <div className="grid gap-5 md:grid-cols-2 2xl:grid-cols-2">
                  {filteredProfiles.map(({ player }, index) => (
                    <motion.div key={player.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.025 }}>
                      <button type="button" className="block w-full text-left" onClick={() => setSelectedProfileId(player.id)}>
                        <PlayerCard player={player} linked={false} />
                      </button>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="overflow-hidden xl:sticky xl:top-24 xl:self-start">
          <CardContent className="space-y-5 p-6">
            {selectedProfile ? (
              <>
                <div className="rounded-[28px] bg-panel-2/70 p-5">
                  <p className="text-label">Decision rail</p>
                  <div className="mt-4 flex items-start gap-4">
                    <div className="flex size-16 items-center justify-center rounded-[24px] bg-gradient-to-br from-green/18 via-blue/18 to-white/[0.08] text-sm font-semibold text-text">
                      {initials(selectedProfile.player.name)}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <DecisionBadge status={selectedProfile.intel.decisionStatus} size="md" />
                        <ConfidenceBadge level={selectedProfile.intel.confidenceLevel} />
                      </div>
                      <h3 className="mt-3 text-3xl font-semibold tracking-tight text-text">{selectedProfile.player.name}</h3>
                      <p className="mt-2 text-sm leading-6 text-muted">
                        {selectedProfile.player.club} · {selectedProfile.player.league}
                      </p>
                      <p className="text-sm leading-6 text-muted">
                        {selectedProfile.player.age} · {selectedProfile.player.position} · {selectedProfile.intel.roleFitLabel}
                      </p>
                    </div>
                  </div>

                  <div className="mt-5 rounded-[24px] bg-white/[0.03] p-5">
                    <p className="text-label">Next action</p>
                    <p className="mt-2 text-2xl font-semibold leading-tight text-text">{selectedProfile.intel.nextAction}</p>
                    <p className="mt-3 text-sm leading-6 text-slate-300">{selectedProfile.intel.decisionReason}</p>
                  </div>

                  <div className="mt-5 grid gap-3 sm:grid-cols-2">
                    <div className="rounded-[20px] bg-white/[0.03] p-4">
                      <p className="text-label">Value gap</p>
                      <div className="mt-3">
                        <ValueGapBadge valueGapM={selectedProfile.intel.valueGapM} valueGapPct={selectedProfile.intel.valueGapPct} />
                      </div>
                    </div>
                    <div className="rounded-[20px] bg-white/[0.03] p-4">
                      <p className="text-label">Price stance</p>
                      <div className="mt-3">
                        <StatusTag label={selectedProfile.intel.priceRealism} />
                      </div>
                    </div>
                    <div className="rounded-[20px] bg-white/[0.03] p-4">
                      <p className="text-label">Contract</p>
                      <p className="mt-2 text-sm font-medium leading-6 text-slate-200">{formatDateLabel(selectedProfile.player.contractExpiry)}</p>
                      <p className="mt-1 text-sm text-muted">{selectedProfile.intel.contractUrgency}</p>
                    </div>
                    <div className="rounded-[20px] bg-white/[0.03] p-4">
                      <p className="text-label">Availability</p>
                      <p className="mt-2 text-sm font-medium leading-6 text-slate-200">{selectedProfile.intel.availability}</p>
                    </div>
                  </div>
                </div>

                <div className="grid gap-2 sm:grid-cols-2">
                  <Button className="gap-2">
                    <BookmarkPlus className="size-4" />
                    {selectedProfile.intel.nextAction}
                  </Button>
                  <Link href={`/players/${selectedProfile.player.slug}`}>
                    <Button variant="secondary" className="w-full gap-2">
                      <ArrowUpRight className="size-4" />
                      Open full dossier
                    </Button>
                  </Link>
                  <Link href="/compare">
                    <Button variant="panel" className="w-full gap-2">
                      <Scale className="size-4" />
                      Compare options
                    </Button>
                  </Link>
                  <Button variant="outline" className="gap-2">
                    <FileDown className="size-4" />
                    Export case note
                  </Button>
                </div>

                <div className="rounded-[24px] bg-white/[0.03] p-5">
                  <p className="text-label">Why this player stays live</p>
                  <p className="mt-3 text-sm leading-7 text-slate-200">{selectedProfile.player.summary}</p>
                  <p className="mt-3 text-sm leading-6 text-muted">{selectedProfile.intel.reliabilityNote}</p>
                </div>

                <div className="rounded-[24px] bg-white/[0.03] p-5">
                  <p className="text-label">Supporting evidence</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <StatusTag label={selectedProfile.intel.readiness} />
                    {selectedProfile.intel.formationFits.map((fit) => (
                      <Badge key={fit} tone="blue" size="sm" caps={false}>
                        {fit}
                      </Badge>
                    ))}
                  </div>
                  <p className="mt-3 text-sm leading-6 text-muted">{selectedProfile.intel.squadNeed}</p>
                </div>

                <div className="rounded-[24px] bg-white/[0.03] p-5">
                  <p className="text-label">Latest scouting note</p>
                  <p className="mt-3 text-sm font-medium text-slate-200">
                    {selectedProfile.player.scoutNotes[0]?.author ?? "Scout team"} · {selectedProfile.player.scoutNotes[0]?.time ?? "Recent"}
                  </p>
                  <p className="mt-2 text-sm leading-6 text-muted">
                    {selectedProfile.player.scoutNotes[0]?.note ?? selectedProfile.intel.consultantAngle}
                  </p>
                </div>
              </>
            ) : (
              <EmptyState
                title="No active decision rail"
                description="Adjust the lens or reopen a player from the queue to inspect the recruitment call."
              />
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
