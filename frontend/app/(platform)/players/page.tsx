"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { BookmarkPlus, LayoutGrid, List, SlidersHorizontal } from "lucide-react";
import { motion } from "framer-motion";
import { DataTable, type TableColumn } from "@/components/ui/data-table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { SectionHeader } from "@/components/ui/section-header";
import { FilterPanel, type PlayerFilters } from "@/components/players/filter-panel";
import { PlayerCard } from "@/components/players/player-card";
import { players } from "@/lib/mock-data";
import type { Player } from "@/lib/types";
import { formatCurrencyMillions } from "@/lib/utils";

const initialFilters: PlayerFilters = {
  age: "",
  nationality: "",
  position: "",
  league: "",
  club: "",
  minutes: "",
  marketValue: "",
  contract: "",
  foot: "",
  height: "",
  score: "",
};

export default function PlayersPage() {
  const [filters, setFilters] = useState<PlayerFilters>(initialFilters);
  const [view, setView] = useState<"table" | "grid">("table");

  const setFilter = (key: keyof PlayerFilters, value: string) => setFilters((current) => ({ ...current, [key]: value }));

  const filteredPlayers = useMemo(() => {
    return players.filter((player) => {
      const checks = [
        !filters.age || String(player.age).includes(filters.age),
        !filters.nationality || player.nationality.toLowerCase().includes(filters.nationality.toLowerCase()),
        !filters.position || player.position.toLowerCase().includes(filters.position.toLowerCase()),
        !filters.league || player.league.toLowerCase().includes(filters.league.toLowerCase()),
        !filters.club || player.club.toLowerCase().includes(filters.club.toLowerCase()),
        !filters.minutes || player.minutes >= Number(filters.minutes),
        !filters.marketValue || player.marketValueM <= Number(filters.marketValue),
        !filters.contract || player.contractExpiry.includes(filters.contract),
        !filters.foot || player.preferredFoot.toLowerCase().includes(filters.foot.toLowerCase()),
        !filters.height || player.heightCm >= Number(filters.height),
        !filters.score || player.scoutingScore >= Number(filters.score),
      ];
      return checks.every(Boolean);
    });
  }, [filters]);

  const columns: TableColumn<Player>[] = [
    {
      key: "name",
      header: "Player",
      render: (player) => (
        <Link href={`/players/${player.slug}`} className="block min-w-[220px]">
          <div className="font-medium text-text">{player.name}</div>
          <div className="mt-1 text-xs text-muted">{player.nationality} · {player.secondaryPositions.join(" / ")}</div>
        </Link>
      ),
    },
    { key: "age", header: "Age", align: "right", render: (player) => player.age },
    { key: "club", header: "Club", render: (player) => player.club },
    { key: "league", header: "League", render: (player) => player.league },
    { key: "position", header: "Pos", render: (player) => player.position },
    { key: "minutes", header: "Minutes", align: "right", render: (player) => player.minutes.toLocaleString("en-GB") },
    { key: "value", header: "Market value", align: "right", render: (player) => formatCurrencyMillions(player.marketValueM) },
    { key: "score", header: "Scouting score", align: "right", render: (player) => <span className="font-semibold text-text">{player.scoutingScore}</span> },
    {
      key: "form",
      header: "Form",
      render: (player) => <Badge tone={player.form === "Rising" ? "green" : player.form === "Stable" ? "blue" : "amber"}>{player.form}</Badge>,
    },
  ];

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Discovery"
        title="Player search and discovery"
        description="Search like a serious scouting database: role, value, contract leverage, physical profile, and performance floor in one premium view."
        action={
          <div className="flex gap-3">
            <Button variant="outline" className="gap-2"><SlidersHorizontal className="size-4" />Saved filters</Button>
            <Button className="gap-2"><BookmarkPlus className="size-4" />Create shortlist</Button>
          </div>
        }
      />

      <FilterPanel filters={filters} setFilter={setFilter} />

      <Card>
        <CardContent className="space-y-5 p-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-label">Results</p>
              <h3 className="mt-2 text-xl font-semibold text-text">{filteredPlayers.length} matching profiles</h3>
              <p className="mt-2 text-sm leading-6 text-muted">Toggle between analytical table view and card-first scouting view without losing the current lens.</p>
            </div>
            <div className="flex items-center gap-3">
              <div className="inline-flex rounded-2xl border border-white/8 bg-white/[0.03] p-1">
                <button onClick={() => setView("table")} className={`rounded-xl px-3 py-2 text-sm ${view === "table" ? "bg-white/10 text-text" : "text-muted"}`}>
                  <span className="inline-flex items-center gap-2"><List className="size-4" />Table</span>
                </button>
                <button onClick={() => setView("grid")} className={`rounded-xl px-3 py-2 text-sm ${view === "grid" ? "bg-white/10 text-text" : "text-muted"}`}>
                  <span className="inline-flex items-center gap-2"><LayoutGrid className="size-4" />Grid</span>
                </button>
              </div>
            </div>
          </div>

          {filteredPlayers.length === 0 ? (
            <EmptyState title="No players match this lens" description="Relax the contract, value, or score filters to widen the scouting surface." />
          ) : view === "table" ? (
            <DataTable columns={columns} data={filteredPlayers} rowKey={(player) => player.id} />
          ) : (
            <div className="grid gap-5 md:grid-cols-2 2xl:grid-cols-3">
              {filteredPlayers.map((player, index) => (
                <motion.div key={player.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.03 }}>
                  <PlayerCard player={player} />
                </motion.div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
