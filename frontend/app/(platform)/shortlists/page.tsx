"use client";

import { useMemo, useState } from "react";
import { CircleGauge, UsersRound } from "lucide-react";
import { shortlists, players } from "@/lib/mock-data";
import { ShortlistCard } from "@/components/shortlists/shortlist-card";
import { DataTable, type TableColumn } from "@/components/ui/data-table";
import { Card, CardContent } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Player } from "@/lib/types";
import { formatCurrencyMillions } from "@/lib/utils";

export default function ShortlistsPage() {
  const [activeShortlistId, setActiveShortlistId] = useState(shortlists[0]?.id ?? "");
  const activeShortlist = shortlists.find((item) => item.id === activeShortlistId) ?? shortlists[0];
  const shortlistPlayers = useMemo(
    () => players.filter((player) => activeShortlist.playerIds.includes(player.id)),
    [activeShortlist],
  );

  const columns: TableColumn<Player>[] = [
    { key: "name", header: "Player", render: (player) => player.name },
    { key: "club", header: "Club", render: (player) => player.club },
    { key: "league", header: "League", render: (player) => player.league },
    { key: "position", header: "Position", render: (player) => player.position },
    { key: "value", header: "Market value", align: "right", render: (player) => formatCurrencyMillions(player.marketValueM) },
    { key: "score", header: "Score", align: "right", render: (player) => player.scoutingScore },
    { key: "status", header: "Status", render: (player) => <Badge tone={player.status === "priority" ? "green" : player.status === "shortlist" ? "blue" : "amber"}>{player.status}</Badge> },
  ];

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Shortlists"
        title="Live shortlists"
        description="Curated recruitment boards with ownership, notes, tags, and a clear decision horizon."
        action={<Button>Create shortlist</Button>}
      />

      <div className="grid gap-5 xl:grid-cols-3">
        {shortlists.map((shortlist) => (
          <button key={shortlist.id} className="text-left" onClick={() => setActiveShortlistId(shortlist.id)}>
            <div className={activeShortlistId === shortlist.id ? "ring-2 ring-blue/60 rounded-[24px]" : ""}>
              <ShortlistCard shortlist={shortlist} />
            </div>
          </button>
        ))}
      </div>

      <Card>
        <CardContent className="space-y-5 p-6">
          <SectionHeader
            eyebrow="Detail"
            title={activeShortlist.name}
            description={activeShortlist.note}
            action={<div className="flex gap-2"><Badge tone="blue">{activeShortlist.owner}</Badge><Badge tone="amber">{activeShortlist.priority}</Badge></div>}
          />
          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
              <div className="flex items-center gap-3"><UsersRound className="size-4 text-green" /><p className="text-label">Targets</p></div>
              <p className="mt-4 text-2xl font-semibold text-text">{activeShortlist.count}</p>
            </div>
            <div className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
              <div className="flex items-center gap-3"><CircleGauge className="size-4 text-blue" /><p className="text-label">Average age</p></div>
              <p className="mt-4 text-2xl font-semibold text-text">{activeShortlist.averageAge.toFixed(1)}</p>
            </div>
            <div className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
              <p className="text-label">Average market value</p>
              <p className="mt-4 text-2xl font-semibold text-text">{formatCurrencyMillions(activeShortlist.averageValueM)}</p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {activeShortlist.tags.map((tag) => <Badge key={tag}>{tag}</Badge>)}
          </div>
          <DataTable columns={columns} data={shortlistPlayers} rowKey={(player) => player.id} />
        </CardContent>
      </Card>
    </div>
  );
}
