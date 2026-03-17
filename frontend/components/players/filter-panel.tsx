"use client";

import { SlidersHorizontal } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { SectionHeader } from "@/components/ui/section-header";

export interface PlayerFilters {
  age: string;
  nationality: string;
  position: string;
  league: string;
  club: string;
  minutes: string;
  marketValue: string;
  contract: string;
  foot: string;
  height: string;
  score: string;
}

export function FilterPanel({ filters, setFilter }: { filters: PlayerFilters; setFilter: (key: keyof PlayerFilters, value: string) => void }) {
  const activeFilters = Object.entries(filters).filter(([, value]) => value.trim() !== "");

  return (
    <Card className="h-full">
      <CardHeader>
        <SectionHeader
          eyebrow="Discovery"
          title="Advanced Filters"
          description="Filter like a real scouting database, then save the lens for later review."
          action={<Button variant="outline" size="sm"><SlidersHorizontal className="mr-2 size-4" />Save view</Button>}
        />
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {[
            ["age", "Age"],
            ["nationality", "Nationality"],
            ["position", "Position"],
            ["league", "League"],
            ["club", "Club"],
            ["minutes", "Minutes played"],
            ["marketValue", "Market value"],
            ["contract", "Contract expiry"],
            ["foot", "Preferred foot"],
            ["height", "Height"],
            ["score", "Scouting score"],
          ].map(([key, label]) => (
            <label key={key} className="space-y-2 text-sm">
              <span className="text-xs font-semibold uppercase tracking-[0.18em] text-muted">{label}</span>
              <input
                value={filters[key as keyof PlayerFilters]}
                onChange={(event) => setFilter(key as keyof PlayerFilters, event.target.value)}
                placeholder={`Any ${label.toLowerCase()}`}
                className="h-11 w-full rounded-2xl border border-white/10 bg-panel-2/80 px-4 text-sm text-text outline-none transition focus:border-blue/60"
              />
            </label>
          ))}
        </div>

        <div className="flex flex-wrap gap-2">
          {activeFilters.length ? (
            activeFilters.map(([key, value]) => <Badge key={key} tone="blue">{key}: {value}</Badge>)
          ) : (
            <Badge tone="neutral">No filters applied</Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
