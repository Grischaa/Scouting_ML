"use client";

import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";
import { Badge } from "@/components/ui/badge";
import type { FilterPreset } from "@/lib/types";

export interface PlayerFilters {
  ageMax: string;
  nationality: string;
  position: string;
  league: string;
  club: string;
  minMinutes: string;
  maxMarketValue: string;
  contractBefore: string;
  foot: string;
  minHeight: string;
  minScore: string;
  minValueGap: string;
  minConfidence: string;
  nonBig5Only: boolean;
  archetype: string;
  readiness: string;
  risk: string;
}

export function FilterPanel({
  filters,
  setFilter,
  resetFilters,
  presets,
  options,
}: {
  filters: PlayerFilters;
  setFilter: <K extends keyof PlayerFilters>(key: K, value: PlayerFilters[K]) => void;
  resetFilters: () => void;
  presets: FilterPreset[];
  options: {
    positions: readonly string[];
    leagues: string[];
    feet: readonly string[];
    archetypes: string[];
    readiness: readonly string[];
    confidence: readonly string[];
  };
}) {
  const filterEntries = [
    { key: "ageMax", label: "Age", type: "input", placeholder: "Max age" },
    { key: "nationality", label: "Nationality", type: "input", placeholder: "Any nation" },
    { key: "position", label: "Position", type: "select", values: options.positions },
    { key: "league", label: "League", type: "select", values: options.leagues },
    { key: "club", label: "Club", type: "input", placeholder: "Any club" },
    { key: "minMinutes", label: "Minutes", type: "input", placeholder: "Min minutes" },
    { key: "maxMarketValue", label: "Market value", type: "input", placeholder: "Max €m" },
    { key: "contractBefore", label: "Contract expiry", type: "input", placeholder: "Before 2027-06-30" },
    { key: "foot", label: "Preferred foot", type: "select", values: options.feet },
    { key: "minHeight", label: "Height", type: "input", placeholder: "Min cm" },
    { key: "minScore", label: "Scouting score", type: "input", placeholder: "Min score" },
    { key: "minValueGap", label: "Value gap", type: "input", placeholder: "Min €m gap" },
    { key: "minConfidence", label: "Confidence", type: "select", values: options.confidence },
    { key: "archetype", label: "Role fit / archetype", type: "select", values: options.archetypes },
    { key: "readiness", label: "Readiness", type: "select", values: options.readiness },
    { key: "risk", label: "Risk flag", type: "select", values: ["Low", "Moderate", "Elevated"] },
  ] as const;

  const labelMap = Object.fromEntries(filterEntries.map((field) => [field.key, field.label])) as Record<string, string>;

  const activeFilters = Object.entries(filters).filter(([, value]) => {
    if (typeof value === "boolean") return value;
    return value.trim() !== "";
  });

  const clearFilter = (key: keyof PlayerFilters) => {
    const nextValue = typeof filters[key] === "boolean" ? false : "";
    setFilter(key, nextValue as PlayerFilters[typeof key]);
  };

  return (
    <Card className="overflow-hidden bg-panel/88">
      <CardHeader className="pb-4">
        <SectionHeader
          eyebrow="Recruitment Lens"
          title="Set the market constraints"
          description="Trim the market by value, contract pressure, conviction, readiness, and role fit before pushing players into live decisions."
          action={
            <div className="flex flex-wrap gap-2">
              {presets.map((preset) => (
                <Badge key={preset.id} tone="neutral" size="sm" caps={false}>
                  {preset.label}
                </Badge>
              ))}
            </div>
          }
        />
      </CardHeader>
      <CardContent className="space-y-5 p-6 pt-0">
        <div className="grid gap-4 xl:grid-cols-1 md:grid-cols-2">
          {filterEntries.map((field) => (
            <label key={field.key} className="space-y-2 text-sm">
              <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted">{field.label}</span>
              {field.type === "select" ? (
                <select
                  value={String(filters[field.key])}
                  onChange={(event) => setFilter(field.key, event.target.value as PlayerFilters[typeof field.key])}
                  className="h-11 w-full rounded-2xl border border-white/[0.08] bg-panel-2/70 px-4 text-sm text-text outline-none transition focus:border-blue/60"
                >
                  <option value="">Any {field.label.toLowerCase()}</option>
                  {field.values?.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  value={String(filters[field.key])}
                  onChange={(event) => setFilter(field.key, event.target.value as PlayerFilters[typeof field.key])}
                  placeholder={field.placeholder}
                  className="h-11 w-full rounded-2xl border border-white/[0.08] bg-panel-2/70 px-4 text-sm text-text outline-none transition focus:border-blue/60"
                />
              )}
            </label>
          ))}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4 rounded-[24px] bg-panel-2/60 px-4 py-4">
          <label className="flex items-center gap-3 text-sm text-slate-200">
            <input
              type="checkbox"
              checked={filters.nonBig5Only}
              onChange={(event) => setFilter("nonBig5Only", event.target.checked)}
              className="size-4 rounded border-white/10 bg-panel-2"
            />
            Outside Big 5 only
          </label>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={resetFilters}>
              Reset filters
            </Button>
            <Button variant="secondary">Save filter set</Button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 border-t border-white/[0.06] pt-5">
          {activeFilters.length ? (
            activeFilters.map(([key, value]) => (
              <Badge key={key} tone="blue" size="sm" caps={false} className="gap-2">
                {key === "nonBig5Only" ? "Outside Big 5 only" : `${labelMap[key] ?? key}: ${value}`}
                <button type="button" onClick={() => clearFilter(key as keyof PlayerFilters)}>
                  <X className="size-3.5" />
                </button>
              </Badge>
            ))
          ) : (
            <Badge tone="neutral" size="sm" caps={false}>
              No active constraints
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
