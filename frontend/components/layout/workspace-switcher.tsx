"use client";

import { ChevronDown, Layers3 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Workspace } from "@/lib/types";

export function WorkspaceSwitcher({
  workspaces,
  activeId,
  onChange,
  compact = false,
}: {
  workspaces: Workspace[];
  activeId: string;
  onChange: (id: string) => void;
  compact?: boolean;
}) {
  const active = workspaces.find((workspace) => workspace.id === activeId) ?? workspaces[0];

  return (
    <div className={cn("rounded-[22px] border border-white/8 bg-panel-2/72 p-1.5", compact && "w-full")}>
      <label className="relative flex items-center gap-3 rounded-[18px] bg-white/[0.03] px-3 py-2.5">
        <div className="flex size-10 items-center justify-center rounded-2xl border border-white/8 bg-panel/80 text-blue">
          <Layers3 className="size-4" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted">Workspace</p>
          <div className="mt-1 flex items-center gap-2">
            <span className="truncate text-sm font-medium text-text">{active.name}</span>
            <span className="hidden truncate text-xs text-muted md:inline">{active.scope}</span>
          </div>
        </div>
        <div className="pointer-events-none flex items-center gap-2 text-muted">
          <ChevronDown className="size-4" />
        </div>
        <select
          className="absolute inset-0 cursor-pointer opacity-0"
          value={active.id}
          onChange={(event) => onChange(event.target.value)}
          aria-label="Select workspace"
        >
          {workspaces.map((workspace) => (
            <option key={workspace.id} value={workspace.id}>
              {workspace.name}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
