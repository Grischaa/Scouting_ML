"use client";

import { useMemo } from "react";
import { usePathname } from "next/navigation";
import { Bell, CalendarRange, Menu, Sparkles } from "lucide-react";
import { WorkspaceSwitcher } from "@/components/layout/workspace-switcher";
import { SearchInput } from "@/components/ui/search-input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { NotificationItem, Workspace } from "@/lib/types";

const routeLabels: Record<string, string> = {
  dashboard: "Executive dashboard",
  discovery: "Discovery workspace",
  players: "Tracked player dossiers",
  compare: "Comparison room",
  shortlists: "Recruitment boards",
  reports: "Report builder",
  teams: "Club context",
  settings: "Admin diagnostics",
};

export function Topbar({
  onOpenSidebar,
  activeWorkspaceId,
  onChangeWorkspace,
  workspaces,
  notifications,
}: {
  onOpenSidebar: () => void;
  activeWorkspaceId: string;
  onChangeWorkspace: (id: string) => void;
  workspaces: Workspace[];
  notifications: NotificationItem[];
}) {
  const pathname = usePathname();

  const routeLabel = useMemo(() => {
    const segment = pathname.split("/").filter(Boolean)[0] ?? "dashboard";
    return routeLabels[segment] ?? "Recruitment workspace";
  }, [pathname]);

  const latestNotification = notifications[0];

  return (
    <header className="sticky top-0 z-30 border-b border-white/8 bg-bg/88 px-4 py-4 backdrop-blur-2xl sm:px-6 lg:px-8">
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={onOpenSidebar}
              className="rounded-2xl border border-white/8 bg-white/[0.03] p-2.5 text-slate-200 lg:hidden"
              aria-label="Open sidebar"
            >
              <Menu className="size-5" />
            </button>
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-green">ScoutML Workspace</p>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-slate-200">
                <span className="font-medium">{routeLabel}</span>
                <Badge tone="neutral" size="sm" caps={false}>
                  March 18, 2026
                </Badge>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden xl:block">
              <WorkspaceSwitcher workspaces={workspaces} activeId={activeWorkspaceId} onChange={onChangeWorkspace} />
            </div>
            <div className="hidden items-center gap-2 rounded-[22px] border border-white/8 bg-panel-2/72 px-4 py-3 text-sm text-muted lg:flex">
              <Bell className="size-4 text-blue" />
              <span>{notifications.length} live updates</span>
            </div>
            <Button variant="panel" className="gap-2">
              <Sparkles className="size-4 text-green" />
              Board pack
            </Button>
            <div className="flex size-11 items-center justify-center rounded-2xl border border-white/8 bg-white/[0.04] text-sm font-semibold text-text">
              LD
            </div>
          </div>
        </div>

        <div className="grid gap-3 xl:grid-cols-[320px_minmax(0,1fr)_300px]">
          <div className="xl:hidden">
            <WorkspaceSwitcher workspaces={workspaces} activeId={activeWorkspaceId} onChange={onChangeWorkspace} compact />
          </div>
          <div className="hidden xl:block">
            <div className="rounded-[22px] border border-white/8 bg-panel-2/72 px-4 py-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-muted">Market pulse</p>
                  <p className="mt-1 text-sm text-slate-200">13 live boards prioritising non-Big-5 value and contract leverage.</p>
                </div>
                <CalendarRange className="size-4 text-amber" />
              </div>
            </div>
          </div>
          <SearchInput />
          <div className="rounded-[22px] border border-white/8 bg-panel-2/72 px-4 py-3">
            <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-muted">Latest alert</p>
            <p className="mt-1 text-sm font-medium text-slate-100">{latestNotification.label}</p>
            <p className="mt-1 text-sm text-muted">{latestNotification.detail}</p>
          </div>
        </div>
      </div>
    </header>
  );
}
