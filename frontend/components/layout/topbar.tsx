"use client";

import { Bell, ChevronDown, Menu, Sparkles } from "lucide-react";
import { SearchInput } from "@/components/ui/search-input";
import { Button } from "@/components/ui/button";

export function Topbar({ onOpenSidebar }: { onOpenSidebar: () => void }) {
  return (
    <header className="sticky top-0 z-30 border-b border-white/8 bg-bg/85 px-4 py-4 backdrop-blur-xl sm:px-6 lg:px-8">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onOpenSidebar} className="rounded-2xl border border-white/8 bg-white/[0.03] p-2.5 text-slate-200 lg:hidden">
            <Menu className="size-5" />
          </button>
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.25em] text-green">Workspace</p>
            <div className="mt-1 inline-flex items-center gap-1.5 text-sm font-medium text-slate-200">
              First-team recruitment
              <ChevronDown className="size-4 text-muted" />
            </div>
          </div>
        </div>

        <div className="grid gap-3 xl:min-w-[720px] xl:grid-cols-[minmax(0,1fr)_auto_auto]">
          <SearchInput />
          <div className="flex items-center gap-2 rounded-2xl border border-white/8 bg-panel px-4 py-3 text-sm text-muted">
            <Bell className="size-4 text-blue" />
            4 updates
          </div>
          <Button variant="outline" className="justify-start gap-2">
            <Sparkles className="size-4 text-green" />
            Club board
          </Button>
        </div>
      </div>
    </header>
  );
}
