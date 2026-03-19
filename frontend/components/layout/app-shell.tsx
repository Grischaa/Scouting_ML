"use client";

import { useState } from "react";
import { Sidebar } from "@/components/layout/sidebar";
import { Topbar } from "@/components/layout/topbar";
import { notifications, workspaces } from "@/lib/platform-data";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState(workspaces[0]?.id ?? "");

  return (
    <div className="min-h-screen bg-bg text-text">
      <div className="pointer-events-none fixed inset-0 bg-pitch-glow opacity-80" />
      <div
        className="relative lg:grid"
        style={{ gridTemplateColumns: collapsed ? "96px minmax(0,1fr)" : "300px minmax(0,1fr)" }}
      >
        <Sidebar
          mobileOpen={mobileOpen}
          collapsed={collapsed}
          onClose={() => setMobileOpen(false)}
          onToggleCollapse={() => setCollapsed((current) => !current)}
        />
        <div className="min-w-0">
          <Topbar
            onOpenSidebar={() => setMobileOpen(true)}
            activeWorkspaceId={activeWorkspaceId}
            onChangeWorkspace={setActiveWorkspaceId}
            workspaces={workspaces}
            notifications={notifications}
          />
          <main className="px-4 py-6 sm:px-6 lg:px-8">
            <div className="mx-auto max-w-[1760px]">{children}</div>
          </main>
        </div>
      </div>
    </div>
  );
}
