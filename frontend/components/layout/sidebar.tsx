"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  ChevronLeft,
  Compass,
  FileText,
  Layers3,
  ListChecks,
  ShieldCheck,
  Settings,
  Target,
  Users2,
  X,
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { href: "/discovery", label: "Discovery", icon: Compass },
  { href: "/players", label: "Players", icon: Users2 },
  { href: "/compare", label: "Compare", icon: Layers3 },
  { href: "/shortlists", label: "Shortlists", icon: Target },
  { href: "/reports", label: "Reports", icon: FileText },
  { href: "/teams", label: "Teams", icon: ListChecks },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar({
  mobileOpen,
  collapsed,
  onClose,
  onToggleCollapse,
}: {
  mobileOpen: boolean;
  collapsed: boolean;
  onClose: () => void;
  onToggleCollapse: () => void;
}) {
  const pathname = usePathname();

  return (
    <>
      <div
        className={cn(
          "fixed inset-0 z-40 bg-black/60 backdrop-blur-sm transition-opacity lg:hidden",
          mobileOpen ? "opacity-100" : "pointer-events-none opacity-0",
        )}
        onClick={onClose}
      />
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex flex-col border-r border-white/8 bg-[#09101d]/95 px-4 py-4 backdrop-blur-2xl transition-all duration-300 lg:sticky lg:top-0 lg:h-screen lg:translate-x-0",
          collapsed ? "w-[96px]" : "w-[300px]",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex items-center justify-between">
          <div className={cn("flex items-center gap-3", collapsed && "justify-center")}>
            <div className="flex size-11 items-center justify-center rounded-2xl bg-gradient-to-br from-green/25 via-blue/15 to-white/10 text-green shadow-[0_12px_24px_rgba(46,194,126,0.18)]">
              <ShieldCheck className="size-5" />
            </div>
            {!collapsed ? (
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-green">ScoutML</p>
                <h1 className="mt-1 text-lg font-semibold text-text">Recruitment OS</h1>
              </div>
            ) : null}
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onToggleCollapse}
              className="hidden rounded-xl border border-white/10 bg-white/[0.04] p-2 text-muted transition hover:text-text lg:inline-flex"
              aria-label="Toggle sidebar"
            >
              <ChevronLeft className={cn("size-4 transition-transform", collapsed && "rotate-180")} />
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl border border-white/10 bg-white/[0.04] p-2 text-muted lg:hidden"
              aria-label="Close sidebar"
            >
              <X className="size-4" />
            </button>
          </div>
        </div>

        <div className={cn("mt-6 rounded-[24px] border border-white/8 bg-white/[0.03] p-4", collapsed && "px-2 py-3")}>
          <div className={cn("flex items-start gap-3", collapsed && "justify-center")}>
            <div className="flex size-10 items-center justify-center rounded-2xl border border-green/20 bg-green/10 text-green">
              <Target className="size-4" />
            </div>
            {!collapsed ? (
              <div>
                <p className="text-sm font-semibold text-text">Summer 2026 focus</p>
                <p className="mt-1 text-sm text-muted">Non-Big-5 undervaluation and contract leverage.</p>
              </div>
            ) : null}
          </div>
        </div>

        <nav className="mt-6 space-y-2">
          {navItems.map((item, index) => {
            const Icon = item.icon;
            const active = pathname === item.href || pathname.startsWith(`${item.href}/`);

            return (
              <motion.div
                key={item.href}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.035 }}
              >
                <Link
                  href={item.href}
                  className={cn(
                    "group flex items-center gap-3 rounded-[22px] px-3 py-3 transition-all",
                    collapsed && "justify-center px-2",
                    active
                      ? "bg-white/[0.08] text-text shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]"
                      : "text-muted hover:bg-white/[0.05] hover:text-text",
                  )}
                  onClick={onClose}
                >
                  <span
                    className={cn(
                      "flex size-10 items-center justify-center rounded-2xl border border-white/8 bg-white/[0.03]",
                      active && "border-blue/25 bg-blue/10 text-blue",
                    )}
                  >
                    <Icon className="size-4" />
                  </span>
                  {!collapsed ? (
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{item.label}</p>
                      <p className="truncate text-xs text-muted/80">
                        {item.label === "Discovery"
                          ? "Market search"
                          : item.label === "Players"
                            ? "Tracked dossiers"
                            : item.label === "Shortlists"
                              ? "Workflow boards"
                              : item.label === "Reports"
                                ? "Board packs"
                                : item.label === "Teams"
                                  ? "Club context"
                                : item.label === "Settings"
                                    ? "Admin diagnostics"
                                    : item.label === "Compare"
                                      ? "Trade-off room"
                                      : "Executive view"}
                      </p>
                    </div>
                  ) : null}
                </Link>
              </motion.div>
            );
          })}
        </nav>

        <div className={cn("mt-auto rounded-[24px] border border-white/8 bg-gradient-to-br from-blue/12 via-white/[0.02] to-green/12 p-4", collapsed && "px-2 py-3")}>
          <div className={cn(collapsed && "text-center")}>
            <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-blue">Live Lens</p>
            {!collapsed ? (
              <>
                <p className="mt-3 text-sm font-semibold text-text">Belgium / Portugal / Turkey</p>
                <p className="mt-2 text-sm leading-6 text-muted">Prioritising controllable fees, role clarity, and resale-positive contract situations.</p>
              </>
            ) : (
              <p className="mt-2 text-xs text-muted">3</p>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}
