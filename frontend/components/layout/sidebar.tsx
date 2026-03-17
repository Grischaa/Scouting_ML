"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, Compass, FileText, Layers3, Settings, ShieldCheck, Target, Users2, X } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { href: "/players", label: "Players", icon: Compass },
  { href: "/compare", label: "Compare", icon: Layers3 },
  { href: "/shortlists", label: "Shortlists", icon: Target },
  { href: "/reports", label: "Reports", icon: FileText },
  { href: "/teams", label: "Teams", icon: Users2 },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar({ mobileOpen, onClose }: { mobileOpen: boolean; onClose: () => void }) {
  const pathname = usePathname();

  return (
    <>
      <div
        className={cn(
          "fixed inset-0 z-40 bg-black/50 backdrop-blur-sm transition-opacity lg:hidden",
          mobileOpen ? "opacity-100" : "pointer-events-none opacity-0",
        )}
        onClick={onClose}
      />
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex w-[280px] flex-col border-r border-white/8 bg-[#0a1120]/95 px-5 py-5 backdrop-blur-xl transition-transform duration-300 lg:sticky lg:top-0 lg:h-screen lg:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="mb-8 flex items-center justify-between">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-green">ScoutML</p>
            <h1 className="mt-2 text-lg font-semibold text-text">Asteria Recruitment</h1>
            <p className="mt-1 text-sm text-muted">European scouting workspace</p>
          </div>
          <button onClick={onClose} className="rounded-xl border border-white/10 p-2 text-muted lg:hidden">
            <X className="size-4" />
          </button>
        </div>

        <div className="mb-6 rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
          <div className="flex items-center gap-3">
            <div className="flex size-11 items-center justify-center rounded-2xl bg-green/15 text-green">
              <ShieldCheck className="size-5" />
            </div>
            <div>
              <p className="text-sm font-semibold text-text">Window: Summer 2026</p>
              <p className="text-xs text-muted">13 live boards · 4 active scouts</p>
            </div>
          </div>
        </div>

        <nav className="space-y-2">
          {navItems.map((item, index) => {
            const Icon = item.icon;
            const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
            return (
              <motion.div key={item.href} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * 0.04 }}>
                <Link
                  href={item.href}
                  className={cn(
                    "group flex items-center gap-3 rounded-2xl px-3.5 py-3 text-sm font-medium transition-all",
                    active
                      ? "bg-white/8 text-text shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]"
                      : "text-muted hover:bg-white/[0.04] hover:text-text",
                  )}
                  onClick={onClose}
                >
                  <span className={cn("flex size-9 items-center justify-center rounded-xl border border-white/8 bg-white/[0.03]", active && "border-green/30 bg-green/10 text-green")}>
                    <Icon className="size-4" />
                  </span>
                  <span>{item.label}</span>
                </Link>
              </motion.div>
            );
          })}
        </nav>

        <div className="mt-auto rounded-[20px] border border-white/8 bg-gradient-to-br from-blue/12 via-transparent to-green/12 p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-blue">Live focus</p>
          <p className="mt-2 text-sm font-semibold text-text">Belgium / Portugal / Turkey</p>
          <p className="mt-2 text-sm leading-6 text-muted">Current boards prioritise controllable value, contract pressure, and immediate first-team fit.</p>
        </div>
      </aside>
    </>
  );
}
