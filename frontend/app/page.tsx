"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, BarChart3, FileText, ShieldCheck, Target, Users2 } from "lucide-react";
import { ConfidenceBadge } from "@/components/recruitment/confidence-badge";
import { DecisionBadge } from "@/components/recruitment/decision-badge";
import { ValueGapBadge } from "@/components/recruitment/value-gap-badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { opportunityBoard, workspaces } from "@/lib/platform-data";
import { formatCurrencyMillions } from "@/lib/utils";

const proofPoints = [
  { label: "Leagues monitored", value: "19", detail: "Built for non-Big-5 discovery" },
  { label: "Live shortlists", value: "12", detail: "Role-based recruitment boards" },
  { label: "Reports shipped", value: "46", detail: "Board and consultant-ready packs" },
];

export default function LandingPage() {
  const featuredProfiles = opportunityBoard.slice(0, 3);

  return (
    <div className="min-h-screen bg-bg px-4 py-5 sm:px-6 lg:px-8">
      <div className="mx-auto grid min-h-[calc(100vh-2.5rem)] max-w-[1500px] gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <Card className="relative overflow-hidden border-white/10 bg-hero">
          <div className="pointer-events-none absolute inset-0 panel-grid opacity-[0.08]" />
          <CardContent className="relative flex h-full flex-col justify-between p-8 sm:p-10 xl:p-12">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.05] px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-green">
                <ShieldCheck className="size-4" />
                ScoutML
              </div>
              <motion.h1
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45 }}
                className="mt-8 max-w-4xl text-4xl font-semibold leading-tight text-text sm:text-6xl"
              >
                Recruitment intelligence for clubs that need market edge, not market spend.
              </motion.h1>
              <motion.p
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, delay: 0.08 }}
                className="mt-6 max-w-3xl text-base leading-8 text-slate-300 sm:text-lg"
              >
                ScoutML helps scouts, recruitment analysts, sporting directors, and consultants identify undervalued players outside the Big 5, assess role fit, and turn pricing signals into explicit recruitment decisions.
              </motion.p>

              <div className="mt-10 grid gap-4 md:grid-cols-3">
                {proofPoints.map((item) => (
                  <div key={item.label} className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
                    <p className="text-label">{item.label}</p>
                    <p className="mt-3 text-3xl font-semibold text-text">{item.value}</p>
                    <p className="mt-2 text-sm text-muted">{item.detail}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
              <div className="rounded-[28px] border border-white/10 bg-[#0d1628]/82 p-5">
                <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.24em] text-blue">
                  <Target className="size-3.5" />
                  Opportunity board
                </div>
                <div className="mt-4 space-y-3">
                  {featuredProfiles.map(({ player, intel }) => (
                    <div key={player.id} className="rounded-[22px] border border-white/8 bg-white/[0.03] p-4">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <p className="font-semibold text-text">{player.name}</p>
                          <p className="mt-1 text-sm text-muted">
                            {player.club} · {player.position} · {intel.roleFitLabel}
                          </p>
                        </div>
                        <ValueGapBadge valueGapM={intel.valueGapM} valueGapPct={intel.valueGapPct} />
                      </div>
                      <div className="mt-3 flex flex-wrap items-center gap-2">
                        <DecisionBadge status={intel.decisionStatus} />
                        <ConfidenceBadge level={intel.confidenceLevel} />
                        <span className="rounded-full border border-white/8 bg-white/[0.04] px-3 py-1 text-xs text-muted">
                          Predicted {formatCurrencyMillions(intel.predictedValueM)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[28px] border border-white/10 bg-[#0d1628]/82 p-5">
                <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.24em] text-green">
                  <FileText className="size-3.5" />
                  Product story
                </div>
                <div className="mt-5 space-y-4 text-sm leading-6 text-slate-300">
                  <div className="flex items-start gap-3">
                    <BarChart3 className="mt-0.5 size-4 text-blue" />
                    Decision-first boards turn value gap, conviction, and price stance into a clear next step.
                  </div>
                  <div className="flex items-start gap-3">
                    <Users2 className="mt-0.5 size-4 text-green" />
                    Discovery moves directly into dossiers, recruitment boards, and consultant-ready reports.
                  </div>
                  <div className="flex items-start gap-3">
                    <Target className="mt-0.5 size-4 text-amber" />
                    Tactical fit, formation fit, and squad need stay visible without overpowering the decision itself.
                  </div>
                </div>

                <div className="mt-6 rounded-[22px] border border-white/8 bg-white/[0.03] p-4">
                  <p className="text-label">Active workspaces</p>
                  <div className="mt-4 space-y-3">
                    {workspaces.map((workspace) => (
                      <div key={workspace.id} className="rounded-[18px] border border-white/8 bg-panel-2/72 px-4 py-3">
                        <p className="text-sm font-semibold text-text">{workspace.name}</p>
                        <p className="mt-1 text-sm text-muted">{workspace.scope}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-white/10 bg-panel/92">
          <CardContent className="flex h-full flex-col justify-between p-8 sm:p-10">
            <div>
              <p className="text-label">Club access</p>
              <h2 className="mt-3 text-3xl font-semibold text-text">Open the recruitment workspace</h2>
              <p className="mt-3 text-sm leading-7 text-muted">
                Mock data only. Designed to feel like a serious internal platform used by clubs and boutique recruitment consultants.
              </p>
            </div>

            <form className="mt-8 space-y-4">
              <label className="block space-y-2">
                <span className="text-label">Club or consultancy email</span>
                <input
                  className="h-12 w-full rounded-2xl border border-white/10 bg-panel-2/80 px-4 text-sm text-text outline-none transition focus:border-blue/60"
                  placeholder="name@club.com"
                />
              </label>
              <label className="block space-y-2">
                <span className="text-label">Password</span>
                <input
                  type="password"
                  className="h-12 w-full rounded-2xl border border-white/10 bg-panel-2/80 px-4 text-sm text-text outline-none transition focus:border-blue/60"
                  placeholder="••••••••"
                />
              </label>

              <div className="rounded-[24px] border border-white/8 bg-white/[0.03] p-4">
                <p className="text-label">Included workflows</p>
                <div className="mt-4 space-y-3 text-sm text-slate-300">
                  <div>Discovery for undervalued non-Big-5 profiles</div>
                  <div>Decision-first dossiers with next actions and price stance</div>
                  <div>Boards, reports, and comparison flows built around one decision model</div>
                </div>
              </div>

              <Link href="/dashboard" className="block">
                <Button className="mt-2 w-full justify-center gap-2" size="lg">
                  Open platform
                  <ArrowRight className="size-4" />
                </Button>
              </Link>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
