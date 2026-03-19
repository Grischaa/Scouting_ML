"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, CalendarRange, CircleGauge, ClipboardList, Radar, ShieldCheck, Sparkles, TrendingUp, Users2 } from "lucide-react";
import { AgeValueScatter } from "@/components/charts/age-value-scatter";
import { ChartCard } from "@/components/charts/chart-card";
import { PerformanceTrendChart } from "@/components/charts/performance-trend-chart";
import { PositionDistributionChart } from "@/components/charts/position-distribution-chart";
import { ConfidenceBadge } from "@/components/recruitment/confidence-badge";
import { DecisionBadge } from "@/components/recruitment/decision-badge";
import { StatusTag } from "@/components/recruitment/status-tag";
import { ValueGapBadge } from "@/components/recruitment/value-gap-badge";
import { PlayerCard } from "@/components/players/player-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";
import { StatCard } from "@/components/ui/stat-card";
import { dashboardActivity, executiveKpis, opportunityBoard, roleLeaders, taskBoard, trendingProfiles } from "@/lib/platform-data";
import { ageValueScatterData, performanceTrendData, positionDistributionData } from "@/lib/mock-data";
import { formatCurrencyMillions, initials } from "@/lib/utils";

const kpiIcons = [Users2, Sparkles, ClipboardList, TrendingUp, ShieldCheck, CircleGauge];

export default function DashboardPage() {
  const primaryOpportunity = opportunityBoard[0];
  const supportingOpportunities = opportunityBoard.slice(1, 5);

  return (
    <div className="space-y-7">
      <SectionHeader
        eyebrow="Dashboard"
        title="What is safe to act on today"
        description="Lead with the strongest live cases, keep the next action explicit, and make it obvious which names deserve club time now."
        action={
          <div className="flex flex-wrap gap-3">
            <Button variant="outline" className="gap-2">
              <CalendarRange className="size-4" />
              Window: Summer 2026
            </Button>
            <Button variant="secondary" className="gap-2">
              <Radar className="size-4" />
              Generate board pack
            </Button>
          </div>
        }
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-6">
        {executiveKpis.map((kpi, index) => {
          const Icon = kpiIcons[index];
          return (
            <motion.div key={kpi.label} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.05 }}>
              <StatCard {...kpi} icon={<Icon className="size-5" />} />
            </motion.div>
          );
        })}
      </div>

      <div className="grid gap-5 xl:grid-cols-12">
        <div className="space-y-5 xl:col-span-8">
          <Card className="overflow-hidden">
            <CardContent className="grid gap-6 p-6 xl:grid-cols-[1.15fr_0.85fr]">
              {primaryOpportunity ? (
                <>
                  <div className="space-y-5">
                    <SectionHeader
                      eyebrow="Primary Case"
                      title={primaryOpportunity.player.name}
                      description="The clearest case to move today when price, confidence, and role fit line up."
                    />

                    <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                      <div className="flex items-start gap-4">
                        <div className="flex size-16 items-center justify-center rounded-[24px] bg-gradient-to-br from-green/18 via-blue/18 to-white/[0.08] text-sm font-semibold text-text">
                          {initials(primaryOpportunity.player.name)}
                        </div>
                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <DecisionBadge status={primaryOpportunity.intel.decisionStatus} size="md" />
                            <ConfidenceBadge level={primaryOpportunity.intel.confidenceLevel} />
                            <Badge size="sm" caps={false}>
                              {primaryOpportunity.player.position}
                            </Badge>
                          </div>
                          <p className="mt-3 text-sm leading-6 text-muted">
                            {primaryOpportunity.player.club} · {primaryOpportunity.player.league} · {primaryOpportunity.intel.roleFitLabel}
                          </p>
                          <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-300">{primaryOpportunity.player.summary}</p>
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <ValueGapBadge valueGapM={primaryOpportunity.intel.valueGapM} valueGapPct={primaryOpportunity.intel.valueGapPct} />
                        <StatusTag label={primaryOpportunity.intel.priceRealism} />
                      </div>
                    </div>

                    <div className="rounded-[24px] bg-panel-2/70 p-5">
                      <p className="text-label">Next action</p>
                      <p className="mt-2 text-2xl font-semibold text-text">{primaryOpportunity.intel.nextAction}</p>
                      <p className="mt-3 text-sm leading-6 text-slate-300">{primaryOpportunity.intel.decisionReason}</p>
                    </div>

                    <div className="grid gap-3 sm:grid-cols-4">
                      <div className="rounded-[20px] bg-white/[0.03] p-4">
                        <p className="text-label">Market</p>
                        <p className="mt-2 text-xl font-semibold text-text">{formatCurrencyMillions(primaryOpportunity.player.marketValueM)}</p>
                      </div>
                      <div className="rounded-[20px] bg-white/[0.03] p-4">
                        <p className="text-label">Predicted</p>
                        <p className="mt-2 text-xl font-semibold text-text">{formatCurrencyMillions(primaryOpportunity.intel.predictedValueM)}</p>
                      </div>
                      <div className="rounded-[20px] bg-white/[0.03] p-4">
                        <p className="text-label">Owner</p>
                        <p className="mt-2 text-xl font-semibold text-text">{primaryOpportunity.intel.watchlistOwner}</p>
                      </div>
                      <div className="rounded-[20px] bg-white/[0.03] p-4">
                        <p className="text-label">Readiness</p>
                        <p className="mt-2 text-xl font-semibold text-text">{primaryOpportunity.intel.readiness}</p>
                      </div>
                    </div>

                    <div className="rounded-[24px] bg-white/[0.03] p-5">
                      <p className="text-label">Why now</p>
                      <p className="mt-3 text-sm leading-6 text-slate-200">{primaryOpportunity.intel.reliabilityNote}</p>
                      <p className="mt-3 text-sm leading-6 text-muted">{primaryOpportunity.intel.contractUrgency}</p>
                    </div>
                  </div>

                  <div className="space-y-5 rounded-[28px] bg-white/[0.03] p-5">
                    <div className="flex items-start justify-between gap-3">
                      <SectionHeader
                        eyebrow="Action Queue"
                        title="Resolve these next"
                        description="The next cases that deserve either live follow-up or a board-level check."
                      />
                      <Link href="/discovery">
                        <Button variant="ghost" className="gap-2">
                          Open discovery
                          <ArrowRight className="size-4" />
                        </Button>
                      </Link>
                    </div>

                    <div className="space-y-3">
                      {supportingOpportunities.map(({ player, intel }) => (
                        <div key={player.id} className="rounded-[22px] bg-panel-2/65 p-4">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="font-semibold text-text">{player.name}</p>
                              <p className="mt-1 text-sm text-muted">
                                {player.club} · {player.position} · {intel.roleFitLabel}
                              </p>
                            </div>
                            <DecisionBadge status={intel.decisionStatus} />
                          </div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            <ValueGapBadge valueGapM={intel.valueGapM} valueGapPct={intel.valueGapPct} compact />
                            <ConfidenceBadge level={intel.confidenceLevel} compact />
                          </div>
                          <p className="mt-3 text-sm leading-6 text-slate-300">{intel.nextAction}</p>
                        </div>
                      ))}
                    </div>

                    <div className="border-t border-white/[0.06] pt-5">
                      <SectionHeader eyebrow="Assignments" title="What blocks the next decision" description="Keep operational work visible around the live cases." />
                      <div className="mt-4 space-y-3">
                        {taskBoard.map((task) => (
                          <div key={task.title} className="rounded-[22px] bg-panel-2/65 p-4">
                            <div className="flex items-center justify-between gap-3">
                              <p className="font-medium text-text">{task.title}</p>
                              <StatusTag label={task.priority} />
                            </div>
                            <div className="mt-3 flex items-center justify-between text-sm text-muted">
                              <span>{task.owner}</span>
                              <span>{task.due}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </>
              ) : null}
            </CardContent>
          </Card>

          <div className="grid gap-5 lg:grid-cols-2">
            <ChartCard title="Performance trend" description="Momentum versus the active board benchmark.">
              <PerformanceTrendChart data={performanceTrendData} />
            </ChartCard>
            <ChartCard title="Role distribution" description="Where current live cases sit across the squad map.">
              <PositionDistributionChart data={positionDistributionData} />
            </ChartCard>
          </div>
        </div>

        <div className="space-y-5 xl:col-span-4">
          <Card>
            <CardContent className="space-y-5 p-6">
              <SectionHeader eyebrow="Market movement" title="Names moving inside the workflow" description="Who is gaining relevance and why the case changed." />
              <div className="space-y-3">
                {trendingProfiles.map(({ player, intel }) => (
                  <div key={player.id} className="rounded-[22px] bg-white/[0.03] p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="font-semibold text-text">{player.name}</p>
                        <p className="mt-1 text-sm text-muted">
                          {player.club} · {player.position}
                        </p>
                      </div>
                      <StatusTag label={player.form} />
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <DecisionBadge status={intel.decisionStatus} />
                      <ConfidenceBadge level={intel.confidenceLevel} compact />
                    </div>
                    <p className="mt-3 text-sm leading-6 text-muted">{intel.decisionReason}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="space-y-5 p-6">
              <SectionHeader eyebrow="Workflow changes" title="Recent internal movement" description="Reports, assignments, and notes that changed the board recently." />
              <div className="space-y-3">
                {dashboardActivity.map((item) => (
                  <div key={`${item.title}-${item.time}`} className="rounded-[22px] bg-panel-2/65 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <StatusTag label={item.type} />
                      <span className="text-xs text-muted">{item.time}</span>
                    </div>
                    <p className="mt-3 text-sm font-semibold text-text">{item.title}</p>
                    <p className="mt-2 text-sm leading-6 text-muted">{item.subtitle}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-12">
        <div className="space-y-5 xl:col-span-7">
          <SectionHeader
            eyebrow="Role leaders"
            title="Best-supported cases by role"
            description="Use these as the fastest route from squad need to live player review."
          />
          <div className="grid gap-5 md:grid-cols-2">
            {roleLeaders.map(({ player }) => (
              <PlayerCard key={player.id} player={player} />
            ))}
          </div>
        </div>
        <div className="xl:col-span-5">
          <ChartCard title="Age vs market value" description="A secondary lens for age runway versus fee level.">
            <AgeValueScatter data={ageValueScatterData} />
          </ChartCard>
        </div>
      </div>
    </div>
  );
}
