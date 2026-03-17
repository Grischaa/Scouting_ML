"use client";

import { motion } from "framer-motion";
import { CalendarRange, ChevronRight, Flame, ListTodo, Radar, TimerReset } from "lucide-react";
import { AgeValueScatter } from "@/components/charts/age-value-scatter";
import { ChartCard } from "@/components/charts/chart-card";
import { PerformanceTrendChart } from "@/components/charts/performance-trend-chart";
import { PositionDistributionChart } from "@/components/charts/position-distribution-chart";
import { PlayerCard } from "@/components/players/player-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";
import { StatCard } from "@/components/ui/stat-card";
import {
  activityFeed,
  ageValueScatterData,
  dashboardKpis,
  performanceTrendData,
  positionDistributionData,
  topPerformersByRole,
  trendingPlayers,
  watchlistTasks,
} from "@/lib/mock-data";

const activityTone = {
  note: "blue",
  report: "green",
  assignment: "amber",
} as const;

export default function DashboardPage() {
  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow="Dashboard"
        title="Scouting command centre"
        description="Track value discipline, momentum in live boards, and the players most likely to move from watchlist to actionable recruitment targets."
        action={
          <div className="flex flex-wrap gap-3">
            <Button variant="outline" className="gap-2"><CalendarRange className="size-4" />Last 30 days</Button>
            <Button variant="secondary" className="gap-2"><Radar className="size-4" />Generate review pack</Button>
          </div>
        }
      />

      <div className="grid gap-4 xl:grid-cols-4">
        {dashboardKpis.map((kpi, index) => (
          <motion.div key={kpi.label} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.05 }}>
            <StatCard {...kpi} />
          </motion.div>
        ))}
      </div>

      <div className="grid gap-5 xl:grid-cols-12">
        <div className="space-y-5 xl:col-span-8">
          <div className="grid gap-5 lg:grid-cols-12">
            <div className="lg:col-span-7">
              <ChartCard title="Performance trend" description="Internal scouting score versus board benchmark over the active observation window.">
                <PerformanceTrendChart data={performanceTrendData} />
              </ChartCard>
            </div>
            <div className="lg:col-span-5">
              <ChartCard title="Position distribution" description="Current mix of live targets across role clusters.">
                <PositionDistributionChart data={positionDistributionData} />
              </ChartCard>
            </div>
            <div className="lg:col-span-12">
              <ChartCard title="Age vs market value" description="Spot the profiles combining age runway, premium score, and accessible market pricing.">
                <AgeValueScatter data={ageValueScatterData} />
              </ChartCard>
            </div>
          </div>
        </div>

        <div className="space-y-5 xl:col-span-4">
          <Card className="overflow-hidden">
            <CardContent className="space-y-4 p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-label">Trending</p>
                  <h3 className="mt-2 text-lg font-semibold text-text">New high-potential players</h3>
                </div>
                <Flame className="size-5 text-amber" />
              </div>
              <div className="space-y-3">
                {trendingPlayers.map((player) => (
                  <div key={player.id} className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="font-medium text-text">{player.name}</p>
                        <p className="mt-1 text-sm text-muted">{player.club} · {player.position}</p>
                      </div>
                      <Badge tone={player.form === "Rising" ? "green" : player.form === "Stable" ? "blue" : "amber"}>{player.form}</Badge>
                    </div>
                    <div className="mt-4 flex items-center justify-between text-sm text-muted">
                      <span>Score {player.scoutingScore}</span>
                      <span>€{player.marketValueM}m</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-label">Tasks</p>
                  <h3 className="mt-2 text-lg font-semibold text-text">Upcoming watchlist actions</h3>
                </div>
                <ListTodo className="size-5 text-blue" />
              </div>
              <div className="mt-4 space-y-3">
                {watchlistTasks.map((task) => (
                  <div key={task.title} className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
                    <div className="flex items-center justify-between gap-3">
                      <p className="font-medium text-text">{task.title}</p>
                      <Badge tone={task.priority === "High" ? "red" : task.priority === "Medium" ? "amber" : "blue"}>{task.priority}</Badge>
                    </div>
                    <div className="mt-3 flex items-center justify-between text-sm text-muted">
                      <span>{task.owner}</span>
                      <span>{task.due}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-12">
        <div className="xl:col-span-8">
          <SectionHeader
            eyebrow="Board"
            title="Top performers by role"
            description="The strongest profiles inside the current model lens, grouped by recruitment role."
            action={<Button variant="ghost" className="gap-2">Open discovery board <ChevronRight className="size-4" /></Button>}
          />
          <div className="mt-5 grid gap-5 md:grid-cols-2">
            {topPerformersByRole.map((player) => (
              <PlayerCard key={player.id} player={player} />
            ))}
          </div>
        </div>
        <div className="xl:col-span-4">
          <Card className="h-full">
            <CardContent className="p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-label">Activity</p>
                  <h3 className="mt-2 text-lg font-semibold text-text">Recent scouting activity</h3>
                </div>
                <TimerReset className="size-5 text-green" />
              </div>
              <div className="mt-5 space-y-4">
                {activityFeed.map((item) => (
                  <div key={`${item.title}-${item.time}`} className="rounded-[20px] border border-white/8 bg-white/[0.03] p-4">
                    <div className="flex items-center justify-between gap-3">
                      <Badge tone={activityTone[item.type]}>{item.type}</Badge>
                      <span className="text-xs text-muted">{item.time}</span>
                    </div>
                    <p className="mt-3 text-sm font-medium text-text">{item.title}</p>
                    <p className="mt-2 text-sm leading-6 text-muted">{item.subtitle}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
