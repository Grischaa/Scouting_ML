"use client";

import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer } from "recharts";
import { ChartCard } from "@/components/charts/chart-card";

export function RadarChartCard({ title, data }: { title: string; data: Array<{ subject: string; value: number }> }) {
  return (
    <ChartCard title={title} description="Role-shape profile against the internal scouting model.">
      <div className="h-[290px]">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data} outerRadius="70%">
            <PolarGrid stroke="rgba(255,255,255,0.1)" />
            <PolarAngleAxis tick={{ fill: "#94A3B8", fontSize: 11 }} dataKey="subject" />
            <Radar dataKey="value" stroke="#4EA1FF" fill="#4EA1FF" fillOpacity={0.3} strokeWidth={2.5} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}
