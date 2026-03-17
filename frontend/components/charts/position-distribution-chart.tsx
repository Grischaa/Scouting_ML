"use client";

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

const colors = ["#2EC27E", "#4EA1FF", "#F4B740", "#7DD3FC", "#34D399", "#93C5FD"];

export function PositionDistributionChart({ data }: { data: Array<{ position: string; value: number }> }) {
  return (
    <div className="grid gap-4 lg:grid-cols-[240px_minmax(0,1fr)] lg:items-center">
      <div className="h-[240px]">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie innerRadius={62} outerRadius={92} paddingAngle={3} data={data} dataKey="value" nameKey="position">
              {data.map((entry, index) => (
                <Cell key={entry.position} fill={colors[index % colors.length]} />
              ))}
            </Pie>
            <Tooltip contentStyle={{ background: "#12192B", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16 }} />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="space-y-3">
        {data.map((entry, index) => (
          <div key={entry.position} className="flex items-center justify-between rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3 text-sm">
            <div className="flex items-center gap-3">
              <span className="size-3 rounded-full" style={{ backgroundColor: colors[index % colors.length] }} />
              <span className="text-slate-200">{entry.position}</span>
            </div>
            <span className="font-medium text-text">{entry.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
