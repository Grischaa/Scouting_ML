"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function PerformanceTrendChart({ data }: { data: Array<{ label: string; scouting: number; benchmark: number }> }) {
  return (
    <div className="h-[320px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: -16, bottom: 0 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
          <XAxis dataKey="label" tick={{ fill: "#94A3B8", fontSize: 12 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: "#94A3B8", fontSize: 12 }} axisLine={false} tickLine={false} domain={[60, 90]} />
          <Tooltip contentStyle={{ background: "#12192B", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16 }} />
          <Line type="monotone" dataKey="benchmark" stroke="rgba(148,163,184,0.6)" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="scouting" stroke="#2EC27E" strokeWidth={3} dot={{ fill: "#2EC27E", r: 4 }} activeDot={{ r: 6 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
