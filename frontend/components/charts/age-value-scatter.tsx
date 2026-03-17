"use client";

import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";

export function AgeValueScatter({ data }: { data: Array<{ name: string; age: number; value: number; score: number; league: string }> }) {
  return (
    <div className="h-[320px]">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 12, right: 12, bottom: 12, left: 0 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
          <XAxis type="number" dataKey="age" tick={{ fill: "#94A3B8", fontSize: 12 }} axisLine={false} tickLine={false} domain={[18, 25]} />
          <YAxis type="number" dataKey="value" tick={{ fill: "#94A3B8", fontSize: 12 }} axisLine={false} tickLine={false} unit="m" />
          <ZAxis type="number" dataKey="score" range={[90, 320]} />
          <Tooltip
            cursor={{ strokeDasharray: "4 4", stroke: "rgba(255,255,255,0.12)" }}
            contentStyle={{ background: "#12192B", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16 }}
            formatter={(value: number, name: string) => [name === "value" ? `€${value}m` : value, name === "value" ? "Market value" : "Scouting score"]}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.name}
          />
          <Scatter data={data} fill="#4EA1FF" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
