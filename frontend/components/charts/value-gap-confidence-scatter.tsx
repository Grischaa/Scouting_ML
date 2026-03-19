"use client";

import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";

export function ValueGapConfidenceScatter({
  data,
}: {
  data: Array<{ name: string; valueGap: number; confidence: number; scoutingScore: number; league: string; position: string }>;
}) {
  return (
    <div className="h-[320px]">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 12, bottom: 12, left: 0 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
          <XAxis
            type="number"
            dataKey="valueGap"
            tick={{ fill: "#94A3B8", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            unit="m"
          />
          <YAxis
            type="number"
            dataKey="confidence"
            tick={{ fill: "#94A3B8", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            domain={[60, 95]}
          />
          <ZAxis type="number" dataKey="scoutingScore" range={[90, 320]} />
          <Tooltip
            cursor={{ strokeDasharray: "4 4", stroke: "rgba(255,255,255,0.12)" }}
            contentStyle={{ background: "#12192B", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 18 }}
            formatter={(value: number, key: string) => {
              if (key === "valueGap") return [`€${value.toFixed(1)}m`, "Value gap"];
              if (key === "confidence") return [value, "Confidence"];
              return [value, key];
            }}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.name}
          />
          <Scatter data={data} fill="#2EC27E" fillOpacity={0.86} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
