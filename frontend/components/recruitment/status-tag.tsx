import { Badge } from "@/components/ui/badge";

const toneMap: Record<string, "green" | "blue" | "amber" | "red" | "neutral"> = {
  priority: "green",
  shortlist: "blue",
  monitor: "amber",
  watch: "neutral",
  "Ready now": "green",
  "Rotation-ready": "blue",
  Developmental: "amber",
  Undervalued: "green",
  "Fair value": "blue",
  "Stretch fee": "red",
  Rising: "green",
  Stable: "blue",
  Cooling: "amber",
  note: "blue",
  report: "green",
  assignment: "amber",
  Live: "blue",
  Video: "amber",
  Model: "green",
  Critical: "red",
  Exploratory: "amber",
  High: "red",
  Medium: "amber",
  Low: "blue",
};

export function StatusTag({
  label,
  size = "sm",
}: {
  label: string;
  size?: "sm" | "md";
}) {
  return (
    <Badge tone={toneMap[label] ?? "neutral"} size={size} caps={false}>
      {label}
    </Badge>
  );
}
