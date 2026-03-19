import { Badge } from "@/components/ui/badge";
import type { ConfidenceLevel } from "@/lib/types";

const toneMap: Record<ConfidenceLevel, "green" | "amber" | "red"> = {
  "High confidence": "green",
  Caution: "amber",
  "Thin evidence": "red",
};

export function ConfidenceBadge({
  level,
  score,
  compact = false,
  showScore = false,
}: {
  level: ConfidenceLevel;
  score?: number;
  compact?: boolean;
  showScore?: boolean;
}) {
  const label = showScore && typeof score === "number" ? `${level} · ${score}` : level;

  return (
    <Badge tone={toneMap[level]} size={compact ? "sm" : "md"} caps={false}>
      {compact ? label : `Confidence: ${label}`}
    </Badge>
  );
}
