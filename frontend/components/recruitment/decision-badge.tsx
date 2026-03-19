import { Badge } from "@/components/ui/badge";
import type { DecisionStatus } from "@/lib/types";

const toneMap: Record<DecisionStatus, "green" | "blue" | "amber" | "neutral" | "red"> = {
  Pursue: "green",
  Review: "blue",
  Watch: "amber",
  "Price Check": "neutral",
  Pass: "red",
};

export function DecisionBadge({
  status,
  size = "sm",
}: {
  status: DecisionStatus;
  size?: "sm" | "md";
}) {
  return (
    <Badge tone={toneMap[status]} size={size} caps={false}>
      {status}
    </Badge>
  );
}
