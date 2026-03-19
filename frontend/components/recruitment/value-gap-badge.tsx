import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { formatPercent, formatSignedCurrencyMillions } from "@/lib/utils";

export function ValueGapBadge({
  valueGapM,
  valueGapPct,
  compact = false,
}: {
  valueGapM: number;
  valueGapPct: number;
  compact?: boolean;
}) {
  const positive = valueGapM >= 0;

  return (
    <Badge tone={positive ? "green" : "red"} size={compact ? "sm" : "md"} caps={false} className="gap-1.5">
      {positive ? <ArrowUpRight className="size-3.5" /> : <ArrowDownRight className="size-3.5" />}
      <span>{compact ? formatSignedCurrencyMillions(valueGapM) : `Value ${formatSignedCurrencyMillions(valueGapM)}`}</span>
      <span className="text-[10px] opacity-80">{formatPercent(valueGapPct)}</span>
    </Badge>
  );
}
