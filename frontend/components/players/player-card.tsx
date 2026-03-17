import Link from "next/link";
import { ArrowUpRight, Clock3 } from "lucide-react";
import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { Player } from "@/lib/types";
import { formatCurrencyMillions, initials } from "@/lib/utils";

export function PlayerCard({ player }: { player: Player }) {
  return (
    <motion.div whileHover={{ y: -4 }} transition={{ duration: 0.2 }}>
      <Link href={`/players/${player.slug}`}>
        <Card className="overflow-hidden">
          <CardContent className="space-y-5 p-5">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="flex size-12 items-center justify-center rounded-2xl bg-gradient-to-br from-green/20 to-blue/20 text-sm font-semibold text-text">
                  {initials(player.name)}
                </div>
                <div>
                  <h3 className="text-base font-semibold text-text">{player.name}</h3>
                  <p className="text-sm text-muted">{player.club} · {player.league}</p>
                </div>
              </div>
              <Badge tone={player.status === "priority" ? "green" : player.status === "shortlist" ? "blue" : "amber"}>{player.status}</Badge>
            </div>

            <div className="grid grid-cols-2 gap-3 rounded-[20px] border border-white/8 bg-white/[0.03] p-3">
              <div>
                <p className="text-[11px] uppercase tracking-[0.18em] text-muted">Score</p>
                <p className="mt-2 text-2xl font-semibold text-text">{player.scoutingScore}</p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-[0.18em] text-muted">Value</p>
                <p className="mt-2 text-2xl font-semibold text-text">{formatCurrencyMillions(player.marketValueM)}</p>
              </div>
            </div>

            <p className="text-sm leading-6 text-muted">{player.summary}</p>

            <div className="flex items-center justify-between text-sm text-muted">
              <div className="inline-flex items-center gap-2">
                <Clock3 className="size-4 text-blue" />
                {player.minutes.toLocaleString("en-GB")} mins
              </div>
              <div className="inline-flex items-center gap-2 text-slate-200">
                Open profile
                <ArrowUpRight className="size-4" />
              </div>
            </div>
          </CardContent>
        </Card>
      </Link>
    </motion.div>
  );
}
