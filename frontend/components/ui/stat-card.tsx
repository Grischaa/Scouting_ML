import { ArrowUpRight, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const toneClasses = {
  green: "from-green/18 to-transparent text-green",
  blue: "from-blue/18 to-transparent text-blue",
  amber: "from-amber/18 to-transparent text-amber",
};

export function StatCard({
  label,
  value,
  delta,
  tone = "green",
}: {
  label: string;
  value: string;
  delta: string;
  tone?: keyof typeof toneClasses;
}) {
  return (
    <motion.div whileHover={{ y: -3 }} transition={{ duration: 0.22 }}>
      <Card className="overflow-hidden">
        <CardContent className="relative p-5">
          <div className={cn("absolute inset-x-0 top-0 h-20 bg-gradient-to-b", toneClasses[tone])} />
          <div className="relative flex items-start justify-between gap-4">
            <div className="space-y-3">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted">{label}</p>
              <div className="space-y-2">
                <p className="text-3xl font-semibold tracking-tight text-text">{value}</p>
                <div className="inline-flex items-center gap-2 rounded-full border border-white/8 bg-white/5 px-3 py-1 text-xs text-slate-300">
                  <ArrowUpRight className="size-3.5 text-green" />
                  {delta}
                </div>
              </div>
            </div>
            <div className="flex size-11 items-center justify-center rounded-2xl border border-white/10 bg-panel-2/85 text-slate-100">
              <Sparkles className="size-5" />
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
