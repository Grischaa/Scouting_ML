import { MessageSquareQuote } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export function ScoutNoteCard({ author, note, time }: { author: string; note: string; time: string }) {
  return (
    <Card className="h-full bg-white/[0.03]">
      <CardContent className="space-y-4 p-4">
        <div className="flex items-center gap-3 text-sm text-muted">
          <div className="flex size-9 items-center justify-center rounded-xl border border-white/8 bg-white/[0.03] text-blue">
            <MessageSquareQuote className="size-4" />
          </div>
          <div>
            <p className="font-medium text-slate-200">{author}</p>
            <p className="text-xs">{time}</p>
          </div>
        </div>
        <p className="text-sm leading-6 text-slate-300">{note}</p>
      </CardContent>
    </Card>
  );
}
