import { MessageSquareQuote } from "lucide-react";
import { StatusTag } from "@/components/recruitment/status-tag";
import { Card, CardContent } from "@/components/ui/card";

export function ScoutNoteCard({
  author,
  note,
  time,
  type = "Live",
}: {
  author: string;
  note: string;
  time: string;
  type?: "Live" | "Video" | "Model";
}) {
  return (
    <Card className="h-full bg-white/[0.03]">
      <CardContent className="space-y-4 p-4">
        <div className="flex items-start justify-between gap-3 text-sm text-muted">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-xl border border-white/8 bg-white/[0.03] text-blue">
              <MessageSquareQuote className="size-4" />
            </div>
            <div>
              <p className="font-medium text-slate-200">{author}</p>
              <p className="text-xs">{time}</p>
            </div>
          </div>
          <StatusTag label={type} />
        </div>
        <div className="rounded-[18px] border border-white/8 bg-panel-2/65 p-4">
          <p className="text-sm leading-6 text-slate-300">{note}</p>
        </div>
      </CardContent>
    </Card>
  );
}
