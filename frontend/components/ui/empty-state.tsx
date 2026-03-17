import { SearchX } from "lucide-react";

export function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex min-h-[220px] flex-col items-center justify-center rounded-[22px] border border-dashed border-white/10 bg-white/[0.03] p-8 text-center">
      <div className="mb-4 flex size-14 items-center justify-center rounded-full border border-white/10 bg-panel-2/80 text-muted">
        <SearchX className="size-6" />
      </div>
      <h3 className="text-lg font-semibold text-text">{title}</h3>
      <p className="mt-2 max-w-md text-sm leading-6 text-muted">{description}</p>
    </div>
  );
}
