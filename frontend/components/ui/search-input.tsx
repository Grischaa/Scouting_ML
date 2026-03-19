import { Command, Search } from "lucide-react";
import { cn } from "@/lib/utils";

export function SearchInput({ className, placeholder = "Search players, clubs, leagues, archetypes..." }: { className?: string; placeholder?: string }) {
  return (
    <div
      className={cn(
        "flex items-center gap-3 rounded-[22px] border border-white/10 bg-panel-2/82 px-4 py-3 text-sm text-muted shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] transition hover:border-white/14 focus-within:border-blue/40",
        className,
      )}
    >
      <div className="flex size-9 items-center justify-center rounded-xl border border-white/8 bg-white/[0.04] text-blue">
        <Search className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted">Command Search</p>
        <input
          className="mt-1 w-full bg-transparent text-sm text-text outline-none placeholder:text-muted"
          placeholder={placeholder}
          type="search"
        />
      </div>
      <span className="inline-flex items-center gap-1 rounded-xl border border-white/10 bg-white/[0.04] px-2.5 py-1.5 text-[11px] uppercase tracking-[0.18em] text-muted">
        <Command className="size-3.5" />
        K
      </span>
    </div>
  );
}
