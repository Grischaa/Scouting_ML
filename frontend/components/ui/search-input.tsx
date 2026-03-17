import { Search } from "lucide-react";
import { cn } from "@/lib/utils";

export function SearchInput({ className, placeholder = "Search players, clubs, leagues..." }: { className?: string; placeholder?: string }) {
  return (
    <div className={cn("flex items-center gap-3 rounded-2xl border border-white/10 bg-panel-2/80 px-4 py-3 text-sm text-muted", className)}>
      <Search className="size-4 text-blue" />
      <input
        className="w-full bg-transparent text-sm text-text outline-none placeholder:text-muted"
        placeholder={placeholder}
        type="search"
      />
      <span className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-[11px] uppercase tracking-[0.18em] text-muted">/</span>
    </div>
  );
}
