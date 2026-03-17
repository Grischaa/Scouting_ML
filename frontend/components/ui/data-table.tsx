import { motion } from "framer-motion";
import { ChevronDown, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";

export interface TableColumn<T> {
  key: keyof T | string;
  header: string;
  align?: "left" | "right";
  render: (row: T) => React.ReactNode;
}

export function DataTable<T>({
  columns,
  data,
  rowKey,
  className,
}: {
  columns: TableColumn<T>[];
  data: T[];
  rowKey: (row: T) => string;
  className?: string;
}) {
  return (
    <div className={cn("overflow-hidden rounded-[22px] border border-white/8 bg-panel-2/50", className)}>
      <div className="overflow-x-auto">
        <table className="min-w-full border-separate border-spacing-0">
          <thead>
            <tr className="bg-white/[0.03] text-left">
              {columns.map((column) => (
                <th
                  key={String(column.key)}
                  className={cn(
                    "sticky top-0 z-[1] whitespace-nowrap border-b border-white/8 px-4 py-3 text-[11px] font-semibold uppercase tracking-[0.22em] text-muted backdrop-blur",
                    column.align === "right" && "text-right",
                  )}
                >
                  <span className="inline-flex items-center gap-1.5">
                    {column.header}
                    <ChevronsUpDown className="size-3.5 opacity-50" />
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, index) => (
              <motion.tr
                key={rowKey(row)}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, delay: index * 0.025 }}
                className="group transition-colors hover:bg-white/[0.03]"
              >
                {columns.map((column) => (
                  <td
                    key={`${rowKey(row)}-${String(column.key)}`}
                    className={cn(
                      "border-b border-white/6 px-4 py-4 text-sm text-slate-200",
                      column.align === "right" && "text-right",
                    )}
                  >
                    {column.render(row)}
                  </td>
                ))}
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between border-t border-white/8 px-4 py-3 text-xs text-muted">
        <span>{data.length} visible rows</span>
        <button className="inline-flex items-center gap-1 text-slate-300">
          More controls <ChevronDown className="size-3.5" />
        </button>
      </div>
    </div>
  );
}
