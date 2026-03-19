"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { ArrowDown, ArrowUp, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";

type SortDirection = "asc" | "desc";

export interface TableColumn<T> {
  key: keyof T | string;
  header: string;
  align?: "left" | "right";
  render: (row: T) => React.ReactNode;
  sortAccessor?: (row: T) => number | string;
  className?: string;
}

export function DataTable<T>({
  columns,
  data,
  rowKey,
  className,
  toolbar,
  defaultSortKey,
  defaultSortDirection = "desc",
  selectedRowKey,
  onRowClick,
  rowClassName,
}: {
  columns: TableColumn<T>[];
  data: T[];
  rowKey: (row: T) => string;
  className?: string;
  toolbar?: React.ReactNode;
  defaultSortKey?: string;
  defaultSortDirection?: SortDirection;
  selectedRowKey?: string | null;
  onRowClick?: (row: T) => void;
  rowClassName?: (row: T) => string | undefined;
}) {
  const [sort, setSort] = useState<{ key: string; direction: SortDirection } | null>(
    defaultSortKey ? { key: defaultSortKey, direction: defaultSortDirection } : null,
  );

  const sortedData = useMemo(() => {
    if (!sort) return data;

    const column = columns.find((item) => String(item.key) === sort.key && item.sortAccessor);
    if (!column?.sortAccessor) return data;

    return [...data].sort((left, right) => {
      const leftValue = column.sortAccessor?.(left);
      const rightValue = column.sortAccessor?.(right);

      if (typeof leftValue === "number" && typeof rightValue === "number") {
        return sort.direction === "asc" ? leftValue - rightValue : rightValue - leftValue;
      }

      const comparison = String(leftValue).localeCompare(String(rightValue), undefined, { numeric: true, sensitivity: "base" });
      return sort.direction === "asc" ? comparison : -comparison;
    });
  }, [columns, data, sort]);

  return (
    <div className={cn("overflow-hidden rounded-[26px] border border-white/[0.07] bg-panel-2/45", className)}>
      {toolbar ? <div className="border-b border-white/[0.06] px-5 py-3.5">{toolbar}</div> : null}
      <div className="overflow-x-auto">
        <table className="min-w-full border-separate border-spacing-0">
          <thead>
            <tr className="bg-white/[0.025] text-left">
              {columns.map((column) => {
                const isSortable = Boolean(column.sortAccessor);
                const isActive = sort?.key === String(column.key);
                const SortIcon = !isActive ? ChevronsUpDown : sort?.direction === "asc" ? ArrowUp : ArrowDown;

                return (
                  <th
                    key={String(column.key)}
                    className={cn(
                      "sticky top-0 z-[1] whitespace-nowrap border-b border-white/[0.06] px-5 py-3.5 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted backdrop-blur",
                      column.align === "right" && "text-right",
                    )}
                  >
                    <button
                      type="button"
                      className={cn(
                        "inline-flex items-center gap-1.5 transition",
                        column.align === "right" && "ml-auto",
                        !isSortable && "cursor-default",
                        isSortable && "hover:text-slate-200",
                      )}
                      onClick={() => {
                        if (!isSortable) return;

                        setSort((current) => {
                          if (!current || current.key !== String(column.key)) {
                            return { key: String(column.key), direction: defaultSortDirection };
                          }

                          return {
                            key: current.key,
                            direction: current.direction === "desc" ? "asc" : "desc",
                          };
                        });
                      }}
                    >
                      {column.header}
                      {isSortable ? <SortIcon className={cn("size-3.5", isActive ? "opacity-100" : "opacity-45")} /> : null}
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sortedData.map((row, index) => {
              const key = rowKey(row);
              const isSelected = selectedRowKey === key;

              return (
                <motion.tr
                  key={key}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2, delay: index * 0.018 }}
                  onClick={() => onRowClick?.(row)}
                  className={cn(
                    "group transition-colors",
                    onRowClick && "cursor-pointer",
                    isSelected ? "bg-blue/10" : "hover:bg-white/[0.025]",
                    rowClassName?.(row),
                  )}
                >
                  {columns.map((column) => (
                    <td
                      key={`${key}-${String(column.key)}`}
                      className={cn(
                        "border-b border-white/[0.05] px-5 py-[18px] text-sm leading-6 text-slate-200",
                        isSelected && "first:shadow-[inset_3px_0_0_0_rgba(78,161,255,0.9)]",
                        column.align === "right" && "text-right",
                        column.className,
                      )}
                    >
                      {column.render(row)}
                    </td>
                  ))}
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between border-t border-white/[0.06] px-5 py-3 text-xs text-muted">
        <span>{sortedData.length} visible rows</span>
        <span>Sortable columns available where relevant</span>
      </div>
    </div>
  );
}
