import { cn } from "@/lib/utils";

export function SectionHeader({
  eyebrow,
  title,
  description,
  action,
  className,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between", className)}>
      <div className="space-y-1.5">
        {eyebrow ? <p className="text-[10px] font-semibold uppercase tracking-[0.26em] text-blue">{eyebrow}</p> : null}
        <div className="space-y-1">
          <h2 className="text-[1.3rem] font-semibold tracking-tight text-text sm:text-[1.45rem]">{title}</h2>
          {description ? <p className="max-w-2xl text-sm leading-6 text-muted">{description}</p> : null}
        </div>
      </div>
      {action ? <div className="shrink-0 pt-1">{action}</div> : null}
    </div>
  );
}
