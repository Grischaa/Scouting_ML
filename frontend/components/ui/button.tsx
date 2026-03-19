import * as React from "react";
import { cn } from "@/lib/utils";

type ButtonVariant = "default" | "secondary" | "ghost" | "outline" | "panel" | "danger";
type ButtonSize = "sm" | "md" | "lg" | "icon";

const variantClasses: Record<ButtonVariant, string> = {
  default:
    "bg-green text-slate-950 hover:bg-green/90 shadow-[0_16px_30px_rgba(46,194,126,0.24)]",
  secondary: "bg-blue text-white hover:bg-blue/90 shadow-[0_16px_30px_rgba(78,161,255,0.2)]",
  ghost: "bg-transparent text-slate-200 hover:bg-white/[0.06]",
  outline: "border border-white/10 bg-white/[0.04] text-slate-100 hover:border-white/16 hover:bg-white/[0.08]",
  panel: "border border-white/8 bg-panel-2/80 text-slate-100 hover:border-white/14 hover:bg-panel-3/80",
  danger: "bg-red text-white hover:bg-red/90",
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: "h-9 px-3 text-sm",
  md: "h-10 px-4 text-sm",
  lg: "h-11 px-5 text-sm",
  icon: "size-10",
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "md", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center rounded-2xl font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue/70 focus-visible:ring-offset-2 focus-visible:ring-offset-bg disabled:pointer-events-none disabled:opacity-50",
          variantClasses[variant],
          sizeClasses[size],
          className,
        )}
        {...props}
      />
    );
  },
);

Button.displayName = "Button";
