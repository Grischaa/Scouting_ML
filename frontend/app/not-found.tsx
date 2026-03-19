import Link from "next/link";
import { ArrowLeft, SearchX } from "lucide-react";

export default function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-bg px-6">
      <div className="w-full max-w-2xl rounded-[32px] border border-white/8 bg-panel/90 p-8 shadow-panel backdrop-blur-xl sm:p-10">
        <div className="mx-auto flex size-16 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] text-blue">
          <SearchX className="size-7" />
        </div>
        <div className="mt-6 text-center">
          <p className="text-label">Page not found</p>
          <h1 className="mt-3 text-4xl font-semibold text-text">This player or workspace view does not exist.</h1>
          <p className="mx-auto mt-4 max-w-xl text-sm leading-7 text-muted">
            The requested page is outside the current mock dataset. Return to the dashboard or go back to the player database.
          </p>
        </div>
        <div className="mt-8 flex flex-col justify-center gap-3 sm:flex-row">
          <Link
            href="/dashboard"
            className="inline-flex h-11 items-center justify-center rounded-2xl bg-green px-5 text-sm font-medium text-slate-950 shadow-[0_10px_24px_rgba(46,194,126,0.24)] transition-all duration-200 hover:bg-green/90"
          >
            Back to dashboard
          </Link>
          <Link
            href="/discovery"
            className="inline-flex h-11 items-center justify-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-5 text-sm font-medium text-slate-100 transition-all duration-200 hover:bg-white/10"
          >
            <ArrowLeft className="size-4" />
            Open discovery
          </Link>
        </div>
      </div>
    </div>
  );
}
