"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, CheckCircle2, ShieldCheck, Sparkles, Target } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const highlights = [
  "Outside-Big-5 opportunity boards tuned for real recruitment decisions",
  "Role fit, price discipline, and report-ready player profiles in one workflow",
  "Designed for scouts, analysts, sporting directors, and consultant teams",
];

const quickStats = [
  { label: "Live profiles", value: "1.2k" },
  { label: "Shortlists", value: "14" },
  { label: "Reports shipped", value: "46" },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-bg px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto grid min-h-[calc(100vh-3rem)] max-w-7xl gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <Card className="relative overflow-hidden border-white/10 bg-hero">
          <CardContent className="flex h-full flex-col justify-between p-8 sm:p-12">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs uppercase tracking-[0.24em] text-green">
                <ShieldCheck className="size-4" />
                ScoutML Platform
              </div>
              <motion.h1
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45 }}
                className="mt-8 max-w-3xl text-4xl font-semibold leading-tight sm:text-6xl"
              >
                Smarter football scouting, powered by data.
              </motion.h1>
              <motion.p
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, delay: 0.08 }}
                className="mt-6 max-w-2xl text-base leading-8 text-slate-300 sm:text-lg"
              >
                A premium scouting workspace built to help clubs and consultants identify undervalued players, compare profiles, and ship better recruitment decisions faster.
              </motion.p>

              <div className="mt-10 grid gap-3 sm:grid-cols-3">
                {quickStats.map((item) => (
                  <div key={item.label} className="rounded-[22px] border border-white/10 bg-white/[0.04] p-4 backdrop-blur">
                    <p className="text-label">{item.label}</p>
                    <p className="mt-3 text-3xl font-semibold text-text">{item.value}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
              <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-5">
                <p className="text-label">Why clubs use it</p>
                <div className="mt-4 space-y-4">
                  {highlights.map((item) => (
                    <div key={item} className="flex items-start gap-3">
                      <CheckCircle2 className="mt-0.5 size-5 text-green" />
                      <p className="text-sm leading-6 text-slate-300">{item}</p>
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded-[24px] border border-white/10 bg-[#0d1628]/90 p-5">
                <p className="text-label">Product focus</p>
                <div className="mt-5 space-y-4 text-sm text-slate-300">
                  <div className="flex items-center gap-3"><Target className="size-4 text-blue" /> Recruitment intelligence</div>
                  <div className="flex items-center gap-3"><Sparkles className="size-4 text-green" /> Premium club reporting</div>
                  <div className="flex items-center gap-3"><ShieldCheck className="size-4 text-amber" /> Decision confidence layers</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-white/10 bg-panel/92">
          <CardContent className="flex h-full flex-col justify-center p-8 sm:p-10">
            <div className="mb-8">
              <p className="text-label">Welcome back</p>
              <h2 className="mt-3 text-3xl font-semibold text-text">Sign in to your recruitment workspace</h2>
              <p className="mt-3 text-sm leading-7 text-muted">
                This demo uses mock data only, but the interface is designed to feel like a real premium internal scouting platform.
              </p>
            </div>
            <form className="space-y-4">
              <label className="block space-y-2">
                <span className="text-label">Club email</span>
                <input className="h-12 w-full rounded-2xl border border-white/10 bg-panel-2/80 px-4 text-sm text-text outline-none transition focus:border-blue/60" placeholder="name@club.com" />
              </label>
              <label className="block space-y-2">
                <span className="text-label">Password</span>
                <input type="password" className="h-12 w-full rounded-2xl border border-white/10 bg-panel-2/80 px-4 text-sm text-text outline-none transition focus:border-blue/60" placeholder="••••••••" />
              </label>
              <div className="flex items-center justify-between text-sm text-muted">
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="size-4 rounded border-white/10 bg-panel-2" />
                  Keep me signed in
                </label>
                <button type="button" className="text-slate-200">Forgot password?</button>
              </div>
              <Link
                href="/dashboard"
                className="mt-4 inline-flex h-11 w-full items-center justify-center gap-2 rounded-2xl bg-green px-5 text-sm font-medium text-slate-950 shadow-[0_10px_24px_rgba(46,194,126,0.24)] transition-all duration-200 hover:bg-green/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue/70 focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
              >
                Open platform
                <ArrowRight className="size-4" />
              </Link>
            </form>
            <div className="mt-8 rounded-[22px] border border-white/8 bg-white/[0.03] p-4">
              <p className="text-label">Platform preview</p>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-white/8 bg-[#111a2d] p-4">
                  <p className="text-sm font-semibold text-text">12 live shortlists</p>
                  <p className="mt-2 text-sm leading-6 text-muted">U23 midfielders, left-backs with progression, and transition forwards outside the Big Five.</p>
                </div>
                <div className="rounded-2xl border border-white/8 bg-[#111a2d] p-4">
                  <p className="text-sm font-semibold text-text">46 internal reports</p>
                  <p className="mt-2 text-sm leading-6 text-muted">Reusable block-based reports for sporting directors and recruitment committees.</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
