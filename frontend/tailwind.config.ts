import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0B1020",
        panel: "#12192B",
        "panel-2": "#182235",
        "panel-3": "#1F2B40",
        border: "rgba(255,255,255,0.08)",
        text: "#F5F7FA",
        muted: "#94A3B8",
        ink: "#0E1527",
        green: "#2EC27E",
        blue: "#4EA1FF",
        amber: "#F4B740",
        red: "#FF6B6B"
      },
      boxShadow: {
        panel: "0 18px 45px rgba(0, 0, 0, 0.34)",
        glow: "0 0 0 1px rgba(255,255,255,0.04), 0 14px 32px rgba(78,161,255,0.12)",
        surface: "0 1px 0 rgba(255,255,255,0.04) inset, 0 24px 60px rgba(0,0,0,0.34)",
        lift: "0 22px 46px rgba(3,8,20,0.52)"
      },
      borderRadius: {
        xl2: "1.35rem"
      },
      backgroundImage: {
        hero: "radial-gradient(circle at 12% 12%, rgba(46,194,126,0.18), transparent 24%), radial-gradient(circle at 82% 16%, rgba(78,161,255,0.18), transparent 22%), linear-gradient(145deg, rgba(18,25,43,0.96), rgba(11,16,32,0.98))",
        "hero-grid": "linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px)",
        "pitch-glow": "radial-gradient(circle at top, rgba(78,161,255,0.12), transparent 36%), radial-gradient(circle at bottom left, rgba(46,194,126,0.1), transparent 28%)"
      }
    }
  },
  plugins: []
};

export default config;
