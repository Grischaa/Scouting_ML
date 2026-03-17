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
        border: "rgba(255,255,255,0.08)",
        text: "#F5F7FA",
        muted: "#94A3B8",
        green: "#2EC27E",
        blue: "#4EA1FF",
        amber: "#F4B740",
        red: "#FF6B6B"
      },
      boxShadow: {
        panel: "0 18px 45px rgba(0, 0, 0, 0.34)",
        glow: "0 0 0 1px rgba(255,255,255,0.04), 0 14px 32px rgba(78,161,255,0.12)"
      },
      borderRadius: {
        xl2: "1.35rem"
      },
      backgroundImage: {
        hero: "radial-gradient(circle at top left, rgba(46,194,126,0.16), transparent 24%), radial-gradient(circle at 85% 18%, rgba(78,161,255,0.16), transparent 24%), linear-gradient(145deg, rgba(18,25,43,0.94), rgba(11,16,32,0.98))"
      }
    }
  },
  plugins: []
};

export default config;
