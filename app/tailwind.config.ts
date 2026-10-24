import type { Config } from "tailwindcss";

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: {
          bg: "var(--noc-bg-deep, #090e1a)",
          surface: "var(--noc-bg-panel, #0f172a)",
          elevated: "var(--noc-bg-card, #1e293b)",
          border: "var(--noc-border, rgba(148, 163, 184, 0.14))",
          hover: "var(--noc-bg-hover, rgba(148, 163, 184, 0.08))",
          muted: "var(--noc-text-muted, #94a3b8)",
        },
        noc: {
          deep: "var(--noc-bg-deep, #090e1a)",
          panel: "var(--noc-bg-panel, #0f172a)",
          card: "var(--noc-bg-card, #1e293b)",
          border: "var(--noc-border, rgba(148, 163, 184, 0.14))",
          borderStrong: "var(--noc-border-strong, rgba(148, 163, 184, 0.28))",
          textPrimary: "var(--noc-text-primary, #f8fafc)",
          textSecondary: "var(--noc-text-secondary, #e2e8f0)",
          textMuted: "var(--noc-text-muted, #94a3b8)",
          ok: "var(--noc-ok, #10b981)",
          warn: "var(--noc-warn, #eab308)",
          crit: "var(--noc-crit, #ef4444)",
          idle: "var(--noc-idle, #94a3b8)",
          cyan: "var(--noc-cyan, #38bdf8)",
          teal: "var(--noc-teal, #06b6d4)",
          indigo: "var(--noc-indigo, #6366f1)",
          indigoText: "var(--noc-indigo-text, #818cf8)",
          purple: "var(--noc-purple, #a855f7)",
          purpleText: "var(--noc-purple-text, #c084fc)",
          pink: "var(--noc-pink, #ec4899)",
          amber: "var(--noc-amber, #f59e0b)",
          orange: "var(--noc-orange, #f97316)",
          emerald: "var(--noc-emerald, #22c55e)",
        },
        accent: {
          primary: "#6366f1",
          secondary: "#818cf8",
          success: "#10b981",
          warning: "#eab308",
          danger: "#ef4444",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
