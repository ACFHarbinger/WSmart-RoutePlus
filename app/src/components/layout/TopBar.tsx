import {
  Moon,
  PanelLeft,
  Sun,
  Monitor,
  AlertTriangle,
  Settings,
  Keyboard,
  Search,
  Compass,
} from "lucide-react";
import { useAppStore } from "../../store/app";
import { nextThemePreference } from "../../utils/charts/theme";
import { useLayoutStore } from "../../store/layout";
import { useProcessStore } from "../../store/process";
import { useDuckDbStore } from "../../store/duckdb";
import { LoadedRunChip } from "./LoadedRunChip";
import { HudStatusPill } from "../ui/telemetry/status";

export const TITLES: Record<string, string> = {
  simulation: "Simulation Digital Twin",
  training: "Training Monitor",
  process_monitor: "Process Monitor",
  cloud_monitor: "Cloud Job Monitor",
  simulation_summary: "Simulation Summary",
  benchmark: "Benchmark Analysis",
  city_comparison: "City Comparison",
  olap_explorer: "OLAP Explorer",
  data_explorer: "Data Explorer",
  experiment_tracker: "Experiment Tracker",
  algorithms: "Algorithm Registry",
  hpo_tracker: "HPO Tracker",
  cloud_analysis: "Cloud Analysis",
  sim_launcher: "Simulation Launcher",
  cloud_launcher: "Cloud Launcher",
  training_hub: "Training & HPO Hub",
  data_gen: "Data Generation Wizard",
  eval_runner: "Evaluation Runner",
  report_studio: "Report Studio",
  config_editor: "Configuration Editor",
  output_browser: "Output Browser",
  system_tools: "System Tools",
  settings: "Settings",
};

export function TopBar() {
  const { mode, theme, setTheme, projectRoot, setMode } = useAppStore();
  const {
    sidebarOpen,
    toggleSidebar,
    setShortcutsOpen,
    setCommandPaletteOpen,
    setGuidedTourOpen,
    setGuidedTourStep,
  } = useLayoutStore();
  const processes = useProcessStore((s) => s.processes);
  const { ready: duckDbReady, loading: duckDbLoading, error: duckDbError } = useDuckDbStore();

  const running = Object.values(processes).filter(
    (p) => p.status === "running"
  ).length;

  return (
    <>
      <header className="h-11 shrink-0 flex items-center justify-between px-4 bg-noc-panel border-b border-noc-border select-none">
        <div className="flex items-center gap-2.5 min-w-0">
          <button
            onClick={toggleSidebar}
            className="btn-ghost p-1.5 shrink-0 hover:bg-canvas-hover rounded-lg transition-colors"
            title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
          >
            <PanelLeft
              size={14}
              className={sidebarOpen ? "text-noc-cyan" : "text-noc-textMuted"}
            />
          </button>
          <div className="flex items-center gap-2 min-w-0">
            <span className="w-1.5 h-1.5 rounded-full bg-noc-cyan shrink-0" />
            <h1 className="text-xs font-bold uppercase tracking-wider text-noc-textPrimary truncate">
              {TITLES[mode] ?? mode}
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-2 sm:gap-2.5">
          <LoadedRunChip />

          {/* DuckDB-Wasm Engine Telemetry Pill */}
          <HudStatusPill
            label="DUCKDB"
            value={duckDbReady ? "READY" : duckDbLoading ? "INIT" : duckDbError ? "ERR" : "OFF"}
            tone={duckDbReady ? "cyan" : duckDbLoading ? "amber" : duckDbError ? "crit" : "idle"}
            title={duckDbError ?? (duckDbReady ? "DuckDB-Wasm OLAP Engine Active" : "Initializing DuckDB-Wasm")}
            pulse={duckDbLoading}
          />

          {/* Background CLI / Python Subprocesses Pill */}
          <HudStatusPill
            label="CLI"
            value={running > 0 ? `${running} RUNNING` : "IDLE"}
            tone={running > 0 ? "ok" : "idle"}
            pulse={running > 0}
            onClick={() => setMode("process_monitor")}
            title="Open Process Monitor"
          />

          <button
            data-tour="command-palette"
            onClick={() => setCommandPaletteOpen(true)}
            className="btn-ghost p-1.5 rounded-lg hover:bg-canvas-hover text-noc-textMuted hover:text-noc-textPrimary transition-colors"
            title="Command palette (Ctrl+K)"
          >
            <Search size={14} />
          </button>

          <button
            onClick={() => {
              setGuidedTourStep(0);
              setGuidedTourOpen(true);
            }}
            className="btn-ghost p-1.5 rounded-lg hover:bg-canvas-hover text-noc-textMuted hover:text-noc-textPrimary transition-colors"
            title="Guided tour"
          >
            <Compass size={14} />
          </button>

          <button
            onClick={() => setShortcutsOpen(true)}
            className="btn-ghost p-1.5 rounded-lg hover:bg-canvas-hover text-noc-textMuted hover:text-noc-textPrimary transition-colors"
            title="Keyboard shortcuts (?)"
          >
            <Keyboard size={14} />
          </button>

          <button
            onClick={() => setTheme(nextThemePreference(theme))}
            className="btn-ghost p-1.5 rounded-lg hover:bg-canvas-hover text-noc-textMuted hover:text-noc-textPrimary transition-colors"
            title={`Theme: ${theme} (click to cycle)`}
          >
            {theme === "dark" ? (
              <Sun size={14} className="text-noc-amber" />
            ) : theme === "light" ? (
              <Moon size={14} className="text-noc-indigo" />
            ) : (
              <Monitor size={14} />
            )}
          </button>
        </div>
      </header>

      {/* First-run banner — shown when projectRoot is not configured */}
      {!projectRoot && mode !== "settings" && (
        <div className="flex items-center gap-2 px-4 py-2 bg-accent-warning/10 border-b border-accent-warning/30 text-xs text-accent-warning">
          <AlertTriangle size={13} />
          <span>
            Project root is not set — launchers and browsers are disabled.
          </span>
          <button
            onClick={() => setMode("settings")}
            className="ml-auto flex items-center gap-1 underline hover:no-underline"
          >
            <Settings size={11} />
            Open Settings
          </button>
        </div>
      )}
    </>
  );
}
