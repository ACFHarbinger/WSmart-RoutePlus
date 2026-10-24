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
import { LoadedRunChip } from "./LoadedRunChip";

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
  const running = Object.values(processes).filter(
    (p) => p.status === "running"
  ).length;

  return (
    <>
      <header className="h-11 shrink-0 flex items-center justify-between px-5 bg-canvas-surface border-b border-canvas-border">
        <div className="flex items-center gap-2 min-w-0">
          <button
            onClick={toggleSidebar}
            className="btn-ghost p-1.5 shrink-0"
            title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
          >
            <PanelLeft
              size={14}
              className={sidebarOpen ? "" : "text-canvas-muted"}
            />
          </button>
          <h1 className="text-sm font-semibold text-canvas-text truncate">
            {TITLES[mode] ?? mode}
          </h1>
        </div>

        <div className="flex items-center gap-3">
          <LoadedRunChip />

          {running > 0 && (
            <button
              type="button"
              onClick={() => setMode("process_monitor")}
              title="Open Process Monitor"
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-accent-success/30 bg-accent-success/10 text-xs text-accent-success tabular-nums hover:bg-accent-success/20 transition-colors cursor-pointer"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-accent-success animate-pulse" />
              {running} running
            </button>
          )}

          <button
            data-tour="command-palette"
            onClick={() => setCommandPaletteOpen(true)}
            className="btn-ghost p-1.5"
            title="Command palette (Ctrl+K)"
          >
            <Search size={14} />
          </button>

          <button
            onClick={() => {
              setGuidedTourStep(0);
              setGuidedTourOpen(true);
            }}
            className="btn-ghost p-1.5"
            title="Guided tour"
          >
            <Compass size={14} />
          </button>

          <button
            onClick={() => setShortcutsOpen(true)}
            className="btn-ghost p-1.5"
            title="Keyboard shortcuts (?)"
          >
            <Keyboard size={14} />
          </button>

          <button
            onClick={() => setTheme(nextThemePreference(theme))}
            className="btn-ghost p-1.5"
            title={`Theme: ${theme} (click to cycle)`}
          >
            {theme === "dark" ? (
              <Sun size={14} />
            ) : theme === "light" ? (
              <Moon size={14} />
            ) : (
              <Monitor size={14} />
            )}
          </button>
        </div>
      </header>

      {/* First-run banner — shown when projectRoot is not configured */}
      {!projectRoot && mode !== "settings" && (
        <div className="flex items-center gap-2 px-5 py-2 bg-accent-warning/10 border-b border-accent-warning/30 text-xs text-accent-warning">
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
