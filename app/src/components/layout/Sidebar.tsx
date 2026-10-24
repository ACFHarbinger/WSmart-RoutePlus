import {
  Activity,
  BarChart2,
  Bot,
  Cloud,
  Cpu,
  Database,
  FileText,
  FlaskConical,
  GitBranch,
  Globe,
  Layers,
  LineChart,
  Map,
  Table2,
  Play,
  Settings,
  Sliders,
  Terminal,
  Zap,
  FolderOpen,
  ClipboardList,
  Presentation,
  Wrench,
} from "lucide-react";
import { useAppStore } from "../../store/app";
import { useLayoutStore } from "../../store/layout";
import { useProcessStore } from "../../store/process";
import type { AppMode, NavSection } from "../../types";
import { prefetchPage } from "../../utils/app/pagePrefetch";

/**
 * Navigation grouped into the 5 user workflows (Run / Watch / Read / Query / Write)
 * plus Files & App (Slice V4).
 */
export const NAV: NavSection[] = [
  {
    label: "Run",
    items: [
      { mode: "sim_launcher", label: "Simulation Launcher" },
      { mode: "training_hub", label: "Training & HPO Hub" },
      { mode: "data_gen", label: "Data Generation" },
      { mode: "eval_runner", label: "Evaluation Runner" },
      { mode: "cloud_launcher", label: "Cloud Launcher" },
    ],
  },
  {
    label: "Watch",
    items: [
      { mode: "simulation", label: "Simulation Digital Twin", streamlitEquivalent: "simulation" },
      { mode: "training", label: "Training Monitor", streamlitEquivalent: "training" },
      { mode: "process_monitor", label: "Process Monitor" },
      { mode: "cloud_monitor", label: "Cloud Job Monitor" },
    ],
  },
  {
    label: "Read",
    items: [
      { mode: "simulation_summary", label: "Simulation Summary", streamlitEquivalent: "simulation_summary" },
      { mode: "benchmark", label: "Benchmark Analysis", streamlitEquivalent: "benchmark" },
      { mode: "city_comparison", label: "City Comparison" },
      { mode: "experiment_tracker", label: "Experiment Tracker", streamlitEquivalent: "experiment_tracker" },
      { mode: "hpo_tracker", label: "HPO Tracker", streamlitEquivalent: "hpo_tracker" },
      { mode: "algorithms", label: "Algorithm Registry", streamlitEquivalent: "algorithms" },
      { mode: "cloud_analysis", label: "Cloud Analysis" },
    ],
  },
  {
    label: "Query",
    items: [
      { mode: "olap_explorer", label: "OLAP Explorer" },
      { mode: "data_explorer", label: "Data Explorer", streamlitEquivalent: "data_explorer" },
    ],
  },
  {
    label: "Write",
    items: [
      { mode: "report_studio", label: "Report Studio" },
    ],
  },
  {
    label: "Files",
    items: [
      { mode: "output_browser", label: "Output Browser" },
      { mode: "config_editor", label: "Config Editor" },
      { mode: "system_tools", label: "System Tools" },
    ],
  },
  {
    label: "App",
    items: [
      { mode: "settings", label: "Settings" },
    ],
  },
];

const ICON: Record<AppMode, React.ReactNode> = {
  simulation: <Map size={15} />,
  training: <Activity size={15} />,
  process_monitor: <Terminal size={15} />,
  cloud_monitor: <Cloud size={15} />,
  cloud_analysis: <LineChart size={15} />,
  simulation_summary: <BarChart2 size={15} />,
  benchmark: <Layers size={15} />,
  city_comparison: <Globe size={15} />,
  olap_explorer: <Table2 size={15} />,
  data_explorer: <Database size={15} />,
  experiment_tracker: <GitBranch size={15} />,
  algorithms: <Bot size={15} />,
  hpo_tracker: <Sliders size={15} />,
  sim_launcher: <Play size={15} />,
  cloud_launcher: <Cloud size={15} />,
  training_hub: <Cpu size={15} />,
  data_gen: <FlaskConical size={15} />,
  config_editor: <FileText size={15} />,
  output_browser: <FolderOpen size={15} />,
  eval_runner: <ClipboardList size={15} />,
  report_studio: <Presentation size={15} />,
  system_tools: <Wrench size={15} />,
  settings: <Settings size={15} />,
};

export function Sidebar() {
  const { mode, setMode } = useAppStore();
  const sidebarOpen = useLayoutStore((s) => s.sidebarOpen);
  const setSidebarOpen = useLayoutStore((s) => s.setSidebarOpen);
  const processes = useProcessStore((s) => s.processes);
  const running = Object.values(processes).filter((p) => p.status === "running").length;

  return (
    <aside
      data-tour="sidebar"
      className={[
        "shrink-0 flex flex-col bg-noc-panel border-r border-noc-border h-screen overflow-y-auto z-30 select-none",
        "w-56 transition-transform duration-200",
        "fixed lg:static inset-y-0 left-0",
        sidebarOpen
          ? "translate-x-0"
          : "-translate-x-full lg:translate-x-0 lg:w-0 lg:border-0 lg:overflow-hidden",
      ].join(" ")}
    >
      {/* Brand */}
      <div className="px-4 py-3.5 border-b border-noc-border bg-noc-deep/50">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-noc-cyan/15 border border-noc-cyan/30 flex items-center justify-center">
            <Zap size={14} className="text-noc-cyan" />
          </div>
          <div>
            <span className="font-bold text-xs tracking-wider uppercase text-noc-textPrimary">
              WSmart-Route
            </span>
            <div className="text-[9px] uppercase font-mono tracking-widest text-noc-cyan">
              NOC Studio
            </div>
          </div>
        </div>
      </div>

      {/* Nav sections */}
      <nav className="flex-1 px-2 py-3 space-y-4">
        {NAV.map((section) => (
          <div key={section.label}>
            <p className="px-2.5 pb-1 text-[10px] font-bold uppercase tracking-wider text-noc-textMuted">
              {section.label}
            </p>
            <ul className="space-y-0.5">
              {section.items.map((item) => {
                const active = mode === item.mode;
                const isProcessMonitor = item.mode === "process_monitor";

                return (
                  <li key={item.mode}>
                    <button
                      data-tour={
                        item.mode === "simulation"
                          ? "simulation"
                          : item.mode === "output_browser"
                          ? "output-browser"
                          : item.mode === "sim_launcher"
                          ? "launch"
                          : undefined
                      }
                      onMouseEnter={() => prefetchPage(item.mode)}
                      onClick={() => {
                        setMode(item.mode);
                        if (window.innerWidth < 1024) setSidebarOpen(false);
                      }}
                      className={[
                        "w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs transition-colors text-left font-medium",
                        active
                          ? "bg-noc-card border-l-2 border-noc-cyan text-noc-textPrimary shadow-sm"
                          : "text-noc-textSecondary hover:text-noc-textPrimary hover:bg-canvas-hover border-l-2 border-transparent",
                      ].join(" ")}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <span className={active ? "text-noc-cyan" : "text-noc-textMuted"}>
                          {ICON[item.mode]}
                        </span>
                        <span className="truncate">{item.label}</span>
                      </div>

                      {isProcessMonitor && running > 0 && (
                        <span className="px-1.5 py-0.2 rounded-full text-[10px] font-mono font-bold bg-noc-ok/20 text-noc-ok border border-noc-ok/40 tabular-nums animate-pulse">
                          {running}
                        </span>
                      )}
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-2.5 border-t border-noc-border text-[10px] font-mono text-noc-textMuted flex items-center justify-between">
        <span>v0.1.0</span>
        <span className="text-noc-ok">CONNECTED</span>
      </div>
    </aside>
  );
}
