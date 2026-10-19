import type { AppMode } from "../../types";

/** Lazy import map — warms route chunks on sidebar hover (§G.7). */
const PREFETCH: Partial<Record<AppMode, () => Promise<unknown>>> = {
  simulation: () => import("../../pages/monitor/SimulationMonitor"),
  training: () => import("../../pages/monitor/TrainingMonitor"),
  process_monitor: () => import("../../pages/monitor/ProcessMonitor"),
  simulation_summary: () => import("../../pages/analysis/SimulationSummary"),
  benchmark: () => import("../../pages/analysis/BenchmarkAnalysis"),
  city_comparison: () => import("../../pages/analysis/CityComparison"),
  olap_explorer: () => import("../../pages/analysis/OlapExplorer"),
  data_explorer: () => import("../../pages/analysis/DataExplorer"),
  experiment_tracker: () => import("../../pages/analysis/ExperimentTracker"),
  algorithms: () => import("../../pages/analysis/AlgorithmComparison"),
  hpo_tracker: () => import("../../pages/analysis/HPOTracker"),
  sim_launcher: () => import("../../pages/launch/SimulationLauncher"),
  cloud_launcher: () => import("../../pages/cloud/CloudLauncher"),
  training_hub: () => import("../../pages/launch/TrainingHub"),
  data_gen: () => import("../../pages/launch/DataGeneration"),
  eval_runner: () => import("../../pages/launch/EvaluationRunner"),
  report_studio: () => import("../../pages/launch/ReportStudio"),
  config_editor: () => import("../../pages/files/ConfigEditor"),
  output_browser: () => import("../../pages/files/OutputBrowser"),
  system_tools: () => import("../../pages/files/SystemTools"),
  settings: () => import("../../pages/app/Settings"),
};

const warmed = new Set<AppMode>();

export function prefetchPage(mode: AppMode): void {
  if (warmed.has(mode)) return;
  const loader = PREFETCH[mode];
  if (!loader) return;
  warmed.add(mode);
  loader().catch(() => warmed.delete(mode));
}
