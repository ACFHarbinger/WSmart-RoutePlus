import type { AppMode } from "../types";

export type PaletteAction = "toggle_theme" | "shortcuts_help" | "import_wsroute" | "guided_tour";

export interface PaletteCommand {
  id: string;
  label: string;
  section: string;
  keywords?: string;
  mode?: AppMode;
  action?: PaletteAction;
}

/** Searchable command palette entries (§G.7). */
export const PALETTE_COMMANDS: PaletteCommand[] = [
  { id: "simulation", label: "Simulation Digital Twin", section: "Monitor", mode: "simulation", keywords: "map geospatial twin" },
  { id: "training", label: "Training Monitor", section: "Monitor", mode: "training", keywords: "lightning metrics" },
  { id: "process_monitor", label: "Process Monitor", section: "Monitor", mode: "process_monitor", keywords: "logs spawn" },
  { id: "simulation_summary", label: "Simulation Summary", section: "Analysis", mode: "simulation_summary", keywords: "overview ranking" },
  { id: "benchmark", label: "Benchmark Analysis", section: "Analysis", mode: "benchmark", keywords: "compare runs" },
  { id: "city_comparison", label: "City Comparison", section: "Analysis", mode: "city_comparison", keywords: "graph scale rm ffz" },
  { id: "olap_explorer", label: "OLAP Explorer", section: "Analysis", mode: "olap_explorer", keywords: "duckdb sql pivot cube" },
  { id: "data_explorer", label: "Data Explorer", section: "Analysis", mode: "data_explorer", keywords: "csv table" },
  { id: "experiment_tracker", label: "Experiment Tracker", section: "Analysis", mode: "experiment_tracker", keywords: "mlflow zenml" },
  { id: "algorithms", label: "Algorithm Registry", section: "Analysis", mode: "algorithms", keywords: "policies constructors" },
  { id: "algorithm_visualizer", label: "Algorithm Visualizer", section: "Analysis", mode: "algorithm_visualizer", keywords: "trace catalog packing routing" },
  { id: "hpo_tracker", label: "HPO Tracker", section: "Analysis", mode: "hpo_tracker", keywords: "optuna trials" },
  { id: "sim_launcher", label: "Simulation Launcher", section: "Launch", mode: "sim_launcher", keywords: "test_sim hydra" },
  { id: "algorithm_builder", label: "Algorithm Builder", section: "Launch", mode: "algorithm_builder", keywords: "trace catalog packing routing" },
  { id: "training_hub", label: "Training & HPO Hub", section: "Launch", mode: "training_hub", keywords: "train eval" },
  { id: "data_gen", label: "Data Generation", section: "Launch", mode: "data_gen", keywords: "dataset bins" },
  { id: "eval_runner", label: "Evaluation Runner", section: "Launch", mode: "eval_runner", keywords: "checkpoint eval" },
  { id: "output_browser", label: "Output Browser", section: "Files", mode: "output_browser", keywords: "artefacts wsroute" },
  { id: "config_editor", label: "Config Editor", section: "Files", mode: "config_editor", keywords: "yaml hydra" },
  { id: "system_tools", label: "System Tools", section: "Files", mode: "system_tools", keywords: "file system delete update cryptography test suite pytest" },
  { id: "report_studio", label: "Report Studio", section: "Launch", mode: "report_studio", keywords: "report presentation deck pptx markdown analysis figures dataset simulation speaker script excel" },
  { id: "settings", label: "Settings", section: "App", mode: "settings", keywords: "project root theme" },
  { id: "toggle_theme", label: "Cycle Theme (Dark / Light / System)", section: "Actions", action: "toggle_theme", keywords: "appearance system monitor" },
  { id: "shortcuts_help", label: "Keyboard Shortcuts Help", section: "Actions", action: "shortcuts_help", keywords: "hotkeys" },
  { id: "import_wsroute", label: "Import .wsroute Bundle", section: "Actions", action: "import_wsroute", keywords: "extract package zip" },
  { id: "guided_tour", label: "Guided Tour", section: "Actions", action: "guided_tour", keywords: "onboarding help walkthrough" },
];
