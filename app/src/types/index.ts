// ── Simulation log types ─────────────────────────────────────────────────────
// Mirror of Python DayLogEntry / SimDayData from logic/src/ui/services/log_parser.py

export interface BinCoord {
  id: number;
  dataset_id?: number;
  lat?: number;
  lng?: number;
}

export interface SimFailureOverflowBin {
  bin_index: number;
  bin_id: number;
  predicted_fill: number;
  actual_fill: number;
  fill_delta: number;
  fill_spike: boolean;
  in_tour: boolean;
  collected: boolean;
  fill_level_after: number;
}

export interface SimFailureSkippedBin {
  bin_index: number;
  bin_id: number;
  fill_level: number;
  mandatory: boolean;
}

export interface SimFailureSummary {
  has_failure: boolean;
  severity: string;
  root_causes: string[];
  summary: string;
  metrics?: {
    new_overflows: number;
    kg_lost: number;
    profit: number;
  };
  overflow_bins?: SimFailureOverflowBin[];
  skipped_high_fill_bins?: SimFailureSkippedBin[];
}

export interface SimDayData {
  tour?: Array<BinCoord | number>;
  bin_state_c?: number[];
  bin_state_collected?: number[];
  mandatory?: number[];
  tour_indices?: number[];
  all_bin_coords?: BinCoord[];
  failure_analysis?: SimFailureSummary;
  // KPI metrics — matches _PRIMARY_KPI_MAP + _SECONDARY_KPI_MAP in kpi.py
  overflows?: number;
  kg?: number;
  km?: number;
  "kg/km"?: number;
  cost?: number;
  profit?: number;
  ncol?: number;
  kg_lost?: number;
}

export interface DayLogEntry {
  policy: string;
  sample_id: number;
  day: number;
  data: SimDayData;
}

// ── Policy telemetry (PolicyVizMixin → Studio §A.3) ─────────────────────────

export type PolicyVizType = "alns" | "hgs" | "aco" | "ils" | "selector" | "generic";

export interface PolicyVizEntry {
  policy: string;
  sample_id: number;
  day: number;
  policy_type: PolicyVizType;
  data: Record<string, Array<number | string | boolean>>;
}

/** Cross-run policy telemetry row from ``assets/telemetry.db`` (§A.3 Option C). */
export interface PolicyTelemetryTrendRow {
  id: number;
  log_path: string;
  run_label: string | null;
  run_created_at: string;
  policy: string;
  sample_idx: number;
  day: number;
  policy_type: PolicyVizType;
  step_count: number;
  final_metric: number | null;
  metric_name: string | null;
  emitted_at: string;
}

export interface PolicyTelemetryTrends {
  db_path: string;
  rows: PolicyTelemetryTrendRow[];
  policy_types: PolicyVizType[];
}

export interface PolicyTrajectorySeries {
  id: number;
  label: string;
  run_label: string | null;
  policy: string;
  day: number;
  policy_type: PolicyVizType;
  metric_name: string;
  x: number[];
  y: number[];
}

export interface PolicyTrajectoryTrends {
  db_path: string;
  series: PolicyTrajectorySeries[];
}

// ── Simulation failure analysis (FailureAnalyzer → Studio §A.6) ─────────────

export interface SimFailureEntry {
  policy: string;
  sample_id: number;
  day: number;
  data: SimFailureSummary;
}

// ── Training log types ───────────────────────────────────────────────────────

export interface TrainingRun {
  name: string;
  path: string;
  has_metrics: boolean;
  has_hparams: boolean;
}

export interface TrainingMetricsRow {
  epoch?: number;
  step?: number;
  train_loss?: number;
  val_loss?: number;
  reward?: number;
  entropy?: number;
  grad_norm?: number;
  lr?: number;
  [key: string]: number | undefined;
}

// ── Runtime attention (AttentionRingBuffer → Studio §A.2 Option A) ───────────

export interface AttentionSnapshot {
  layer: number;
  head: number;
  decode_step: number;
  n_nodes: number;
  matrix: number[][];
}

export interface AttentionVizEntry {
  phase: string;
  epoch: number;
  step: number;
  snapshots: AttentionSnapshot[];
}

// ── Training health (TrainingHealthCallback → Studio §A.4) ───────────────────

export interface TrainingHealthEntry {
  code: string;
  severity: "warning" | "critical" | string;
  epoch: number;
  step: number;
  message: string;
  details: Record<string, unknown>;
}

// ── Process management ───────────────────────────────────────────────────────

export type ProcessStatus = "running" | "completed" | "cancelled" | "failed";

export interface ProcessEntry {
  id: string;
  command: string;
  pid: number;
  status: ProcessStatus;
  startTime: number;
  exitCode?: number;
  logLines: string[];
  lastEventAt?: number;
}

export interface ProcessSpawned {
  id: string;
  command: string;
  pid: number;
  start_time: number;
}

export interface StdoutLine {
  id: string;
  line: string;
}

export interface StdoutBatch {
  id: string;
  lines: string[];
}

export interface SimBatchUpdate {
  day_entries: DayLogEntry[];
  viz_entries: PolicyVizEntry[];
  failure_entries: SimFailureEntry[];
}

export interface StatusUpdate {
  id: string;
  status: ProcessStatus;
  exit_code?: number;
}

export interface DirEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size_bytes: number;
  extension: string;
}

export interface OutputDir {
  name: string;
  path: string;
  created_at: string;
  size_bytes: number;
}

// ── Policy registry (§G.9) ─────────────────────────────────────────────────

export interface SimPolicyEntry {
  id: string;
  config_key: string;
}

// ── Eval analytics (§G.12) ─────────────────────────────────────────────────

export interface BenchmarkLogRef {
  path: string;
  label: string;
}

export interface EvalAnalyticsRow {
  checkpoint: string;
  checkpointPath?: string;
  cost?: number;
  gap?: number;
  time?: number;
  policy?: string;
  [key: string]: number | string | undefined;
}

// ── Optuna HPO (§G.18) ───────────────────────────────────────────────────────

export interface OptunaStudySummary {
  name: string;
  n_trials: number;
  n_complete: number;
  best_value: number | null;
}

export interface OptunaTrial {
  number: number;
  value: number | null;
  state: string;
  params: Record<string, string | number | boolean>;
  user_attrs?: Record<string, string | number | boolean | null>;
}

export interface DatasetPreviewStats {
  num_instances: number;
  num_nodes: number | null;
  demand_mean: number | null;
  demand_std: number | null;
  demand_histogram: number[];
  distance_mean: number | null;
  file_size_bytes: number;
}

export interface OptunaStudyData {
  name: string;
  trials: OptunaTrial[];
  importances: Record<string, number>;
  best_value: number | null;
  best_params: Record<string, string | number | boolean>;
}

export interface HpoReportExportResult {
  report_dir: string | null;
  files: string[];
  n_complete: number;
  message: string;
}

// ── MLflow experiment tracking (§G.18) ─────────────────────────────────────

export interface MlflowRun {
  run_id: string;
  run_name: string;
  experiment_id: string;
  status: string;
  start_time: number | null;
  end_time: number | null;
  artifact_uri: string;
  params: Record<string, string>;
  metrics: Record<string, number>;
  tags: Record<string, string>;
}

export interface MlflowMetricPoint {
  step: number;
  value: number;
}

// ── ZenML pipeline tracking (§G.18) ──────────────────────────────────────────

export interface ZenmlPipelineRun {
  id: string;
  pipeline: string;
  status: string;
  created: string;
  updated: string;
  stack: string;
}

export interface ZenmlPipelineStep {
  name: string;
  status: string;
  created: string;
  updated: string;
  duration_seconds: number | null;
}

// ── Studio data bundles (§G.8) ───────────────────────────────────────────────

export interface WsrouteBundleFile {
  path: string;
  size_bytes: number;
}

export interface WsrouteBundleInfo {
  path: string;
  version: string | null;
  created_at: string | null;
  arrow_sidecars: number | null;
  files: WsrouteBundleFile[];
}

export interface WsrouteExtractResult {
  dest_dir: string;
  extracted_files: string[];
  log_path: string | null;
}

// ── ML introspection (§G.5) ──────────────────────────────────────────────────

export interface NpzArrayInfo {
  key: string;
  shape: number[];
  dtype: string;
  size_bytes: number;
}

export interface NpzArchiveInfo {
  path: string;
  arrays: NpzArrayInfo[];
  total_bytes: number;
  used_memmap: boolean;
}

export interface TensorSlicePreview {
  key: string;
  full_shape: number[];
  slice_shape: number[];
  indices: number[];
  values: number[][];
  min: number;
  max: number;
  rust_ms: number;
  used_memmap: boolean;
  used_decompress_slice?: boolean;
}

export interface NpzVectorData {
  key: string;
  values: number[];
}

// ── App navigation ───────────────────────────────────────────────────────────

/** Ephemeral map compare intent from Algorithm Comparison → Simulation Monitor. */
export interface PendingMapCompare {
  policies: string[];
  layout: "overlay" | "split";
  mapMode?: "echarts" | "deckgl";
}

export type AppMode =
  // Monitor modes (Streamlit parity)
  | "simulation"        // Simulation Digital Twin  — logic/src/ui/pages/simulation/
  | "training"          // Training Monitor         — logic/src/ui/pages/training.py
  | "simulation_summary"// Simulation Summary       — logic/src/ui/pages/simulation/summary.py
  | "benchmark"         // Benchmark Analysis       — logic/src/ui/components/benchmark_charts.py
  | "city_comparison"   // City Comparison          — §G.1.6 dedicated page
  | "olap_explorer"     // OLAP Data Cube Explorer  — §G.6 standalone page
  | "data_explorer"     // Data Explorer            — logic/src/ui/pages/data_explorer.py
  | "experiment_tracker"// Experiment Tracker       — logic/src/ui/pages/experiment_tracker.py
  | "algorithms"        // Algorithms Registry      — logic/src/ui/pages/algorithms.py
  | "hpo_tracker"       // HPO Tracker              — logic/src/ui/pages/hpo_tracker.py
  | "cloud_analysis"    // Cloud Analysis Dashboard — CR-16 (§J.1)
  // Studio-only modes
  | "process_monitor"   // Process Monitor          — §G.15
  | "cloud_monitor"     // Cloud Job Monitor        — CR-15 (§J.1)
  | "cloud_launcher"    // Cloud Launcher           — CR-14 (§J.1)
  | "sim_launcher"      // Simulation Launcher      — §G.9
  | "training_hub"      // Training & HPO Hub       — §G.10
  | "data_gen"          // Data Generation Wizard   — §G.11
  | "config_editor"     // Configuration Editor     — §G.13
  | "output_browser"    // Output Browser           — §G.14
  | "eval_runner"       // Evaluation Runner        — §G.12
  | "report_studio"     // Report & Deck Generator  — §H interim (archived logic/gen pipeline)
  | "system_tools"      // File System Tools & Test Suite — PySide6 GUI parity
  | "cloud_launcher"    // Cloud Launcher           — CR-14 (§J.1)
  | "settings";         // App Settings             — §G.19

export interface NavSection {
  label: string;
  items: NavItem[];
}

export interface NavItem {
  mode: AppMode;
  label: string;
  streamlitEquivalent?: string;
}

export * from "./cloud";
