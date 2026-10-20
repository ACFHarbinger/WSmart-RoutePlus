/**
//! Cloud Runner types mirroring Rust / Python models for CR-13.
*/

export type CloudProvider = "local" | "gcp" | "aws" | "azure";
export type CloudComputeKind = "process" | "batch" | "serverless" | "vm";
export type CloudGurobiMode = "none" | "wls_secret" | "byo";

export interface CloudJobSummary {
  job_id: string;
  name: string;
  state: "queued" | "running" | "succeeded" | "failed" | "cancelled" | string;
  counts: Record<string, number>;
}

export interface CloudTaskSummary {
  task_id: string;
  cell_id: string;
  state: "queued" | "placing" | "starting" | "running" | "preempted" | "retrying" | "succeeded" | "failed" | "cancelled" | string;
  attempt: number;
  message?: string | null;
  started_at?: number | null;
  finished_at?: number | null;
  duration?: number | null;
  expected_wall_seconds?: number | null;
  policy?: string | null;
  cost?: number | null;
}

export interface CloudJobStatus {
  job_id: string;
  name: string;
  state: "queued" | "running" | "succeeded" | "failed" | "cancelled" | string;
  counts: Record<string, number>;
  provider?: "local" | "gcp" | "aws" | "azure" | string | null;
  compute?: "process" | "batch" | "serverless" | "vm" | string | null;
  spot?: boolean | null;
  machine_type?: string | null;
  region?: string | null;
  price_per_hour?: number | null;
  on_demand_price_per_hour?: number | null;
  cost_source?: string | null;
  tasks: CloudTaskSummary[];
}

export interface CloudSubmitRequest {
  manifest?: string;
  command?: string;
  policy?: string;
  name?: string;
  provider?: "local" | "gcp" | "aws" | "azure" | string;
  compute?: "process" | "batch" | "serverless" | "vm" | string;
  max_parallel?: number;
  cpus?: number;
  memory_gb?: number;
  spot?: boolean;
  retries?: number;
  max_wall?: number;
  idle?: number;
  gurobi?: "none" | "wls_secret" | "byo" | string;
  gurobi_secret?: string;
  wait?: boolean;
  poll?: number;
}

export interface CloudSubmitResult {
  job_id: string;
  tasks: number;
  state?: string | null;
}

export interface CloudCancelResult {
  job_id: string;
  cancel_requested: boolean;
}

export interface CloudRetryResult {
  job_id: string;
  retried: number;
}

export interface CloudTaskLogsResult {
  task_id: string;
  log: string;
}

export interface CloudDownloadedFile {
  relpath: string;
  sha256: string;
  size: number;
}

export interface CloudDownloadResult {
  job_id: string;
  dest: string;
  files: number;
  /** Per-file listing; empty with older CLIs. */
  file_list: CloudDownloadedFile[];
}

export interface BackendCapabilities {
  provider: string;
  kinds: string[];
  max_wall_seconds?: number | null;
  supports_spot: boolean;
  regions: string[];
  machine_types: Record<string, unknown>;
}

export interface CloudBackendEntry {
  provider: string;
  compute: string;
  capabilities: BackendCapabilities;
}

export interface CloudWorkerTickResult {
  changes: number;
}

export interface CloudJobStatusUpdateEvent {
  job_id: string;
  status: CloudJobStatus;
  source: "local_poller" | "firestore_realtime" | "supabase_realtime" | string;
  timestamp_ms: number;
}

export interface CloudDownloadProgressEvent {
  job_id: string;
  dest: string;
  stage: "starting" | "downloading" | "verifying" | "completed" | "failed" | string;
  files_downloaded: number;
  total_files?: number | null;
  message?: string | null;
}

export interface CloudJobEstimate {
  cost_usd?: number | null;
  eta_seconds?: number | null;
  memory_required_gb?: number | null;
  source?: string | null;
  details?: Record<string, unknown>;
  estimated_wall_seconds?: number;
  estimated_cost_usd?: number;
  hourly_rate_usd?: number;
  machine_type?: string | null;
  spot?: boolean;
  provider?: string;
  compute?: string;
}

export interface CloudTaskAnalysis {
  task_id: string;
  cell_id: string;
  policy?: string | null;
  state: string;
  attempt: number;
  duration_seconds: number | null;
  expected_wall_seconds?: number | null;
  cost_usd: number | null;
  cpu_percent?: number | null;
  memory_rss_mb?: number | null;
  message?: string | null;
}

export interface CloudJobAnalysis {
  job_id: string;
  name: string;
  state: string;
  provider: string;
  compute: string;
  spot: boolean | null;
  machine_type?: string | null;
  region?: string | null;
  total_tasks: number;
  succeeded_tasks: number;
  failed_tasks: number;
  cancelled_tasks: number;
  preempted_tasks: number;
  total_wall_seconds: number | null;
  total_compute_seconds: number | null;
  accrued_cost_usd: number | null;
  estimated_cost_usd?: number | null;
  on_demand_baseline_cost_usd?: number | null;
  spot_savings_usd?: number | null;
  spot_savings_pct?: number | null;
  tasks: CloudTaskAnalysis[];
}
