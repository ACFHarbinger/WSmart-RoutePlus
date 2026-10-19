/**
//! Cloud Runner types mirroring Rust / Python models for CR-13.
*/

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
}

export interface CloudJobStatus {
  job_id: string;
  name: string;
  state: "queued" | "running" | "succeeded" | "failed" | "cancelled" | string;
  counts: Record<string, number>;
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

export interface CloudDownloadResult {
  job_id: string;
  dest: string;
  files: number;
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

export type CloudProvider = "local" | "gcp" | "aws" | "azure";
export type CloudComputeKind = "process" | "batch" | "serverless" | "vm";
export type CloudGurobiMode = "none" | "wls_secret" | "byo";

export interface CloudJobEstimate {
  cost_usd?: number | null;
  eta_seconds?: number | null;
  memory_required_gb?: number | null;
  source?: string | null;
  details?: Record<string, unknown>;
}
