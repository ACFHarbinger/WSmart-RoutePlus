//! Data transfer objects and event types for the WSmart-Route Cloud Runner.
//!
//! Mirrors the Python dataclasses in `logic/controllers/cloud/model.py` and
//! the payloads returned by `logic/controllers/cloud/cli.py`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Summary of a cloud job returned by `cloud status` (without job_id).
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudJobSummary {
    pub job_id: String,
    pub name: String,
    pub state: String,
    #[serde(default)]
    pub counts: HashMap<String, u64>,
}

/// Summary of a single task within a cloud job.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudTaskSummary {
    pub task_id: String,
    pub cell_id: String,
    pub state: String,
    pub attempt: u32,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub started_at: Option<f64>,
    #[serde(default)]
    pub finished_at: Option<f64>,
    #[serde(default)]
    pub duration: Option<f64>,
    #[serde(default)]
    pub expected_wall_seconds: Option<f64>,
    #[serde(default)]
    pub policy: Option<String>,
    #[serde(default)]
    pub cost: Option<f64>,
}

/// Detailed status of a cloud job including its individual task records.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudJobStatus {
    pub job_id: String,
    pub name: String,
    pub state: String,
    #[serde(default)]
    pub counts: HashMap<String, u64>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub compute: Option<String>,
    #[serde(default)]
    pub spot: Option<bool>,
    #[serde(default)]
    pub machine_type: Option<String>,
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub price_per_hour: Option<f64>,
    #[serde(default)]
    pub on_demand_price_per_hour: Option<f64>,
    #[serde(default)]
    pub cost_source: Option<String>,
    #[serde(default)]
    pub tasks: Vec<CloudTaskSummary>,
}

/// Parameters for submitting a new cloud job.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct CloudSubmitRequest {
    /// Path to manifest file (native or rerun format, JSON/YAML).
    pub manifest: Option<String>,
    /// One test_sim command line, quoted.
    pub command: Option<String>,
    /// Keep only this routing algorithm's cells (rerun manifests).
    pub policy: Option<String>,
    /// Human-readable job name.
    pub name: Option<String>,
    /// Provider ("local", "gcp", "aws", "azure").
    pub provider: Option<String>,
    /// Compute kind ("process", "batch", "serverless", "vm").
    pub compute: Option<String>,
    /// Maximum concurrent tasks of this job.
    pub max_parallel: Option<u32>,
    /// Requested vCPUs per task.
    pub cpus: Option<f64>,
    /// Requested memory in GiB per task.
    pub memory_gb: Option<f64>,
    /// Use spot or preemptible capacity.
    pub spot: Option<bool>,
    /// Automatic retries per task.
    pub retries: Option<u32>,
    /// Wall-time limit per attempt, in seconds.
    pub max_wall: Option<f64>,
    /// Kill an attempt after this many seconds without progress.
    pub idle: Option<f64>,
    /// Gurobi licensing mode ("none", "wls_secret", "byo").
    pub gurobi: Option<String>,
    /// Secret-store name of Gurobi license (never the secret value itself).
    pub gurobi_secret: Option<String>,
    /// Provider machine type (prices and runtime predictions use it, CR-10/CR-11).
    #[serde(default)]
    pub machine_type: Option<String>,
    /// Provider region.
    #[serde(default)]
    pub region: Option<String>,
    /// Job cost cap in USD (CR-10).
    #[serde(default)]
    pub budget: Option<f64>,
    /// Wait for the job to finish synchronously in CLI runner.
    pub wait: Option<bool>,
    /// Seconds between scheduling passes when wait is enabled.
    pub poll: Option<f64>,
}

/// Response payload from `cloud submit`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudSubmitResult {
    pub job_id: String,
    pub tasks: usize,
    #[serde(default)]
    pub state: Option<String>,
}

/// Response payload from `cloud cancel`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudCancelResult {
    pub job_id: String,
    pub cancel_requested: bool,
}

/// Response payload from `cloud retry`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudRetryResult {
    pub job_id: String,
    pub retried: usize,
}

/// Response payload from `cloud logs`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudTaskLogsResult {
    pub task_id: String,
    pub log: String,
}

/// Response payload from `cloud download`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudDownloadResult {
    pub job_id: String,
    pub dest: String,
    pub files: usize,
    /// Per-file listing (`relpath`, `sha256`, `size`); absent from older CLIs.
    #[serde(default)]
    pub file_list: Vec<CloudDownloadedFile>,
}

/// One downloaded result file, as listed by `cloud download --json`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudDownloadedFile {
    pub relpath: String,
    pub sha256: String,
    pub size: u64,
}

/// Capabilities of a compute backend.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BackendCapabilities {
    pub provider: String,
    #[serde(default)]
    pub kinds: Vec<String>,
    pub max_wall_seconds: Option<f64>,
    pub supports_spot: bool,
    #[serde(default)]
    pub regions: Vec<String>,
    #[serde(default)]
    pub machine_types: HashMap<String, serde_json::Value>,
}

/// Configured backend entry returned by `cloud backends`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudBackendEntry {
    pub provider: String,
    pub compute: String,
    pub capabilities: BackendCapabilities,
}

/// Response payload from `cloud worker --once`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudWorkerTickResult {
    pub changes: usize,
}

/// Event emitted on Tauri event `cloud://job-updated`.
///
/// Sent by the polling fallback, and designed to be fed by Firestore
/// or Supabase realtime (CR-04) later under the same contract.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudJobStatusUpdateEvent {
    pub job_id: String,
    pub status: CloudJobStatus,
    /// Origin feed: "local_poller", "firestore_realtime", or "supabase_realtime".
    pub source: String,
    /// Milliseconds since Unix epoch.
    pub timestamp_ms: u64,
}

/// Event emitted on Tauri event `cloud://download-progress`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CloudDownloadProgressEvent {
    pub job_id: String,
    pub dest: String,
    /// Stage indicator: "starting", "downloading", "verifying", "completed", "failed".
    pub stage: String,
    pub files_downloaded: usize,
    pub total_files: Option<usize>,
    pub message: Option<String>,
}
