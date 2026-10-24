//! Tauri commands for the WSmart-Route Cloud Runner (CR-13).
//!
//! Provides desktop control tower integration for cloud job submission, status inspection,
//! cancellation, retry, log tailing, artifact downloading, backend discovery, and OS keychain
//! credential management.

pub mod cli;
pub mod download;
pub mod keychain;
pub mod models;
pub mod poller;

use tauri::AppHandle;

use self::cli::run_cloud_cli;
#[allow(unused_imports)]
pub use self::download::{compute_file_sha256, execute_cloud_download, verify_file_sha256};
#[allow(unused_imports)]
pub use self::keychain::{
    delete_credential, has_credential, make_account_key, store_credential, KEYCHAIN_SERVICE,
};
#[allow(unused_imports)]
pub use self::models::{
    BackendCapabilities, CloudBackendEntry, CloudCancelResult, CloudDownloadProgressEvent,
    CloudDownloadResult, CloudJobStatus, CloudJobStatusUpdateEvent, CloudJobSummary,
    CloudRetryResult, CloudSubmitRequest, CloudSubmitResult, CloudTaskLogsResult, CloudTaskSummary,
    CloudWorkerTickResult,
};
#[allow(unused_imports)]
pub use self::poller::{get_default_poller, CloudJobWatcher, LocalJobPoller};

/// List summary status for all jobs tracked by the control plane.
#[tauri::command]
pub async fn cloud_list_jobs(
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
) -> Result<Vec<CloudJobSummary>, String> {
    let subcmd = vec!["status".to_string()];
    run_cloud_cli::<Vec<CloudJobSummary>>(
        project_root.as_deref(),
        root.as_deref(),
        python_executable.as_deref(),
        &subcmd,
    )
    .await
}

/// Retrieve detailed status for a specific cloud job and its constituent tasks.
#[tauri::command]
pub async fn cloud_job_status(
    job_id: String,
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
) -> Result<CloudJobStatus, String> {
    if job_id.trim().is_empty() {
        return Err("job_id cannot be empty".to_string());
    }
    let subcmd = vec!["status".to_string(), job_id];
    run_cloud_cli::<CloudJobStatus>(
        project_root.as_deref(),
        root.as_deref(),
        python_executable.as_deref(),
        &subcmd,
    )
    .await
}

/// Submit a new job to the cloud runner (either a cell manifest or single test_sim command).
#[tauri::command]
pub async fn cloud_submit(
    request: CloudSubmitRequest,
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
) -> Result<CloudSubmitResult, String> {
    let mut subcmd = vec!["submit".to_string()];

    if let Some(ref m) = request.manifest {
        subcmd.push("--manifest".to_string());
        subcmd.push(m.clone());
    } else if let Some(ref c) = request.command {
        subcmd.push("--command".to_string());
        subcmd.push(c.clone());
    } else {
        return Err("cloud_submit requires either 'manifest' or 'command' in request".to_string());
    }

    if let Some(ref p) = request.policy {
        subcmd.push("--policy".to_string());
        subcmd.push(p.clone());
    }
    if let Some(ref n) = request.name {
        subcmd.push("--name".to_string());
        subcmd.push(n.clone());
    }
    if let Some(ref prov) = request.provider {
        subcmd.push("--provider".to_string());
        subcmd.push(prov.clone());
    }
    if let Some(ref comp) = request.compute {
        subcmd.push("--compute".to_string());
        subcmd.push(comp.clone());
    }
    if let Some(mp) = request.max_parallel {
        subcmd.push("--max-parallel".to_string());
        subcmd.push(mp.to_string());
    }
    if let Some(cpus) = request.cpus {
        subcmd.push("--cpus".to_string());
        subcmd.push(cpus.to_string());
    }
    if let Some(mem) = request.memory_gb {
        subcmd.push("--memory-gb".to_string());
        subcmd.push(mem.to_string());
    }
    if let Some(true) = request.spot {
        subcmd.push("--spot".to_string());
    }
    if let Some(retries) = request.retries {
        subcmd.push("--retries".to_string());
        subcmd.push(retries.to_string());
    }
    if let Some(wall) = request.max_wall {
        subcmd.push("--max-wall".to_string());
        subcmd.push(wall.to_string());
    }
    if let Some(idle) = request.idle {
        subcmd.push("--idle".to_string());
        subcmd.push(idle.to_string());
    }
    if let Some(ref g) = request.gurobi {
        subcmd.push("--gurobi".to_string());
        subcmd.push(g.clone());
    }
    if let Some(ref gs) = request.gurobi_secret {
        subcmd.push("--gurobi-secret".to_string());
        subcmd.push(gs.clone());
    }
    push_cost_args(&mut subcmd, &request);
    if let Some(b) = request.budget {
        subcmd.push("--budget".to_string());
        subcmd.push(b.to_string());
    }
    if let Some(p) = request.priority {
        subcmd.push("--priority".to_string());
        subcmd.push(p.to_string());
    }
    for target in request.targets.iter().flatten() {
        if !target.trim().is_empty() {
            subcmd.push("--target".to_string());
            subcmd.push(target.trim().to_string());
        }
    }
    if let Some(true) = request.pool {
        subcmd.push("--pool".to_string());
    }
    if let Some(true) = request.wait {
        subcmd.push("--wait".to_string());
    }
    if let Some(poll) = request.poll {
        subcmd.push("--poll".to_string());
        subcmd.push(poll.to_string());
    }

    run_cloud_cli::<CloudSubmitResult>(
        project_root.as_deref(),
        root.as_deref(),
        python_executable.as_deref(),
        &subcmd,
    )
    .await
}

/// Request cancellation of a cloud job.
#[tauri::command]
pub async fn cloud_cancel(
    job_id: String,
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
) -> Result<CloudCancelResult, String> {
    if job_id.trim().is_empty() {
        return Err("job_id cannot be empty".to_string());
    }
    let subcmd = vec!["cancel".to_string(), job_id];
    run_cloud_cli::<CloudCancelResult>(
        project_root.as_deref(),
        root.as_deref(),
        python_executable.as_deref(),
        &subcmd,
    )
    .await
}

/// Requeue failed and cancelled tasks for a job.
#[tauri::command]
pub async fn cloud_retry(
    job_id: String,
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
) -> Result<CloudRetryResult, String> {
    if job_id.trim().is_empty() {
        return Err("job_id cannot be empty".to_string());
    }
    let subcmd = vec!["retry".to_string(), job_id];
    run_cloud_cli::<CloudRetryResult>(
        project_root.as_deref(),
        root.as_deref(),
        python_executable.as_deref(),
        &subcmd,
    )
    .await
}

/// Machine type and region flags shared by `submit` and `cost estimate`.
fn push_cost_args(subcmd: &mut Vec<String>, request: &CloudSubmitRequest) {
    if let Some(ref m) = request.machine_type {
        if !m.trim().is_empty() {
            subcmd.push("--machine-type".to_string());
            subcmd.push(m.trim().to_string());
        }
    }
    if let Some(ref r) = request.region {
        if !r.trim().is_empty() {
            subcmd.push("--region".to_string());
            subcmd.push(r.trim().to_string());
        }
    }
}

/// Pre-launch cost and ETA estimate of a submit request (CR-10), without submitting.
///
/// Returns the `cloud cost estimate --json` payload (the app's `CloudJobEstimate`), or
/// `null` when no runtime predictor is available.
#[tauri::command]
pub async fn cloud_cost_estimate(
    request: CloudSubmitRequest,
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
) -> Result<serde_json::Value, String> {
    let mut subcmd = vec!["cost".to_string(), "estimate".to_string()];
    if let Some(ref m) = request.manifest {
        subcmd.push("--manifest".to_string());
        subcmd.push(m.clone());
    } else if let Some(ref c) = request.command {
        subcmd.push("--command".to_string());
        subcmd.push(c.clone());
    } else {
        return Err("cloud_cost_estimate requires either 'manifest' or 'command' in request".to_string());
    }
    if let Some(ref p) = request.policy {
        subcmd.push("--policy".to_string());
        subcmd.push(p.clone());
    }
    if let Some(ref prov) = request.provider {
        subcmd.push("--provider".to_string());
        subcmd.push(prov.clone());
    }
    if let Some(ref comp) = request.compute {
        subcmd.push("--compute".to_string());
        subcmd.push(comp.clone());
    }
    if let Some(mp) = request.max_parallel {
        subcmd.push("--max-parallel".to_string());
        subcmd.push(mp.to_string());
    }
    if let Some(true) = request.spot {
        subcmd.push("--spot".to_string());
    }
    push_cost_args(&mut subcmd, &request);
    run_cloud_cli::<serde_json::Value>(
        project_root.as_deref(),
        root.as_deref(),
        python_executable.as_deref(),
        &subcmd,
    )
    .await
}

/// Retrieve standard output/logs from a specific attempt of a task.
#[tauri::command]
pub async fn cloud_task_logs(
    task_id: String,
    tail: Option<u32>,
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
) -> Result<CloudTaskLogsResult, String> {
    if task_id.trim().is_empty() {
        return Err("task_id cannot be empty".to_string());
    }
    let mut subcmd = vec!["logs".to_string(), task_id];
    if let Some(t) = tail {
        subcmd.push("--tail".to_string());
        subcmd.push(t.to_string());
    }
    run_cloud_cli::<CloudTaskLogsResult>(
        project_root.as_deref(),
        root.as_deref(),
        python_executable.as_deref(),
        &subcmd,
    )
    .await
}

/// A task's telemetry (CR-12): resource samples, peaks, latest progress and per-day KPIs.
///
/// Returns the `cloud metrics --json` payload unchanged (the app's `CloudTaskMetrics`).
#[tauri::command]
pub async fn cloud_task_metrics(
    task_id: String,
    since: Option<f64>,
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
) -> Result<serde_json::Value, String> {
    if task_id.trim().is_empty() {
        return Err("task_id cannot be empty".to_string());
    }
    let mut subcmd = vec!["metrics".to_string(), task_id];
    if let Some(s) = since {
        subcmd.push("--since".to_string());
        subcmd.push(s.to_string());
    }
    run_cloud_cli::<serde_json::Value>(
        project_root.as_deref(),
        root.as_deref(),
        python_executable.as_deref(),
        &subcmd,
    )
    .await
}

/// Download all completed task result files for a job into local output directory.
#[tauri::command]
pub async fn cloud_download(
    job_id: String,
    dest: Option<String>,
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
    app: AppHandle,
) -> Result<CloudDownloadResult, String> {
    if job_id.trim().is_empty() {
        return Err("job_id cannot be empty".to_string());
    }
    execute_cloud_download(job_id, dest, project_root, root, python_executable, app).await
}

/// Verify a local file's SHA-256 hash against expected hexadecimal checksum.
#[tauri::command]
pub fn cloud_verify_download(file_path: String, expected_sha256: String) -> Result<bool, String> {
    let path = std::path::Path::new(&file_path);
    if !path.exists() {
        return Err(format!("File '{}' does not exist", file_path));
    }
    verify_file_sha256(path, &expected_sha256)
}

/// List available compute backends and their capabilities.
#[tauri::command]
pub async fn cloud_backends(
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
) -> Result<Vec<CloudBackendEntry>, String> {
    let subcmd = vec!["backends".to_string()];
    run_cloud_cli::<Vec<CloudBackendEntry>>(
        project_root.as_deref(),
        root.as_deref(),
        python_executable.as_deref(),
        &subcmd,
    )
    .await
}

/// Execute a single scheduler pass on the worker.
#[tauri::command]
pub async fn cloud_worker_tick(
    job: Option<String>,
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
) -> Result<CloudWorkerTickResult, String> {
    let mut subcmd = vec!["worker".to_string(), "--once".to_string()];
    if let Some(ref j) = job {
        subcmd.push("--job".to_string());
        subcmd.push(j.clone());
    }
    run_cloud_cli::<CloudWorkerTickResult>(
        project_root.as_deref(),
        root.as_deref(),
        python_executable.as_deref(),
        &subcmd,
    )
    .await
}

/// Securely store a provider secret credential in the OS keychain.
#[tauri::command]
pub fn cloud_store_credential(
    profile: String,
    provider: String,
    secret: String,
) -> Result<(), String> {
    store_credential(&profile, &provider, &secret)
}

/// Remove a stored credential from the OS keychain.
#[tauri::command]
pub fn cloud_delete_credential(profile: String, provider: String) -> Result<(), String> {
    delete_credential(&profile, &provider)
}

/// Check if a credential is configured in the OS keychain without revealing it.
#[tauri::command]
pub fn cloud_has_credential(profile: String, provider: String) -> Result<bool, String> {
    has_credential(&profile, &provider)
}

/// Start background status polling for a job, emitting `cloud://job-updated`.
#[tauri::command]
pub fn cloud_watch_job(
    job_id: String,
    interval_ms: Option<u64>,
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
    app: AppHandle,
) -> Result<(), String> {
    if job_id.trim().is_empty() {
        return Err("job_id cannot be empty".to_string());
    }
    let interval = interval_ms.unwrap_or(2000);
    get_default_poller().watch_job(
        job_id,
        interval,
        project_root,
        root,
        python_executable,
        app,
    )
}

/// Cancel status polling for a specific job.
#[tauri::command]
pub fn cloud_unwatch_job(job_id: String) -> Result<(), String> {
    get_default_poller().unwatch_job(&job_id)
}

/// Cancel all active job status polling tasks.
#[tauri::command]
pub fn cloud_stop_all_watchers() -> Result<(), String> {
    get_default_poller().stop_all()
}

/// Return all job IDs currently being polled.
#[tauri::command]
pub fn cloud_list_watched_jobs() -> Vec<String> {
    get_default_poller().watched_jobs()
}

#[cfg(test)]
mod tests;
