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
mod tests {
    use super::keychain::make_account_key;
    use super::models::*;

    const BACKENDS_JSON: &str = include_str!("fixtures/backends.json");
    const JOBS_EMPTY_JSON: &str = include_str!("fixtures/jobs_empty.json");
    const JOBS_LIST_JSON: &str = include_str!("fixtures/jobs_list.json");
    const JOB_STATUS_JSON: &str = include_str!("fixtures/job_status.json");
    const SUBMIT_RESULT_JSON: &str = include_str!("fixtures/submit_result.json");
    const CANCEL_RESULT_JSON: &str = include_str!("fixtures/cancel_result.json");
    const RETRY_RESULT_JSON: &str = include_str!("fixtures/retry_result.json");
    const LOGS_RESULT_JSON: &str = include_str!("fixtures/logs_result.json");
    const DOWNLOAD_RESULT_JSON: &str = include_str!("fixtures/download_result.json");
    const WORKER_TICK_JSON: &str = include_str!("fixtures/worker_tick.json");

    #[test]
    fn test_parse_backends_fixture() {
        let backends: Vec<CloudBackendEntry> =
            serde_json::from_str(BACKENDS_JSON).expect("failed to parse backends.json");
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].provider, "local");
        assert_eq!(backends[0].compute, "process");
        assert_eq!(backends[0].capabilities.kinds, vec!["process"]);
        assert!(!backends[0].capabilities.supports_spot);
    }

    #[test]
    fn test_parse_empty_jobs_fixture() {
        let jobs: Vec<CloudJobSummary> =
            serde_json::from_str(JOBS_EMPTY_JSON).expect("failed to parse jobs_empty.json");
        assert!(jobs.is_empty());
    }

    #[test]
    fn test_parse_jobs_list_fixture() {
        let jobs: Vec<CloudJobSummary> =
            serde_json::from_str(JOBS_LIST_JSON).expect("failed to parse jobs_list.json");
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].name, "test-job");
        assert_eq!(jobs[0].state, "queued");
        assert_eq!(jobs[0].counts.get("queued"), Some(&1));
    }

    #[test]
    fn test_parse_job_status_fixture() {
        let status: CloudJobStatus =
            serde_json::from_str(JOB_STATUS_JSON).expect("failed to parse job_status.json");
        assert_eq!(status.name, "test-job");
        assert_eq!(status.state, "queued");
        assert_eq!(status.tasks.len(), 1);
        assert_eq!(status.tasks[0].cell_id, "cell");
        assert_eq!(status.tasks[0].state, "queued");
        assert_eq!(status.tasks[0].attempt, 1);
    }

    #[test]
    fn test_parse_submit_result_fixture() {
        let res: CloudSubmitResult =
            serde_json::from_str(SUBMIT_RESULT_JSON).expect("failed to parse submit_result.json");
        assert!(res.job_id.starts_with("job-"));
        assert_eq!(res.tasks, 1);
    }

    #[test]
    fn test_parse_cancel_result_fixture() {
        let res: CloudCancelResult =
            serde_json::from_str(CANCEL_RESULT_JSON).expect("failed to parse cancel_result.json");
        assert!(res.job_id.starts_with("job-"));
        assert!(res.cancel_requested);
    }

    #[test]
    fn test_parse_retry_result_fixture() {
        let res: CloudRetryResult =
            serde_json::from_str(RETRY_RESULT_JSON).expect("failed to parse retry_result.json");
        assert!(res.job_id.starts_with("job-"));
        assert_eq!(res.retried, 1);
    }

    #[test]
    fn test_parse_logs_result_fixture() {
        let res: CloudTaskLogsResult =
            serde_json::from_str(LOGS_RESULT_JSON).expect("failed to parse logs_result.json");
        assert!(res.task_id.starts_with("task-"));
        assert!(res.log.contains("Task started"));
    }

    #[test]
    fn test_parse_download_result_fixture() {
        let res: CloudDownloadResult =
            serde_json::from_str(DOWNLOAD_RESULT_JSON).expect("failed to parse download_result.json");
        assert!(res.job_id.starts_with("job-"));
        assert_eq!(res.files, 3);
    }

    #[test]
    fn test_parse_worker_tick_fixture() {
        let res: CloudWorkerTickResult =
            serde_json::from_str(WORKER_TICK_JSON).expect("failed to parse worker_tick.json");
        assert_eq!(res.changes, 1);
    }

    #[test]
    fn test_keychain_account_key_convention() {
        assert_eq!(make_account_key("GCP", "default"), "gcp/default");
        assert_eq!(make_account_key("  aws ", " prod_1  "), "aws/prod_1");
        assert_eq!(make_account_key("Azure", "staging-cluster"), "azure/staging-cluster");
    }

    #[test]
    fn test_file_sha256_verification() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("wsr_cloud_test_{unique}.txt"));
        std::fs::write(&path, b"WSmart-Route Cloud Runner verification test")
            .expect("write test file");

        let hash = super::download::compute_file_sha256(&path).expect("hash");
        assert_eq!(hash.len(), 64);
        assert!(super::download::verify_file_sha256(&path, &hash).expect("verify"));
        assert!(!super::download::verify_file_sha256(
            &path,
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        .expect("verify mismatch"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_poller_event_serialization() {
        let event = CloudJobStatusUpdateEvent {
            job_id: "job-123".to_string(),
            status: CloudJobStatus {
                job_id: "job-123".to_string(),
                name: "sim-test".to_string(),
                state: "running".to_string(),
                counts: [("running".to_string(), 1)].into_iter().collect(),
                provider: Some("aws".to_string()),
                compute: Some("batch".to_string()),
                spot: Some(true),
                machine_type: Some("c6i.large".to_string()),
                region: Some("us-east-1".to_string()),
                price_per_hour: Some(0.034),
                on_demand_price_per_hour: Some(0.085),
                tasks: vec![CloudTaskSummary {
                    task_id: "task-456".to_string(),
                    cell_id: "c0".to_string(),
                    state: "running".to_string(),
                    attempt: 1,
                    message: None,
                    started_at: Some(1729345670.0),
                    finished_at: None,
                    duration: Some(8.0),
                    expected_wall_seconds: Some(120.0),
                    policy: Some("bpc".to_string()),
                    cost: Some(0.000076),
                }],
            },
            source: "local_poller".to_string(),
            timestamp_ms: 1729345678000,
        };

        let json = serde_json::to_string(&event).expect("serialize event");
        let deserialized: CloudJobStatusUpdateEvent =
            serde_json::from_str(&json).expect("deserialize event");
        assert_eq!(event, deserialized);
    }

    #[tokio::test]
    async fn test_submit_request_validation_missing_source() {
        let req = CloudSubmitRequest::default();
        let err = super::cloud_submit(req, None, None, None)
            .await
            .expect_err("should fail with missing source");
        assert!(err.contains("requires either 'manifest' or 'command'"));
    }

    #[tokio::test]
    async fn test_empty_id_validation() {
        let err = super::cloud_job_status("   ".to_string(), None, None, None)
            .await
            .expect_err("should fail with empty job_id");
        assert!(err.contains("job_id cannot be empty"));

        let err = super::cloud_cancel("".to_string(), None, None, None)
            .await
            .expect_err("should fail with empty job_id");
        assert!(err.contains("job_id cannot be empty"));

        let err = super::cloud_retry("".to_string(), None, None, None)
            .await
            .expect_err("should fail with empty job_id");
        assert!(err.contains("job_id cannot be empty"));

        let err = super::cloud_task_logs("".to_string(), None, None, None, None)
            .await
            .expect_err("should fail with empty task_id");
        assert!(err.contains("task_id cannot be empty"));
    }

    #[test]
    fn test_keychain_validation_empty_inputs() {
        assert!(super::store_credential("", "gcp", "secret").is_err());
        assert!(super::store_credential("my-profile", "", "secret").is_err());
        assert!(super::store_credential("my-profile", "gcp", "").is_err());
        assert!(super::delete_credential("", "gcp").is_err());
        assert!(super::delete_credential("my-profile", "").is_err());
        assert_eq!(super::has_credential("", "gcp").unwrap(), false);
        assert_eq!(super::has_credential("my-profile", "").unwrap(), false);
    }

    #[test]
    fn test_poller_registration_lifecycle() {
        use super::poller::CloudJobWatcher;
        let poller = super::poller::LocalJobPoller::new();
        assert!(poller.watched_jobs().is_empty());

        // Unwatch non-existent job does not error
        assert!(poller.unwatch_job("job-fake").is_ok());

        // Stop all on empty does not error
        assert!(poller.stop_all().is_ok());
    }

    #[test]
    fn test_malformed_json_error_handling() {
        let malformed = "{ \"job_id\": \"job-1\", unclosed json";
        let parse_result = serde_json::from_str::<CloudJobStatus>(malformed);
        assert!(parse_result.is_err());
    }
}
