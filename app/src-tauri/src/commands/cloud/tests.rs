use super::keychain::make_account_key;
use super::models::*;
use super::poller::CloudJobWatcher;

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
    assert_eq!(
        make_account_key("Azure", "staging-cluster"),
        "azure/staging-cluster"
    );
}

#[test]
fn test_file_sha256_verification() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("wsr_cloud_test_{unique}.txt"));
    std::fs::write(&path, b"WSmart-Route Cloud Runner verification test").expect("write test file");

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
            cost_source: Some("scheduler_elapsed_price_hint".to_string()),
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
fn test_poller_spawn_outside_tokio_context() {
    // This thread has no Tokio reactor, matching a sync command on the GTK thread.
    let (tx, rx) = std::sync::mpsc::channel();
    super::poller::spawn_watcher(async move {
        let _ = tx.send(());
    });
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .expect("watcher task did not start");
}

#[test]
fn test_poller_registration_lifecycle() {
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
