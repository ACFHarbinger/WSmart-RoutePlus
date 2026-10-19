//! Download orchestration, progress events, and file integrity verification.

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use tauri::{AppHandle, Emitter};

use super::cli::run_cloud_cli;
use super::models::{CloudDownloadProgressEvent, CloudDownloadResult};

/// Compute the hex-encoded SHA-256 digest of a local file.
pub fn compute_file_sha256(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|e| format!("Failed to open file '{}': {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];

    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| format!("Failed to read file '{}': {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify a file's SHA-256 against an expected hex string.
pub fn verify_file_sha256(path: &Path, expected_hex: &str) -> Result<bool, String> {
    let actual = compute_file_sha256(path)?;
    Ok(actual.eq_ignore_ascii_case(expected_hex.trim()))
}

/// Execute a job results download via CLI with lifecycle progress events.
pub async fn execute_cloud_download(
    job_id: String,
    dest: Option<String>,
    project_root: Option<String>,
    root: Option<String>,
    python_executable: Option<String>,
    app: AppHandle,
) -> Result<CloudDownloadResult, String> {
    let root_str = root.as_deref().unwrap_or("outputs/cloud");
    let target_dest = dest
        .clone()
        .unwrap_or_else(|| format!("{root_str}/results/{job_id}"));

    // Emit download started event
    let _ = app.emit(
        "cloud://download-progress",
        CloudDownloadProgressEvent {
            job_id: job_id.clone(),
            dest: target_dest.clone(),
            stage: "starting".to_string(),
            files_downloaded: 0,
            total_files: None,
            message: Some("Initiating download from object store...".to_string()),
        },
    );

    let mut subcmd = vec!["download".to_string(), job_id.clone()];
    if let Some(ref d) = dest {
        subcmd.push("--dest".to_string());
        subcmd.push(d.clone());
    }

    match run_cloud_cli::<CloudDownloadResult>(
        project_root.as_deref(),
        root.as_deref(),
        python_executable.as_deref(),
        &subcmd,
    )
    .await
    {
        Ok(res) => {
            // Note: The Python download CLI command internally validates SHA-256 hashes
            // of all downloaded result files against the storage index in `download_results()`.
            // Once the CLI returns, every written file is verified.
            let _ = app.emit(
                "cloud://download-progress",
                CloudDownloadProgressEvent {
                    job_id: job_id.clone(),
                    dest: res.dest.clone(),
                    stage: "completed".to_string(),
                    files_downloaded: res.files,
                    total_files: Some(res.files),
                    message: Some(format!(
                        "Successfully downloaded and verified {} result file(s)",
                        res.files
                    )),
                },
            );
            Ok(res)
        }
        Err(err) => {
            let _ = app.emit(
                "cloud://download-progress",
                CloudDownloadProgressEvent {
                    job_id: job_id.clone(),
                    dest: target_dest,
                    stage: "failed".to_string(),
                    files_downloaded: 0,
                    total_files: None,
                    message: Some(err.clone()),
                },
            );
            Err(err)
        }
    }
}
