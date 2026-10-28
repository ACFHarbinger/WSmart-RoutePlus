/// System inspection commands: validate paths, probe Python version, Hydra config dump.
use std::process::Command;

use super::process::resolve_python;

#[cfg(desktop)]
use std::sync::Mutex;

#[cfg(desktop)]
use tauri::{AppHandle, State};

#[cfg(desktop)]
use tauri_plugin_updater::{Update, UpdaterExt};

/// Holds a signed update discovered by `check_for_updates` until `install_app_update` runs.
#[cfg(desktop)]
pub struct PendingUpdate(pub Mutex<Option<Update>>);

/// Return the Studio app version from Cargo.toml (§G.8 / §G.19).
#[tauri::command]
pub fn get_app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[derive(serde::Serialize)]
pub struct UpdateCheckResult {
    pub available: bool,
    pub current_version: String,
    pub latest_version: Option<String>,
    pub message: String,
    /// True when the Tauri updater plugin can download and install a signed artefact.
    pub can_install: bool,
    pub notes: Option<String>,
}

#[cfg(desktop)]
fn updater_env_configured() -> Option<(String, String)> {
    let pubkey = std::env::var("WSMART_UPDATER_PUBKEY")
        .ok()
        .filter(|s| !s.trim().is_empty())?;
    let url = std::env::var("WSMART_UPDATE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())?;
    Some((pubkey, url))
}

#[cfg(desktop)]
async fn check_signed_update(
    app: &AppHandle,
    pending: &PendingUpdate,
) -> Result<Option<UpdateCheckResult>, String> {
    let (pubkey, url) = match updater_env_configured() {
        Some(cfg) => cfg,
        None => return Ok(None),
    };

    let endpoint = url
        .parse()
        .map_err(|e| format!("Invalid WSMART_UPDATE_URL: {e}"))?;

    let update = app
        .updater_builder()
        .pubkey(pubkey)
        .endpoints(vec![endpoint])
        .map_err(|e| format!("Updater endpoints invalid: {e}"))?
        .build()
        .map_err(|e| format!("Updater build failed: {e}"))?
        .check()
        .await
        .map_err(|e| format!("Signed update check failed: {e}"))?;

    let Some(found) = update else {
        let current = env!("CARGO_PKG_VERSION").to_string();
        return Ok(Some(UpdateCheckResult {
            available: false,
            current_version: current,
            latest_version: None,
            message: "You are on the latest signed release".to_string(),
            can_install: false,
            notes: None,
        }));
    };

    let result = UpdateCheckResult {
        available: true,
        current_version: found.current_version.clone(),
        latest_version: Some(found.version.clone()),
        message: "A signed update is available".to_string(),
        can_install: true,
        notes: found.body.clone(),
    };

    *pending.0.lock().map_err(|e| e.to_string())? = Some(found);
    Ok(Some(result))
}

#[cfg(not(desktop))]
async fn check_signed_update(
    _app: &tauri::AppHandle,
    _pending: &(),
) -> Result<Option<UpdateCheckResult>, String> {
    Ok(None)
}

async fn check_manifest_update() -> Result<UpdateCheckResult, String> {
    let current = env!("CARGO_PKG_VERSION").to_string();
    let url = std::env::var("WSMART_UPDATE_URL").unwrap_or_default();

    if url.is_empty() {
        return Ok(UpdateCheckResult {
            available: false,
            current_version: current,
            latest_version: None,
            message: "Update server not configured (set WSMART_UPDATE_URL)".to_string(),
            can_install: false,
            notes: None,
        });
    }

    let body = reqwest::get(&url)
        .await
        .map_err(|e| format!("Update check failed: {e}"))?
        .text()
        .await
        .map_err(|e| format!("Failed to read update manifest: {e}"))?;

    let manifest: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("Invalid update manifest: {e}"))?;
    let latest = manifest
        .get("version")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let notes = manifest
        .get("notes")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let available = latest.as_ref().is_some_and(|l| l != &current);

    Ok(UpdateCheckResult {
        available,
        current_version: current,
        latest_version: latest,
        message: if available {
            "A newer version is available".to_string()
        } else {
            "You are on the latest version".to_string()
        },
        can_install: false,
        notes,
    })
}

/// Check for app updates (§G.8).
///
/// When `WSMART_UPDATER_PUBKEY` and `WSMART_UPDATE_URL` are both set, uses the Tauri
/// updater plugin for signed artefact verification. Otherwise falls back to a simple
/// JSON manifest version comparison on `WSMART_UPDATE_URL`.
#[tauri::command]
pub async fn check_for_updates(
    app: AppHandle,
    #[cfg(desktop)] pending: State<'_, PendingUpdate>,
) -> Result<UpdateCheckResult, String> {
    #[cfg(desktop)]
    {
        if let Some(signed) = check_signed_update(&app, &pending).await? {
            return Ok(signed);
        }
    }

    check_manifest_update().await
}

/// Download and install a signed update previously returned by `check_for_updates`.
#[cfg(desktop)]
#[tauri::command]
pub async fn install_app_update(
    app: AppHandle,
    pending: State<'_, PendingUpdate>,
) -> Result<(), String> {
    let update = pending
        .0
        .lock()
        .map_err(|e| e.to_string())?
        .take()
        .ok_or_else(|| {
            "No pending signed update — configure WSMART_UPDATER_PUBKEY and check again".to_string()
        })?;

    update
        .download_and_install(|_chunk_length, _content_length| {}, || {})
        .await
        .map_err(|e| format!("Update install failed: {e}"))?;

    app.restart();
}

#[cfg(not(desktop))]
#[tauri::command]
pub async fn install_app_update() -> Result<(), String> {
    Err("Signed updates are only supported on desktop targets".to_string())
}

/// Check that `path` is a directory containing `main.py`.
/// Returns an Ok confirmation or a descriptive error.
#[tauri::command]
pub fn validate_project_root(path: String) -> Result<String, String> {
    let dir = std::path::Path::new(&path);
    if !dir.exists() {
        return Err(format!("Directory does not exist: {path}"));
    }
    if !dir.is_dir() {
        return Err(format!("Path is not a directory: {path}"));
    }
    if !dir.join("main.py").exists() {
        return Err(format!("main.py not found in {path}"));
    }
    Ok(format!("main.py found in {path}"))
}

/// Run `<python_path> --version` synchronously and return the version string.
/// Works on both Python 2 (stderr) and Python 3 (stdout).
#[tauri::command]
pub fn probe_python(python_path: String) -> Result<String, String> {
    let output = Command::new(&python_path)
        .arg("--version")
        .output()
        .map_err(|e| format!("Failed to run '{python_path}': {e}"))?;

    // Python 3 prints to stdout; Python 2 prints to stderr.
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let version = if !stdout.is_empty() { stdout } else { stderr };

    if version.is_empty() {
        return Err(format!("'{python_path} --version' produced no output"));
    }
    Ok(version)
}

/// Run `python main.py <task> --cfg job` and return the resolved Hydra config as YAML.
#[tauri::command]
pub fn dump_hydra_config(
    task: String,
    project_root: String,
    python_executable: Option<String>,
) -> Result<String, String> {
    let python = resolve_python(&project_root, python_executable);
    let output = Command::new(&python)
        .args(["main.py", &task, "--cfg", "job"])
        .current_dir(&project_root)
        .output()
        .map_err(|e| format!("Failed to run Hydra config dump: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        let msg = if !stderr.is_empty() { stderr } else { stdout };
        return Err(format!("Hydra --cfg job failed: {msg}"));
    }

    if stdout.trim().is_empty() {
        return Err("Hydra --cfg job produced no output".to_string());
    }

    Ok(stdout)
}
