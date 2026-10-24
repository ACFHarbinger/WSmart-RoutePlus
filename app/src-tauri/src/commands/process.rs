/// Process management commands: spawn Python subprocesses, stream stdout, cancel.
///
/// Replaces the Streamlit file-tailer pattern with event-driven stdout streaming.
/// Each spawned process receives a unique `id`; stdout lines are emitted as
/// `process:stdout` events and status changes as `process:status` events.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ProcessStatus {
    Running,
    Completed,
    Cancelled,
    Failed,
}

/// Emitted immediately after a process is spawned so the frontend can
/// register the process in its store before any stdout lines arrive.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProcessSpawned {
    pub id: String,
    pub command: String,
    pub pid: u32,
    pub start_time: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StdoutBatch {
    pub id: String,
    pub lines: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StatusUpdate {
    pub id: String,
    pub status: ProcessStatus,
    pub exit_code: Option<i32>,
}

type Registry = Arc<Mutex<HashMap<String, (u32, tokio::sync::watch::Sender<bool>)>>>;

static PROCESS_REGISTRY: std::sync::OnceLock<Registry> = std::sync::OnceLock::new();

fn registry() -> &'static Registry {
    PROCESS_REGISTRY.get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
}

#[tauri::command]
pub async fn spawn_python_process(
    id: String,
    python_args: Vec<String>,
    working_dir: String,
    python_executable: Option<String>,
    app: AppHandle,
) -> Result<u32, String> {
    let python = resolve_python(&working_dir, python_executable);
    let mut cmd = Command::new(&python);
    cmd.args(&python_args)
        .current_dir(&working_dir)
        // The simulator streams live policy telemetry (POLICY_VIZ_START lines) only to the Studio
        .env("WSR_STUDIO", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(false);

    let mut child = cmd.spawn().map_err(|e| format!("spawn failed: {e}"))?;
    let pid = child.id().unwrap_or(0);

    let (cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(false);
    {
        let mut reg = registry().lock().unwrap();
        reg.insert(id.clone(), (pid, cancel_tx));
    }

    // Notify the frontend immediately so it can register this process in its store
    let command_str = format!("{} {}", python, python_args.join(" "));
    let start_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let _ = app.emit(
        "process:spawn",
        ProcessSpawned {
            id: id.clone(),
            command: command_str,
            pid,
            start_time,
        },
    );

    let task_id = id.clone();
    let task_app = app.clone();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    tokio::spawn(async move {
        let (line_tx, mut line_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let out_handle = stdout.map(|out| {
            let tx = line_tx.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(out).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx.send(line);
                }
            })
        });

        let err_handle = stderr.map(|err| {
            let tx = line_tx.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(err).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx.send(format!("[stderr] {line}"));
                }
            })
        });
        drop(line_tx);

        let batch_app = task_app.clone();
        let batch_id = task_id.clone();
        let drain_handle = tokio::spawn(async move {
            let mut batch: Vec<String> = Vec::with_capacity(50);
            let mut ticker = tokio::time::interval(tokio::time::Duration::from_millis(50));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    maybe_line = line_rx.recv() => {
                        match maybe_line {
                            Some(line) => {
                                batch.push(line);
                                if batch.len() >= 50 {
                                    let lines = std::mem::replace(&mut batch, Vec::with_capacity(50));
                                    let _ = batch_app.emit(
                                        "process:stdout_batch",
                                        StdoutBatch { id: batch_id.clone(), lines },
                                    );
                                }
                            }
                            None => {
                                if !batch.is_empty() {
                                    let _ = batch_app.emit(
                                        "process:stdout_batch",
                                        StdoutBatch { id: batch_id.clone(), lines: batch },
                                    );
                                }
                                break;
                            }
                        }
                    }
                    _ = ticker.tick() => {
                        if !batch.is_empty() {
                            let lines = std::mem::replace(&mut batch, Vec::with_capacity(50));
                            let _ = batch_app.emit(
                                "process:stdout_batch",
                                StdoutBatch { id: batch_id.clone(), lines },
                            );
                        }
                    }
                }
            }
        });

        let exit_status = tokio::select! {
            status = child.wait() => status.ok(),
            _ = cancel_rx.changed() => {
                let _ = child.kill().await;
                None
            }
        };

        if let Some(h) = out_handle {
            let _ = h.await;
        }
        if let Some(h) = err_handle {
            let _ = h.await;
        }
        let _ = drain_handle.await;

        let (status, code) = match exit_status {
            Some(s) if s.success() => (ProcessStatus::Completed, Some(0)),
            Some(s) => (
                ProcessStatus::Failed,
                s.code(),
            ),
            None => (ProcessStatus::Cancelled, None),
        };

        {
            let mut reg = registry().lock().unwrap();
            reg.remove(&task_id);
        }

        let _ = task_app.emit(
            "process:status",
            StatusUpdate { id: task_id, status, exit_code: code },
        );
    });

    Ok(pid)
}

#[tauri::command]
pub fn cancel_process(id: String) -> Result<(), String> {
    let mut reg = registry().lock().unwrap();
    if let Some((_, tx)) = reg.remove(&id) {
        let _ = tx.send(true);
        Ok(())
    } else {
        Err(format!("No running process with id '{id}'"))
    }
}

#[tauri::command]
pub fn list_processes() -> Vec<String> {
    registry()
        .lock()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

/// Resolve the Python executable to use for spawning subprocesses.
///
/// Priority order:
///   1. Explicit `python_executable` override (non-empty)
///   2. `<working_dir>/.venv/bin/python`  — uv-managed venv (Linux/macOS)
///   3. `<working_dir>/.venv/Scripts/python.exe` — uv-managed venv (Windows)
///   4. `python3` / `python` on PATH — system fallback
pub fn resolve_python(working_dir: &str, python_executable: Option<String>) -> String {
    python_executable
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| which_python(working_dir))
}

/// Resolve the Python executable to use for spawning subprocesses.
fn which_python(working_dir: &str) -> String {
    let candidates = [
        format!("{working_dir}/.venv/bin/python"),
        format!("{working_dir}/.venv/Scripts/python.exe"),
    ];
    for candidate in &candidates {
        if std::path::Path::new(candidate).exists() {
            return candidate.clone();
        }
    }
    for candidate in &["python3", "python"] {
        if std::process::Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok()
        {
            return candidate.to_string();
        }
    }
    "python3".to_string()
}
