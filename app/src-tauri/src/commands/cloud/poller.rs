//! Background poller and status feed abstraction for cloud jobs.
//!
//! Emits `cloud://job-updated` events with `CloudJobStatusUpdateEvent` payloads.
//! Provides the local polling fallback for the SQLite control plane; designed so
//! Firestore and Supabase realtime (CR-04) can feed the same events later behind
//! the [`CloudJobWatcher`] trait boundary.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter};

use super::cli::run_cloud_cli;
use super::models::{CloudJobStatus, CloudJobStatusUpdateEvent};

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Common trait for status watchers / event feeds.
pub trait CloudJobWatcher: Send + Sync {
    /// Start watching a job. Status updates will be emitted as `cloud://job-updated`.
    fn watch_job(
        &self,
        job_id: String,
        interval_ms: u64,
        project_root: Option<String>,
        root: Option<String>,
        python_executable: Option<String>,
        app: AppHandle,
    ) -> Result<(), String>;

    /// Cancel watching a specific job.
    fn unwatch_job(&self, job_id: &str) -> Result<(), String>;

    /// Stop all active watchers.
    fn stop_all(&self) -> Result<(), String>;

    /// List currently watched job IDs.
    fn watched_jobs(&self) -> Vec<String>;
}

/// Polling-based implementation of [`CloudJobWatcher`] for the local control plane.
/// A watcher's cancel channel and generation; the generation keeps a finished
/// poller from unregistering a newer watcher of the same job.
type WatcherEntry = (u64, tokio::sync::watch::Sender<bool>);

pub struct LocalJobPoller {
    active_watchers: Arc<Mutex<HashMap<String, WatcherEntry>>>,
    next_generation: std::sync::atomic::AtomicU64,
}

impl Default for LocalJobPoller {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalJobPoller {
    pub fn new() -> Self {
        Self {
            active_watchers: Arc::new(Mutex::new(HashMap::new())),
            next_generation: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl CloudJobWatcher for LocalJobPoller {
    fn watch_job(
        &self,
        job_id: String,
        interval_ms: u64,
        project_root: Option<String>,
        root: Option<String>,
        python_executable: Option<String>,
        app: AppHandle,
    ) -> Result<(), String> {
        // Each poll starts a Python process; keep at least a second between them.
        let safe_interval = interval_ms.max(1000);
        let generation = self
            .next_generation
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Cancel previous watcher for this job if already running
        {
            let mut guard = self.active_watchers.lock().map_err(|e| e.to_string())?;
            if let Some((_, old_tx)) = guard.remove(&job_id) {
                let _ = old_tx.send(true);
            }
        }

        let (cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(false);
        {
            let mut guard = self.active_watchers.lock().map_err(|e| e.to_string())?;
            guard.insert(job_id.clone(), (generation, cancel_tx));
        }

        let registry = Arc::clone(&self.active_watchers);
        let task_job_id = job_id.clone();

        // Sync Tauri commands on Linux run inside WebKit's URI-scheme callback.
        // That thread has no Tokio context, so `tokio::spawn` panics here and the
        // process aborts because the callback cannot unwind.
        spawn_watcher(async move {
            let mut ticker =
                tokio::time::interval(tokio::time::Duration::from_millis(safe_interval));
            // A slow poll delays the next one instead of triggering a burst of catch-up polls.
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

            loop {
                tokio::select! {
                    _ = cancel_rx.changed() => {
                        if *cancel_rx.borrow() {
                            break;
                        }
                    }
                    _ = ticker.tick() => {
                        let subcmd = vec!["status".to_string(), task_job_id.clone()];
                        match run_cloud_cli::<CloudJobStatus>(
                            project_root.as_deref(),
                            root.as_deref(),
                            python_executable.as_deref(),
                            &subcmd,
                        ).await {
                            Ok(status) => {
                                let terminal = matches!(
                                    status.state.as_str(),
                                    "succeeded" | "failed" | "cancelled"
                                );
                                let event = CloudJobStatusUpdateEvent {
                                    job_id: task_job_id.clone(),
                                    status,
                                    source: "local_poller".to_string(),
                                    timestamp_ms: now_millis(),
                                };
                                let _ = app.emit("cloud://job-updated", &event);

                                if terminal {
                                    // Terminal state reached: unregister and exit poller loop
                                    break;
                                }
                            }
                            Err(err) => {
                                eprintln!("[cloud_poller] failed to query job '{task_job_id}': {err}");
                            }
                        }
                    }
                }
            }

            // Cleanup from active map
            if let Ok(mut guard) = registry.lock() {
                if guard.get(&task_job_id).map(|(g, _)| *g) == Some(generation) {
                    guard.remove(&task_job_id);
                }
            }
        });

        Ok(())
    }

    fn unwatch_job(&self, job_id: &str) -> Result<(), String> {
        let mut guard = self.active_watchers.lock().map_err(|e| e.to_string())?;
        if let Some((_, tx)) = guard.remove(job_id) {
            let _ = tx.send(true);
        }
        Ok(())
    }

    fn stop_all(&self) -> Result<(), String> {
        let mut guard = self.active_watchers.lock().map_err(|e| e.to_string())?;
        for (_, (_, tx)) in guard.drain() {
            let _ = tx.send(true);
        }
        Ok(())
    }

    fn watched_jobs(&self) -> Vec<String> {
        self.active_watchers
            .lock()
            .map(|g| g.keys().cloned().collect())
            .unwrap_or_default()
    }
}

/// Spawn a watcher on Tauri's runtime.
///
/// Safe to call from a thread that is not inside a Tokio runtime. `tokio::spawn`
/// is not: it requires a current reactor and panics on the GTK main thread.
pub(super) fn spawn_watcher<F>(task: F)
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    tauri::async_runtime::spawn(task);
}

static GLOBAL_POLLER: OnceLock<LocalJobPoller> = OnceLock::new();

pub fn get_default_poller() -> &'static LocalJobPoller {
    GLOBAL_POLLER.get_or_init(LocalJobPoller::new)
}
