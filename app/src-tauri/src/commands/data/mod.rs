/// Data loading commands for simulation logs, CSVs, and training metrics.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::commands::sim_watcher::{parse_policy_viz_line, DayLogEntry, PolicyVizEntry, SimFailureEntry};

pub mod bundle;
pub mod files;
pub mod parquet;
pub mod sim_log;

#[cfg(test)]
mod tests;

pub use bundle::*;
pub use files::*;
pub use parquet::*;
pub use sim_log::SimulationDirectoryCatalog;

/// Matches the `CsvFile` TypeScript interface in DataExplorer.tsx
#[derive(Debug, Serialize, Deserialize)]
pub struct CsvFile {
    pub path: String,
    pub headers: Vec<String>,
    pub rows: Vec<HashMap<String, serde_json::Value>>,
}

/// Matches the `OutputDir` TypeScript interface in ExperimentTracker.tsx
#[derive(Debug, Serialize, Deserialize)]
pub struct OutputDir {
    pub name: String,
    pub path: String,
    pub created_at: String,
    pub size_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingRun {
    pub name: String,
    pub path: String,
    pub has_metrics: bool,
    pub has_hparams: bool,
}

/// Training instability alert from ``TrainingHealthCallback`` (§A.4).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingHealthEntry {
    pub code: String,
    pub severity: String,
    pub epoch: i32,
    pub step: i32,
    pub message: String,
    pub details: serde_json::Value,
}

/// Runtime encoder attention snapshot from ``AttentionRingBuffer`` (§A.2 Option A).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AttentionSnapshot {
    pub layer: i32,
    pub head: i32,
    pub decode_step: i32,
    pub n_nodes: i32,
    pub matrix: Vec<Vec<f64>>,
}

/// Attention ring-buffer payload emitted during eval/validation (§A.2 Option A).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AttentionVizEntry {
    pub phase: String,
    pub epoch: i32,
    pub step: i32,
    pub snapshots: Vec<AttentionSnapshot>,
}

pub fn parse_training_health_line(line: &str) -> Option<TrainingHealthEntry> {
    let line = line.trim();
    let prefix = "TRAINING_HEALTH_START:";
    let idx = line.find(prefix)?;
    let payload = &line[idx + prefix.len()..];
    serde_json::from_str(payload).ok()
}

pub fn parse_attention_viz_line(line: &str) -> Option<AttentionVizEntry> {
    let line = line.trim();
    let prefix = "ATTENTION_VIZ_START:";
    let idx = line.find(prefix)?;
    let payload = &line[idx + prefix.len()..];
    serde_json::from_str(payload).ok()
}

#[tauri::command]
pub fn load_simulation_log(path: String) -> Result<Vec<DayLogEntry>, String> {
    let p = Path::new(&path);
    if !p.exists() {
        return Err(format!("File or directory not found: {path}"));
    }
    if p.is_dir() {
        let files = sim_log::find_simulation_logs_in_dir(p)?;
        if files.is_empty() {
            return Err(format!("No simulation logs found under directory: {path}"));
        }
        let mut all_entries = Vec::new();
        for file in files {
            if let Ok(entries) = sim_log::load_single_simulation_log(&file) {
                all_entries.extend(entries);
            }
        }
        all_entries.sort_by(|a, b| {
            a.policy
                .cmp(&b.policy)
                .then(a.sample_id.cmp(&b.sample_id))
                .then(a.day.cmp(&b.day))
        });
        return Ok(all_entries);
    }
    sim_log::load_single_simulation_log(p)
}

#[tauri::command]
pub fn load_policy_viz_log(path: String) -> Result<Vec<PolicyVizEntry>, String> {
    let p = Path::new(&path);
    if !p.exists() {
        return Ok(vec![]);
    }
    if p.is_dir() {
        let files = sim_log::find_simulation_logs_in_dir(p)?;
        let mut all_viz = Vec::new();
        for file in files {
            if let Ok(content) = std::fs::read_to_string(&file) {
                let entries: Vec<PolicyVizEntry> = content
                    .lines()
                    .filter_map(parse_policy_viz_line)
                    .collect();
                all_viz.extend(entries);
            }
        }
        return Ok(all_viz);
    }
    let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let entries: Vec<PolicyVizEntry> = content.lines().filter_map(parse_policy_viz_line).collect();
    Ok(entries)
}

#[tauri::command]
pub fn load_sim_failure_log(path: String) -> Result<Vec<SimFailureEntry>, String> {
    let p = Path::new(&path);
    if !p.exists() {
        return Ok(vec![]);
    }
    if p.is_dir() {
        let files = sim_log::find_simulation_logs_in_dir(p)?;
        let mut all_failures = Vec::new();
        for file in files {
            if let Ok(entries) = sim_log::load_single_sim_failure_log(&file) {
                all_failures.extend(entries);
            }
        }
        return Ok(all_failures);
    }
    sim_log::load_single_sim_failure_log(p)
}

#[tauri::command]
pub fn scan_simulation_directory(dir_path: String) -> Result<SimulationDirectoryCatalog, String> {
    let p = Path::new(&dir_path);
    if !p.is_dir() {
        return Err(format!("Path is not a directory: {dir_path}"));
    }
    sim_log::scan_simulation_directory_tree(p)
}

#[tauri::command]
pub fn load_training_health_log(path: String) -> Result<Vec<TrainingHealthEntry>, String> {
    let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let entries: Vec<TrainingHealthEntry> = content
        .lines()
        .filter_map(parse_training_health_line)
        .collect();
    Ok(entries)
}

#[tauri::command]
pub fn load_attention_viz_log(path: String) -> Result<Vec<AttentionVizEntry>, String> {
    let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let entries: Vec<AttentionVizEntry> = content
        .lines()
        .filter_map(parse_attention_viz_line)
        .collect();
    Ok(entries)
}

#[tauri::command]
pub fn load_csv_file(path: String) -> Result<CsvFile, String> {
    let mut reader = csv::Reader::from_path(&path).map_err(|e| e.to_string())?;
    let header_record = reader.headers().map_err(|e| e.to_string())?.clone();
    let headers: Vec<String> = header_record.iter().map(|h| h.to_string()).collect();

    let mut rows: Vec<HashMap<String, serde_json::Value>> = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| e.to_string())?;
        let mut row = HashMap::new();
        for (header, field) in headers.iter().zip(record.iter()) {
            let value = if let Ok(n) = field.parse::<f64>() {
                serde_json::Value::Number(
                    serde_json::Number::from_f64(n).unwrap_or(serde_json::Number::from(0)),
                )
            } else {
                serde_json::Value::String(field.to_string())
            };
            row.insert(header.clone(), value);
        }
        rows.push(row);
    }
    Ok(CsvFile {
        path,
        headers,
        rows,
    })
}

#[tauri::command]
pub fn list_output_dirs(output_path: String) -> Result<Vec<OutputDir>, String> {
    let base = Path::new(&output_path);
    if !base.exists() {
        return Ok(vec![]);
    }

    let mut dirs: Vec<OutputDir> = std::fs::read_dir(base)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| {
            let path = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            let meta = std::fs::metadata(&path).ok();
            let created_at = meta
                .as_ref()
                .and_then(|m| m.created().ok())
                .and_then(|t| {
                    t.duration_since(std::time::UNIX_EPOCH)
                        .ok()
                        .map(|d| d.as_secs())
                })
                .map(|s| {
                    let dt = s;
                    format!("{}", dt)
                })
                .unwrap_or_default();

            // Sum size of all files in the dir
            let size_bytes: u64 = std::fs::read_dir(&path)
                .map(|rd| {
                    rd.filter_map(|f| f.ok())
                        .filter_map(|f| std::fs::metadata(f.path()).ok())
                        .map(|m| m.len())
                        .sum()
                })
                .unwrap_or(0);

            OutputDir {
                name,
                path: path.to_string_lossy().to_string(),
                created_at,
                size_bytes,
            }
        })
        .collect();

    dirs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(dirs)
}

#[tauri::command]
pub fn list_training_runs(logs_path: String) -> Result<Vec<TrainingRun>, String> {
    let base = Path::new(&logs_path);
    if !base.exists() {
        return Ok(vec![]);
    }

    let mut runs: Vec<TrainingRun> = std::fs::read_dir(base)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| {
            let path = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            let has_metrics = path.join("metrics.csv").exists();
            let has_hparams = path.join("hparams.yaml").exists();
            TrainingRun {
                name,
                path: path.to_string_lossy().to_string(),
                has_metrics,
                has_hparams,
            }
        })
        .collect();

    runs.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(runs)
}

#[tauri::command]
pub fn load_training_metrics(
    run_path: String,
) -> Result<Vec<HashMap<String, serde_json::Value>>, String> {
    let metrics_path = Path::new(&run_path).join("metrics.csv");
    if !metrics_path.exists() {
        return Ok(vec![]);
    }
    load_csv_file(metrics_path.to_string_lossy().to_string()).map(|f| f.rows)
}
