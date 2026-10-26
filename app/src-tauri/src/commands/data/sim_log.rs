use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::commands::sim_watcher::{
    parse_day_log_line, parse_sim_failure_line, DayLogEntry, SimFailureEntry,
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DiscoveredSimulationLog {
    pub path: String,
    pub relative_path: String,
    pub days: Option<i32>,
    pub region: Option<String>,
    pub num_bins: Option<i32>,
    pub waste_type: Option<String>,
    pub distribution: Option<String>,
    pub run_name: Option<String>,
    pub policy: String,
    pub num_instances: Option<i32>,
    pub file_size: u64,
    pub is_realtime: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DiscoveredSimulationRun {
    pub id: String,
    pub label: String,
    pub days: Option<i32>,
    pub region: Option<String>,
    pub num_bins: Option<i32>,
    pub waste_type: Option<String>,
    pub distribution: Option<String>,
    pub run_name: Option<String>,
    pub logs: Vec<DiscoveredSimulationLog>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SimulationDirectoryCatalog {
    pub root_path: String,
    pub total_logs: usize,
    pub runs: Vec<DiscoveredSimulationRun>,
}

/// Known stochastic waste fill rate distributions used in the simulator.
pub fn is_known_distribution(s: &str) -> bool {
    let lower = s.to_lowercase();
    matches!(
        lower.as_str(),
        "emp"
            | "empirical"
            | "gamma1"
            | "gamma2"
            | "gamma3"
            | "gamma"
            | "poisson"
            | "normal"
            | "gaussian"
            | "uniform"
            | "beta"
            | "weibull"
            | "lognormal"
    )
}

/// Extract clean policy name from a log filename (e.g. `log_last_minute_cf90_bpc_custom_ftsp_1N.json`
/// -> `last_minute_cf90_bpc_custom_ftsp`).
pub fn extract_policy_name_from_filename(filename: &str) -> String {
    let mut stem = filename;
    if let Some(pos) = stem.rfind('.') {
        stem = &stem[..pos];
    }
    if let Some(stripped) = stem.strip_prefix("log_") {
        stem = stripped;
    }
    // Strip trailing _<digits>N (e.g. _1N, _10N)
    if let Some(idx) = stem.rfind('_') {
        let suffix = &stem[idx + 1..];
        if suffix.ends_with('N') && suffix[..suffix.len() - 1].chars().all(|c| c.is_ascii_digit()) {
            stem = &stem[..idx];
        }
    }
    if stem.is_empty() {
        "simulation_policy".to_string()
    } else {
        stem.to_string()
    }
}

/// Extract sample instance count N from filename (e.g. `log_alns_10N.json` -> 10).
pub fn extract_instance_count_from_filename(filename: &str) -> Option<i32> {
    let mut stem = filename;
    if let Some(pos) = stem.rfind('.') {
        stem = &stem[..pos];
    }
    if let Some(idx) = stem.rfind('_') {
        let suffix = &stem[idx + 1..];
        if let Some(num_str) = suffix.strip_suffix('N') {
            if let Ok(n) = num_str.parse::<i32>() {
                return Some(n);
            }
        }
    }
    None
}

/// Check if a path corresponds to a simulation log file.
pub fn is_simulation_log_file(path: &Path) -> bool {
    let filename = match path.file_name().and_then(|f| f.to_str()) {
        Some(f) => f.to_lowercase(),
        None => return false,
    };
    if filename.starts_with('.') {
        return false;
    }
    if filename.ends_with(".jsonl") {
        return true;
    }
    if filename.ends_with(".json") && (filename.starts_with("log_") || filename.contains("_n.json")) {
        return true;
    }
    false
}

/// Parse simulation directory metadata according to the standard simulator hierarchy:
/// `<num_days>days/<region><num_bins>_[<waste_type>|all]/[<data_distribution>/|'']<run_name>/log_*<num_instances>N.json`
pub fn extract_sim_hierarchy_metadata(file_path: &Path, root_path: Option<&Path>) -> DiscoveredSimulationLog {
    let filename = file_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or_default();
    let is_realtime = filename.contains("realtime") || filename.ends_with(".jsonl");
    let policy = extract_policy_name_from_filename(filename);
    let num_instances = extract_instance_count_from_filename(filename);
    let file_size = fs::metadata(file_path).map(|m| m.len()).unwrap_or(0);

    let relative_path = if let Some(root) = root_path {
        file_path
            .strip_prefix(root)
            .ok()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| file_path.to_string_lossy().to_string())
    } else {
        file_path.to_string_lossy().to_string()
    };

    let ancestors: Vec<String> = file_path
        .parent()
        .map(|p| {
            p.components()
                .filter_map(|c| c.as_os_str().to_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let mut days: Option<i32> = None;
    let mut region: Option<String> = None;
    let mut num_bins: Option<i32> = None;
    let mut waste_type: Option<String> = None;
    let mut region_idx: Option<usize> = None;

    for (i, part) in ancestors.iter().enumerate() {
        if part.ends_with("days") {
            let num_str = &part[..part.len() - 4];
            if let Ok(d) = num_str.parse::<i32>() {
                days = Some(d);
            }
        }
        if let Some(idx_under) = part.find('_') {
            let prefix = &part[..idx_under];
            let waste = &part[idx_under + 1..];
            if let Some(num_start) = prefix.find(|c: char| c.is_ascii_digit()) {
                let reg = &prefix[..num_start];
                let bins_str = &prefix[num_start..];
                if let Ok(b) = bins_str.parse::<i32>() {
                    region = Some(reg.to_string());
                    num_bins = Some(b);
                    waste_type = Some(waste.to_string());
                    region_idx = Some(i);
                }
            }
        }
    }

    let mut distribution: Option<String> = None;
    let mut run_name: Option<String> = None;

    if let Some(r_idx) = region_idx {
        if r_idx + 1 < ancestors.len() {
            let sub = &ancestors[r_idx + 1..];
            if sub.len() >= 2 {
                distribution = Some(sub[0].clone());
                run_name = Some(sub[1..].join("/"));
            } else if sub.len() == 1 {
                if is_known_distribution(&sub[0]) {
                    distribution = Some(sub[0].clone());
                } else {
                    run_name = Some(sub[0].clone());
                }
            }
        }
    } else if let Some(parent) = file_path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()) {
        run_name = Some(parent.to_string());
    }

    DiscoveredSimulationLog {
        path: file_path.to_string_lossy().to_string(),
        relative_path,
        days,
        region,
        num_bins,
        waste_type,
        distribution,
        run_name,
        policy,
        num_instances,
        file_size,
        is_realtime,
    }
}

/// Recursively collect all simulation log files under a directory.
pub fn find_simulation_logs_in_dir(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut logs = Vec::new();
    let mut stack = vec![dir.to_path_buf()];

    while let Some(current_dir) = stack.pop() {
        let entries = match fs::read_dir(&current_dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Avoid recursing into hidden, venv, or target directories
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with('.') && name_str != "node_modules" && name_str != "target" && name_str != "__pycache__" {
                    stack.push(path);
                }
            } else if path.is_file() && is_simulation_log_file(&path) {
                logs.push(path);
            }
        }
    }

    logs.sort();
    Ok(logs)
}

/// Scan a simulation directory tree and construct a catalog grouped into runs.
pub fn scan_simulation_directory_tree(dir: &Path) -> Result<SimulationDirectoryCatalog, String> {
    let files = find_simulation_logs_in_dir(dir)?;
    let mut runs_map: BTreeMap<String, DiscoveredSimulationRun> = BTreeMap::new();
    let mut total_logs = 0;

    for file in files {
        let meta = extract_sim_hierarchy_metadata(&file, Some(dir));
        total_logs += 1;

        let run_id = format!(
            "{}|{}|{}|{}|{}|{}",
            meta.days.map(|d| format!("{d}d")).unwrap_or_default(),
            meta.region.as_deref().unwrap_or_default(),
            meta.num_bins.map(|b| b.to_string()).unwrap_or_default(),
            meta.waste_type.as_deref().unwrap_or_default(),
            meta.distribution.as_deref().unwrap_or_default(),
            meta.run_name.as_deref().unwrap_or_default()
        );

        let entry = runs_map.entry(run_id).or_insert_with(|| {
            let mut label_parts = Vec::new();
            if let Some(d) = meta.days {
                label_parts.push(format!("{d}d"));
            }
            if let (Some(reg), Some(bins), Some(waste)) = (&meta.region, meta.num_bins, &meta.waste_type) {
                label_parts.push(format!("{reg}{bins}_{waste}"));
            }
            if let Some(dist) = &meta.distribution {
                if !dist.is_empty() {
                    label_parts.push(dist.clone());
                }
            }
            if let Some(run) = &meta.run_name {
                if !run.is_empty() {
                    label_parts.push(run.clone());
                }
            }
            let label = if label_parts.is_empty() {
                "Run".to_string()
            } else {
                label_parts.join(" · ")
            };

            DiscoveredSimulationRun {
                id: label.clone(),
                label,
                days: meta.days,
                region: meta.region.clone(),
                num_bins: meta.num_bins,
                waste_type: meta.waste_type.clone(),
                distribution: meta.distribution.clone(),
                run_name: meta.run_name.clone(),
                logs: Vec::new(),
            }
        });

        entry.logs.push(meta);
    }

    let mut runs: Vec<DiscoveredSimulationRun> = runs_map.into_values().collect();
    for run in &mut runs {
        let n_pol = run.logs.len();
        run.label = format!("{} ({} {})", run.label, n_pol, if n_pol == 1 { "policy" } else { "policies" });
    }

    Ok(SimulationDirectoryCatalog {
        root_path: dir.to_string_lossy().to_string(),
        total_logs,
        runs,
    })
}

/// Parse a simulation JSON log file (`log_*<N>N.json`) with top-level `daily` dictionary into `Vec<DayLogEntry>`.
pub fn parse_simulation_json_log(content: &str, file_path: Option<&Path>) -> Result<Vec<DayLogEntry>, String> {
    let json: serde_json::Value =
        serde_json::from_str(content).map_err(|e| format!("Failed to parse simulation JSON: {e}"))?;

    let daily_obj = match json.get("daily").and_then(|v| v.as_object()) {
        Some(d) => d,
        None => return Ok(vec![]),
    };

    let policy_name = file_path
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map(extract_policy_name_from_filename)
        .unwrap_or_else(|| "simulation_policy".to_string());

    let mut entries = Vec::new();

    for (sample_key, sample_val) in daily_obj {
        let sample_id: i32 = sample_key.parse().unwrap_or(0);
        let sample_map = match sample_val.as_object() {
            Some(m) => m,
            None => continue,
        };

        // Determine total days from any non-empty array
        let n_days = sample_map
            .values()
            .filter_map(|v| v.as_array())
            .map(|a| a.len())
            .max()
            .unwrap_or(0);

        for day_idx in 0..n_days {
            let mut day_data = serde_json::Map::new();
            let mut day_num: i32 = (day_idx as i32) + 1;

            for (k, v) in sample_map {
                if let Some(arr) = v.as_array() {
                    if let Some(item) = arr.get(day_idx) {
                        if k == "day" {
                            if let Some(d) = item.as_i64() {
                                day_num = d as i32;
                            }
                        } else if k == "mandatory_nodes" {
                            day_data.insert("mandatory".to_string(), item.clone());
                        } else if k == "tour" {
                            day_data.insert("tour".to_string(), item.clone());
                            if let Some(tour_nodes) = item.as_array() {
                                let tour_indices: Vec<i64> = tour_nodes
                                    .iter()
                                    .filter_map(|n| n.as_i64())
                                    .filter(|&n| n > 0)
                                    .map(|n| n - 1)
                                    .collect();
                                day_data.insert(
                                    "tour_indices".to_string(),
                                    serde_json::to_value(tour_indices).unwrap(),
                                );
                            }
                        } else {
                            day_data.insert(k.clone(), item.clone());
                        }
                    }
                }
            }

            entries.push(DayLogEntry {
                policy: policy_name.clone(),
                sample_id,
                day: day_num,
                data: serde_json::Value::Object(day_data),
            });
        }
    }

    entries.sort_by(|a, b| {
        a.policy
            .cmp(&b.policy)
            .then(a.sample_id.cmp(&b.sample_id))
            .then(a.day.cmp(&b.day))
    });

    Ok(entries)
}

/// Load a single simulation log from a file (supporting both JSONL and standard JSON).
pub fn load_single_simulation_log(path: &Path) -> Result<Vec<DayLogEntry>, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let path_str = path.to_string_lossy().to_lowercase();

    if path_str.ends_with(".json") {
        let json_entries = parse_simulation_json_log(&content, Some(path))?;
        if !json_entries.is_empty() {
            return Ok(json_entries);
        }
    }

    // Try parsing line-by-line (JSONL with GUI_DAY_LOG_START:)
    let line_entries: Vec<DayLogEntry> = content
        .lines()
        .filter_map(parse_day_log_line)
        .collect();

    if !line_entries.is_empty() {
        return Ok(line_entries);
    }

    // Fallback: attempt JSON parsing even if filename was not .json
    parse_simulation_json_log(&content, Some(path))
}

/// Load simulation failure entries from a single log file (supporting both JSONL and JSON failure_analysis).
pub fn load_single_sim_failure_log(path: &Path) -> Result<Vec<SimFailureEntry>, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let line_entries: Vec<SimFailureEntry> = content
        .lines()
        .filter_map(parse_sim_failure_line)
        .collect();

    if !line_entries.is_empty() {
        return Ok(line_entries);
    }

    // Fallback: check if JSON contains daily failure_analysis
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
        if let Some(daily_obj) = json.get("daily").and_then(|v| v.as_object()) {
            let policy_name = extract_policy_name_from_filename(
                path.file_name().and_then(|f| f.to_str()).unwrap_or_default(),
            );
            let mut failures = Vec::new();
            for (sample_key, sample_val) in daily_obj {
                let sample_id: i32 = sample_key.parse().unwrap_or(0);
                if let Some(sample_map) = sample_val.as_object() {
                    if let Some(fail_arr) = sample_map.get("failure_analysis").and_then(|v| v.as_array()) {
                        let day_arr = sample_map.get("day").and_then(|v| v.as_array());
                        for (idx, item) in fail_arr.iter().enumerate() {
                            if item.is_object() && item.get("has_failure").and_then(|v| v.as_bool()).unwrap_or(false) {
                                let day = day_arr
                                    .and_then(|a| a.get(idx))
                                    .and_then(|v| v.as_i64())
                                    .map(|d| d as i32)
                                    .unwrap_or((idx as i32) + 1);
                                failures.push(SimFailureEntry {
                                    policy: policy_name.clone(),
                                    sample_id,
                                    day,
                                    data: item.clone(),
                                });
                            }
                        }
                    }
                }
            }
            if !failures.is_empty() {
                return Ok(failures);
            }
        }
    }

    Ok(vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_policy_name_from_filename() {
        assert_eq!(
            extract_policy_name_from_filename("log_last_minute_cf90_bpc_custom_ftsp_1N.json"),
            "last_minute_cf90_bpc_custom_ftsp"
        );
        assert_eq!(
            extract_policy_name_from_filename("log_alns_custom_10N.json"),
            "alns_custom"
        );
        assert_eq!(
            extract_policy_name_from_filename("log_greedy.json"),
            "greedy"
        );
        assert_eq!(
            extract_policy_name_from_filename("log_realtime_emp_1N.jsonl"),
            "realtime_emp"
        );
    }

    #[test]
    fn test_extract_sim_hierarchy_metadata() {
        let p = Path::new("outputs/simulations-cptp/90days/figueiradafoz350_plastic/gamma3/lm_ftsp/log_last_minute_cf90_bpc_custom_ftsp_1N.json");
        let meta = extract_sim_hierarchy_metadata(p, None);
        assert_eq!(meta.days, Some(90));
        assert_eq!(meta.region.as_deref(), Some("figueiradafoz"));
        assert_eq!(meta.num_bins, Some(350));
        assert_eq!(meta.waste_type.as_deref(), Some("plastic"));
        assert_eq!(meta.distribution.as_deref(), Some("gamma3"));
        assert_eq!(meta.run_name.as_deref(), Some("lm_ftsp"));
        assert_eq!(meta.policy, "last_minute_cf90_bpc_custom_ftsp");
        assert_eq!(meta.num_instances, Some(1));
    }

    #[test]
    fn test_extract_sim_hierarchy_metadata_no_distribution() {
        let p = Path::new("outputs/simulations/30days/riomaior100_all/run_greedy/log_greedy_1N.json");
        let meta = extract_sim_hierarchy_metadata(p, None);
        assert_eq!(meta.days, Some(30));
        assert_eq!(meta.region.as_deref(), Some("riomaior"));
        assert_eq!(meta.num_bins, Some(100));
        assert_eq!(meta.waste_type.as_deref(), Some("all"));
        assert_eq!(meta.distribution, None);
        assert_eq!(meta.run_name.as_deref(), Some("run_greedy"));
        assert_eq!(meta.policy, "greedy");
        assert_eq!(meta.num_instances, Some(1));
    }

    #[test]
    fn test_parse_simulation_json_log_mock() {
        let json_text = r#"{
            "samples": { "0": { "profit": 100.0 } },
            "daily": {
                "0": {
                    "day": [1, 2],
                    "overflows": [0.0, 1.0],
                    "kg": [50.0, 75.0],
                    "km": [10.0, 15.0],
                    "profit": [20.0, 30.0],
                    "tour": [[0, 1, 2], [0, 2]]
                }
            }
        }"#;

        let entries = parse_simulation_json_log(json_text, Some(Path::new("log_alns_1N.json"))).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].policy, "alns");
        assert_eq!(entries[0].sample_id, 0);
        assert_eq!(entries[0].day, 1);
        assert_eq!(entries[0].data["kg"], 50.0);
        assert_eq!(entries[0].data["tour_indices"], serde_json::json!([0, 1]));

        assert_eq!(entries[1].day, 2);
        assert_eq!(entries[1].data["kg"], 75.0);
        assert_eq!(entries[1].data["tour_indices"], serde_json::json!([1]));
    }
}
