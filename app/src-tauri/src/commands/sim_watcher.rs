/// Real-time simulation log watcher.
///
/// Polls the JSONL simulation output file for new `GUI_DAY_LOG_START:` lines
/// and emits a `sim:day_update` Tauri event on each new entry. This replaces
/// the Streamlit auto-refresh polling loop with an event-driven design that
/// delivers new simulation day data in < 200 ms without re-running any Python.
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

/// Mirrors the Python `DayLogEntry` dataclass from `logic/src/ui/services/log_parser.py`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DayLogEntry {
    pub policy: String,
    pub sample_id: i32,
    pub day: i32,
    pub data: serde_json::Value,
}

/// Policy iteration telemetry from ``PolicyVizMixin`` (§A.3).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PolicyVizEntry {
    pub policy: String,
    pub sample_id: i32,
    pub day: i32,
    pub policy_type: String,
    pub data: serde_json::Value,
}

/// Simulation failure root-cause summary from ``FailureAnalyzer`` (§A.6).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SimFailureEntry {
    pub policy: String,
    pub sample_id: i32,
    pub day: i32,
    pub data: serde_json::Value,
}

pub fn parse_day_log_line(line: &str) -> Option<DayLogEntry> {
    let line = line.trim();
    if !line.starts_with("GUI_DAY_LOG_START:") {
        return None;
    }
    let content = &line["GUI_DAY_LOG_START:".len()..];
    // Split into exactly 4 parts: policy, sample_id, day, json_payload
    let parts: Vec<&str> = content.splitn(4, ',').collect();
    if parts.len() < 4 {
        return None;
    }
    let policy = parts[0].trim().to_string();
    let sample_id: i32 = parts[1].trim().parse().ok()?;
    let day: i32 = parts[2].trim().parse().ok()?;
    let data: serde_json::Value = serde_json::from_str(parts[3].trim()).ok()?;
    Some(DayLogEntry { policy, sample_id, day, data })
}

pub fn parse_policy_viz_line(line: &str) -> Option<PolicyVizEntry> {
    let line = line.trim();
    if !line.starts_with("POLICY_VIZ_START:") {
        return None;
    }
    let content = &line["POLICY_VIZ_START:".len()..];
    // policy, sample_id, day, policy_type, json_payload
    let parts: Vec<&str> = content.splitn(5, ',').collect();
    if parts.len() < 5 {
        return None;
    }
    let policy = parts[0].trim().to_string();
    let sample_id: i32 = parts[1].trim().parse().ok()?;
    let day: i32 = parts[2].trim().parse().ok()?;
    let policy_type = parts[3].trim().to_string();
    let data: serde_json::Value = serde_json::from_str(parts[4].trim()).ok()?;
    Some(PolicyVizEntry {
        policy,
        sample_id,
        day,
        policy_type,
        data,
    })
}

pub fn parse_sim_failure_line(line: &str) -> Option<SimFailureEntry> {
    let line = line.trim();
    if !line.starts_with("SIM_FAILURE_START:") {
        return None;
    }
    let content = &line["SIM_FAILURE_START:".len()..];
    // policy, sample_id, day, json_payload
    let parts: Vec<&str> = content.splitn(4, ',').collect();
    if parts.len() < 4 {
        return None;
    }
    let policy = parts[0].trim().to_string();
    let sample_id: i32 = parts[1].trim().parse().ok()?;
    let day: i32 = parts[2].trim().parse().ok()?;
    let data: serde_json::Value = serde_json::from_str(parts[3].trim()).ok()?;
    Some(SimFailureEntry {
        policy,
        sample_id,
        day,
        data,
    })
}

/// Incrementally reads newly appended complete lines (`\n`-terminated) from `path`
/// starting at `*byte_offset`. If the file was truncated below `*byte_offset`,
/// resets `*byte_offset` to 0 before reading.
pub fn read_new_log_lines(path: &Path, byte_offset: &mut u64) -> std::io::Result<Vec<String>> {
    let mut file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    if file_len < *byte_offset {
        *byte_offset = 0;
    }
    if file_len == *byte_offset {
        return Ok(Vec::new());
    }
    file.seek(SeekFrom::Start(*byte_offset))?;
    let mut reader = BufReader::new(file);
    let mut new_lines = Vec::new();
    let mut line_buf = String::new();
    loop {
        line_buf.clear();
        let bytes_read = reader.read_line(&mut line_buf)?;
        if bytes_read == 0 {
            break;
        }
        // Only consume complete lines ending in '\n' so in-flight partial writes wait for next poll
        if line_buf.ends_with('\n') {
            *byte_offset += bytes_read as u64;
            new_lines.push(line_buf.trim_end_matches(&['\r', '\n'][..]).to_string());
        } else {
            break;
        }
    }
    Ok(new_lines)
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SimBatchUpdate {
    pub day_entries: Vec<DayLogEntry>,
    pub viz_entries: Vec<PolicyVizEntry>,
    pub failure_entries: Vec<SimFailureEntry>,
}

/// Global watcher cancellation flag (path → stop signal).
static WATCHER_STOP: std::sync::OnceLock<Arc<Mutex<Option<String>>>> = std::sync::OnceLock::new();

#[tauri::command]
pub async fn start_sim_watcher(path: String, app: AppHandle) -> Result<(), String> {
    let stop_cell = WATCHER_STOP.get_or_init(|| Arc::new(Mutex::new(None)));
    {
        let mut guard = stop_cell.lock().unwrap();
        *guard = None; // clear any previous stop request
    }
    let stop = Arc::clone(stop_cell);
    let watch_path = path.clone();

    tokio::spawn(async move {
        let mut byte_offset: u64 = 0;

        loop {
            // Check for stop signal
            if stop.lock().map(|g| g.is_some()).unwrap_or(false) {
                break;
            }

            let p = std::path::PathBuf::from(&watch_path);
            let prev_offset = byte_offset;
            if let Ok((lines, next_offset)) = tokio::task::spawn_blocking(move || {
                let mut offset = prev_offset;
                let res = read_new_log_lines(&p, &mut offset);
                res.map(|l| (l, offset))
            })
            .await
            .unwrap_or_else(|e| Err(std::io::Error::other(e.to_string())))
            {
                byte_offset = next_offset;
                if !lines.is_empty() {
                    let mut batch = SimBatchUpdate::default();
                    for line in &lines {
                        if let Some(entry) = parse_day_log_line(line) {
                            batch.day_entries.push(entry);
                        } else if let Some(viz) = parse_policy_viz_line(line) {
                            batch.viz_entries.push(viz);
                        } else if let Some(failure) = parse_sim_failure_line(line) {
                            batch.failure_entries.push(failure);
                        }
                    }
                    if !batch.day_entries.is_empty()
                        || !batch.viz_entries.is_empty()
                        || !batch.failure_entries.is_empty()
                    {
                        let _ = app.emit("sim:batch_update", &batch);
                    }
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }
    });

    Ok(())
}

#[tauri::command]
pub fn stop_sim_watcher() -> Result<(), String> {
    if let Some(stop) = WATCHER_STOP.get() {
        let mut guard = stop.lock().unwrap();
        *guard = Some("stop".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        parse_day_log_line, parse_policy_viz_line, parse_sim_failure_line, read_new_log_lines,
    };
    use std::io::Write;

    #[test]
    fn incremental_reader_tracks_byte_offset_across_appends() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join(format!("wsr_watcher_test_{}.jsonl", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).expect("create temp");
            writeln!(f, "GUI_DAY_LOG_START:hgs,0,1,{{\"profit\":100}}").unwrap();
        }
        let mut offset = 0u64;
        let first = read_new_log_lines(&path, &mut offset).expect("read 1");
        assert_eq!(first.len(), 1);
        assert!(offset > 0);

        // Append a second complete line and a partial third line (no trailing newline)
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
            write!(f, "GUI_DAY_LOG_START:alns,0,1,{{\"profit\":95}}\nPARTIAL_LINE").unwrap();
        }
        let second = read_new_log_lines(&path, &mut offset).expect("read 2");
        assert_eq!(second.len(), 1);
        assert!(second[0].contains("alns"));

        // Finish the partial line
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
            writeln!(f, "_COMPLETED").unwrap();
        }
        let third = read_new_log_lines(&path, &mut offset).expect("read 3");
        assert_eq!(third, vec!["PARTIAL_LINE_COMPLETED".to_string()]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn parse_policy_viz_line_valid() {
        let line = r#"POLICY_VIZ_START:ALNS + Ftsp,0,2,alns,{"iteration":[0,1],"best_cost":[10.0,9.0]}"#;
        let entry = parse_policy_viz_line(line).expect("parse");
        assert_eq!(entry.policy, "ALNS + Ftsp");
        assert_eq!(entry.sample_id, 0);
        assert_eq!(entry.day, 2);
        assert_eq!(entry.policy_type, "alns");
        assert_eq!(entry.data["best_cost"][0], 10.0);
    }

    #[test]
    fn parse_policy_viz_line_ignores_day_log() {
        let line = r#"GUI_DAY_LOG_START:greedy,0,1,{"profit":10}"#;
        assert!(parse_policy_viz_line(line).is_none());
        assert!(parse_day_log_line(line).is_some());
    }

    #[test]
    fn parse_sim_failure_line_valid() {
        let line = r#"SIM_FAILURE_START:greedy,0,2,{"has_failure":true,"severity":"warning","root_causes":["overflow_event"]}"#;
        let entry = parse_sim_failure_line(line).expect("parse");
        assert_eq!(entry.policy, "greedy");
        assert_eq!(entry.sample_id, 0);
        assert_eq!(entry.day, 2);
        assert_eq!(entry.data["severity"], "warning");
    }
}
