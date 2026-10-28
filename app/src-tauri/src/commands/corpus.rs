/// Walk simulation JSON and JSONL files into a Summary corpus (#176).
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Map, Number, Value};

use crate::commands::sim_watcher::{parse_day_log_line, DayLogEntry};

const MAX_BYTES: u64 = 32 * 1024 * 1024;
const SKIP_DIRS: &[&str] = &["node_modules", "target", ".git", ".venv", "dist"];
const POLICY_KEYS: &[&str] = &["policy", "policy_name", "full_policy"];
const CONTAINER_KEYS: &[&str] = &["days", "entries", "records", "log"];
const FILL_KEYS: &[&str] = &["fills", "fill_levels", "bin_fills", "bin_state_c"];

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CorpusBin {
    pub bin_id: String,
    pub fill: Option<f64>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CorpusDay {
    pub day: Option<Number>,
    pub policy: Option<String>,
    pub city: Option<String>,
    pub metrics: Map<String, Value>,
    pub kg_lost: Option<f64>,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    pub bins: Vec<CorpusBin>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CorpusFileRecord {
    pub path: String,
    pub partial: bool,
    pub days: Vec<CorpusDay>,
}

#[derive(Debug, Serialize, Clone)]
pub struct CorpusFailure {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct CorpusScanResult {
    pub roots: Vec<String>,
    pub files: Vec<CorpusFileRecord>,
    pub kept: u64,
    pub skipped: u64,
    pub failed: Vec<CorpusFailure>,
}

#[tauri::command]
pub async fn scan_simulation_corpus(roots: Vec<String>) -> Result<CorpusScanResult, String> {
    tokio::task::spawn_blocking(move || scan_roots(roots))
        .await
        .map_err(|err| err.to_string())
}

fn scan_roots(roots: Vec<String>) -> CorpusScanResult {
    let mut scanner = Scanner::default();
    for root in &roots {
        scanner.ingest(Path::new(root));
    }
    scanner.finish(roots)
}

#[derive(Default)]
struct Scanner {
    files: Vec<CorpusFileRecord>,
    kept: u64,
    skipped: u64,
    failed: Vec<CorpusFailure>,
    seen_files: HashSet<PathBuf>,
    seen_dirs: HashSet<PathBuf>,
}

impl Scanner {
    fn finish(mut self, roots: Vec<String>) -> CorpusScanResult {
        self.files.sort_by(|left, right| left.path.cmp(&right.path));
        CorpusScanResult {
            roots,
            files: self.files,
            kept: self.kept,
            skipped: self.skipped,
            failed: self.failed,
        }
    }

    fn ingest(&mut self, path: &Path) {
        if path.is_dir() {
            self.walk(path);
            return;
        }
        if path.is_file() {
            self.consider(path, true);
            return;
        }
        self.fail(path, "not found");
    }

    fn walk(&mut self, dir: &Path) {
        if !self.seen_dirs.insert(canonical(dir)) {
            return;
        }
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => {
                self.fail(dir, "unreadable");
                return;
            }
        };
        for entry in entries {
            let Ok(entry) = entry else {
                continue;
            };
            let path = entry.path();
            if path.is_dir() {
                if is_skipped_dir(&entry.file_name()) {
                    continue;
                }
                self.walk(&path);
                continue;
            }
            if path.is_file() {
                self.consider(&path, false);
            }
        }
    }

    fn consider(&mut self, path: &Path, partial: bool) {
        if !self.seen_files.insert(canonical(path)) {
            self.skipped += 1;
            return;
        }
        if !is_corpus_ext(path) {
            self.skipped += 1;
            return;
        }
        let meta = match fs::metadata(path) {
            Ok(meta) => meta,
            Err(err) => {
                self.fail(path, io_reason(&err));
                return;
            }
        };
        if meta.len() > MAX_BYTES {
            self.fail(path, "file too large");
            return;
        }
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(err) => {
                self.fail(path, io_reason(&err));
                return;
            }
        };
        match parse_document(&text) {
            Ok(days) if days.is_empty() => self.skipped += 1,
            Ok(days) => {
                self.kept += 1;
                self.files.push(CorpusFileRecord {
                    path: path.to_string_lossy().to_string(),
                    partial,
                    days,
                });
            }
            Err(()) => self.fail(path, "invalid json"),
        }
    }

    fn fail(&mut self, path: &Path, reason: &str) {
        self.failed.push(CorpusFailure {
            path: path.to_string_lossy().to_string(),
            reason: reason.to_string(),
        });
    }
}

fn canonical(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn is_skipped_dir(name: &std::ffi::OsStr) -> bool {
    SKIP_DIRS.iter().any(|skip| name == *skip)
}

fn is_corpus_ext(path: &Path) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => ext.eq_ignore_ascii_case("json") || ext.eq_ignore_ascii_case("jsonl"),
        None => false,
    }
}

fn io_reason(err: &io::Error) -> &'static str {
    match err.kind() {
        io::ErrorKind::NotFound => "not found",
        io::ErrorKind::InvalidData => "invalid utf-8",
        _ => "unreadable",
    }
}

fn parse_document(text: &str) -> Result<Vec<CorpusDay>, ()> {
    let text = text.trim_start_matches('\u{feff}');
    let logged = log_days(text);
    if !logged.is_empty() {
        return Ok(logged);
    }
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        return Ok(days_from_value(&value));
    }
    let mut days = Vec::new();
    let mut saw_json = false;
    let mut saw_bad = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(value) => {
                saw_json = true;
                if let Some(day) = as_day(&value) {
                    days.push(day);
                }
            }
            Err(_) => saw_bad = true,
        }
    }
    if !days.is_empty() || (saw_json && !saw_bad) {
        return Ok(days);
    }
    Err(())
}

fn log_days(text: &str) -> Vec<CorpusDay> {
    text.lines()
        .filter_map(parse_day_log_line)
        .map(|entry| {
            let DayLogEntry {
                policy,
                day,
                data,
                ..
            } = entry;
            day_from_payload(&data, Some(policy), Some(Number::from(day)))
        })
        .collect()
}

fn days_from_value(value: &Value) -> Vec<CorpusDay> {
    if let Some(items) = value.as_array() {
        return items.iter().filter_map(as_day).collect();
    }
    let Some(obj) = value.as_object() else {
        return Vec::new();
    };
    for key in CONTAINER_KEYS {
        if let Some(Value::Array(items)) = obj.get(*key) {
            return items.iter().filter_map(as_day).collect();
        }
    }
    as_day(value).into_iter().collect()
}

fn as_day(value: &Value) -> Option<CorpusDay> {
    if !is_day_object(value) {
        return None;
    }
    Some(day_from_payload(value, None, None))
}

fn is_day_object(value: &Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };
    if obj.get("day").and_then(as_number).is_some() {
        return true;
    }
    if text_field(obj, POLICY_KEYS).is_some() {
        return true;
    }
    ["profit", "kg", "km", "cost", "overflows", "kg/km", "kg_per_km"]
        .iter()
        .any(|key| obj.get(*key).and_then(as_number).is_some())
}

fn day_from_payload(payload: &Value, policy: Option<String>, day: Option<Number>) -> CorpusDay {
    let obj = payload.as_object();
    let policy = policy.or_else(|| obj.and_then(|item| text_field(item, POLICY_KEYS)));
    let day = day.or_else(|| obj.and_then(|item| item.get("day").and_then(as_number)));
    let city = obj.and_then(|item| text_field(item, &["city", "city_name"]));
    let metrics = obj.map(metrics_of).unwrap_or_default();
    let kg_lost = obj.and_then(|item| first_f64(item, &["kg_lost", "kgLost"]));
    let (lat, lon) = obj.map(resolve_lat_lon).unwrap_or((None, None));
    let bins = obj.map(bins_of).unwrap_or_default();
    CorpusDay {
        day,
        policy,
        city,
        metrics,
        kg_lost,
        lat,
        lon,
        bins,
    }
}

fn metrics_of(obj: &Map<String, Value>) -> Map<String, Value> {
    const FIELDS: &[(&str, &[&str])] = &[
        ("profit", &["profit"]),
        ("kg", &["kg"]),
        ("km", &["km"]),
        ("cost", &["cost"]),
        ("overflows", &["overflows"]),
        ("kg_per_km", &["kg_per_km", "kg/km"]),
    ];
    let mut metrics = Map::new();
    for (name, keys) in FIELDS {
        if let Some(number) = keys.iter().find_map(|key| obj.get(*key).and_then(as_number)) {
            metrics.insert((*name).to_string(), Value::Number(number));
        }
    }
    metrics
}

fn bins_of(obj: &Map<String, Value>) -> Vec<CorpusBin> {
    if let Some(Value::Array(items)) = obj.get("bins") {
        let parsed: Vec<CorpusBin> = items
            .iter()
            .filter_map(|item| item.as_object().and_then(bin_from_object))
            .collect();
        if !parsed.is_empty() {
            return parsed;
        }
    }
    let Some(Value::Array(coords)) = obj.get("all_bin_coords") else {
        return Vec::new();
    };
    let fills = parallel_fills(obj);
    coords
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            let item = item.as_object()?;
            let bin_id = item
                .get("id")
                .and_then(id_string)
                .unwrap_or_else(|| index.to_string());
            let fill = first_f64(item, &["fill", "fill_level"]).or_else(|| {
                fills
                    .as_ref()
                    .and_then(|rows| rows.get(index).copied().flatten())
            });
            Some(CorpusBin { bin_id, fill })
        })
        .collect()
}

fn bin_from_object(obj: &Map<String, Value>) -> Option<CorpusBin> {
    let bin_id = obj
        .get("id")
        .or_else(|| obj.get("bin_id"))
        .or_else(|| obj.get("binId"))
        .and_then(id_string)?;
    Some(CorpusBin {
        bin_id,
        fill: first_f64(obj, &["fill", "fill_level"]),
    })
}

fn parallel_fills(obj: &Map<String, Value>) -> Option<Vec<Option<f64>>> {
    for key in FILL_KEYS {
        if let Some(Value::Array(items)) = obj.get(*key) {
            if items.is_empty() {
                continue;
            }
            return Some(
                items
                    .iter()
                    .map(|item| as_number(item).and_then(|number| number.as_f64()))
                    .collect(),
            );
        }
    }
    None
}

fn resolve_lat_lon(obj: &Map<String, Value>) -> (Option<f64>, Option<f64>) {
    let lat = first_f64(obj, &["lat", "latitude"]);
    let lon = first_f64(obj, &["lng", "lon", "longitude"]);
    if lat.is_some() && lon.is_some() {
        return (lat, lon);
    }
    let mean = obj
        .get("all_bin_coords")
        .and_then(Value::as_array)
        .map(|items| mean_of_array(items))
        .filter(|pair| pair.0.is_some() && pair.1.is_some())
        .or_else(|| {
            obj.get("bins")
                .and_then(Value::as_array)
                .map(|items| mean_of_array(items))
        });
    match mean {
        Some((Some(mean_lat), Some(mean_lon))) => (Some(mean_lat), Some(mean_lon)),
        _ => (lat, lon),
    }
}

fn mean_of_array(items: &[Value]) -> (Option<f64>, Option<f64>) {
    let mut sum_lat = 0.0;
    let mut sum_lon = 0.0;
    let mut count = 0.0;
    for item in items {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let Some(lat) = first_f64(obj, &["lat", "latitude"]) else {
            continue;
        };
        let Some(lon) = first_f64(obj, &["lng", "lon", "longitude"]) else {
            continue;
        };
        sum_lat += lat;
        sum_lon += lon;
        count += 1.0;
    }
    if count == 0.0 {
        (None, None)
    } else {
        (Some(sum_lat / count), Some(sum_lon / count))
    }
}

fn text_field(obj: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(Value::String(text)) = obj.get(*key) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn first_f64(obj: &Map<String, Value>, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| obj.get(*key).and_then(as_number))
        .and_then(|number| number.as_f64())
}

fn as_number(value: &Value) -> Option<Number> {
    match value {
        Value::Number(number) => number
            .as_f64()
            .filter(|parsed| parsed.is_finite())
            .map(|_| number.clone()),
        Value::String(text) => {
            let trimmed = text.trim();
            if let Ok(integer) = trimmed.parse::<i64>() {
                Some(Number::from(integer))
            } else {
                trimmed
                    .parse::<f64>()
                    .ok()
                    .filter(|parsed| parsed.is_finite())
                    .and_then(Number::from_f64)
            }
        }
        _ => None,
    }
}

fn id_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.trim().is_empty() => Some(text.trim().to_string()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

#[cfg(test)]
#[path = "corpus_tests.rs"]
mod tests;
