/// Arrow IPC serialization for CSV and simulation logs (§G.0 Phase 0).
use std::fs::{self, File};

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Float64Builder, Int32Builder, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;
use serde::Serialize;

use crate::commands::sim_watcher::{parse_day_log_line, DayLogEntry};

#[derive(Debug, Serialize)]
pub struct ArrowIpcFile {
    pub path: String,
    pub row_count: usize,
    pub column_count: usize,
    pub rust_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct ArrowPipelineBenchmark {
    pub row_count: usize,
    pub column_count: usize,
    pub rust_ms: u64,
    pub ipc_path: String,
}

fn arrow_temp_path(label: &str) -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join("wsmart-studio-arrow");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let stamp = SystemTimeNow::millis();
    Ok(dir.join(format!("{label}_{stamp}.arrow")))
}

struct SystemTimeNow;
impl SystemTimeNow {
    fn millis() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    }
}

fn write_ipc_file(batch: &RecordBatch, path: &PathBuf) -> Result<(), String> {
    let file = File::create(path).map_err(|e| e.to_string())?;
    let mut writer = FileWriter::try_new(file, &batch.schema()).map_err(|e| e.to_string())?;
    writer.write(batch).map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())?;
    Ok(())
}

fn json_float(data: &serde_json::Value, key: &str) -> Option<f64> {
    data.get(key).and_then(|v| v.as_f64())
}

fn simulation_entries_to_batch(entries: &[DayLogEntry]) -> Result<RecordBatch, String> {
    let n = entries.len();
    let mut policy_b = StringBuilder::with_capacity(n, n * 16);
    let mut sample_b = Int32Builder::with_capacity(n);
    let mut day_b = Int32Builder::with_capacity(n);
    let mut profit_b = Float64Builder::with_capacity(n);
    let mut km_b = Float64Builder::with_capacity(n);
    let mut overflows_b = Float64Builder::with_capacity(n);
    let mut kg_b = Float64Builder::with_capacity(n);
    let mut kgkm_b = Float64Builder::with_capacity(n);
    let mut cost_b = Float64Builder::with_capacity(n);
    let mut ncol_b = Float64Builder::with_capacity(n);
    let mut kg_lost_b = Float64Builder::with_capacity(n);

    for e in entries {
        policy_b.append_value(&e.policy);
        sample_b.append_value(e.sample_id);
        day_b.append_value(e.day);
        push_opt_f64(&mut profit_b, json_float(&e.data, "profit"));
        push_opt_f64(&mut km_b, json_float(&e.data, "km"));
        push_opt_f64(&mut overflows_b, json_float(&e.data, "overflows"));
        push_opt_f64(&mut kg_b, json_float(&e.data, "kg"));
        push_opt_f64(&mut kgkm_b, json_float(&e.data, "kg/km"));
        push_opt_f64(&mut cost_b, json_float(&e.data, "cost"));
        push_opt_f64(&mut ncol_b, json_float(&e.data, "ncol"));
        push_opt_f64(&mut kg_lost_b, json_float(&e.data, "kg_lost"));
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("policy", DataType::Utf8, false),
        Field::new("sample_id", DataType::Int32, false),
        Field::new("day", DataType::Int32, false),
        Field::new("profit", DataType::Float64, true),
        Field::new("km", DataType::Float64, true),
        Field::new("overflows", DataType::Float64, true),
        Field::new("kg", DataType::Float64, true),
        Field::new("kg_per_km", DataType::Float64, true),
        Field::new("cost", DataType::Float64, true),
        Field::new("ncol", DataType::Float64, true),
        Field::new("kg_lost", DataType::Float64, true),
    ]));

    let columns: Vec<ArrayRef> = vec![
        Arc::new(policy_b.finish()),
        Arc::new(sample_b.finish()),
        Arc::new(day_b.finish()),
        Arc::new(profit_b.finish()),
        Arc::new(km_b.finish()),
        Arc::new(overflows_b.finish()),
        Arc::new(kg_b.finish()),
        Arc::new(kgkm_b.finish()),
        Arc::new(cost_b.finish()),
        Arc::new(ncol_b.finish()),
        Arc::new(kg_lost_b.finish()),
    ];

    RecordBatch::try_new(schema, columns).map_err(|e| e.to_string())
}

fn push_opt_f64(builder: &mut Float64Builder, value: Option<f64>) {
    if let Some(v) = value {
        builder.append_value(v);
    } else {
        builder.append_null();
    }
}

fn column_is_numeric(values: &[String]) -> bool {
    values
        .iter()
        .all(|v| v.trim().is_empty() || v.parse::<f64>().is_ok())
}

fn csv_to_batch(path: &str) -> Result<RecordBatch, String> {
    let mut reader = csv::Reader::from_path(path).map_err(|e| e.to_string())?;
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| e.to_string())?
        .iter()
        .map(|h| h.to_string())
        .collect();

    let mut rows: Vec<Vec<String>> = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| e.to_string())?;
        rows.push(record.iter().map(|f| f.to_string()).collect());
    }

    let row_count = rows.len();
    let col_count = headers.len();
    let mut col_values: Vec<Vec<String>> = vec![Vec::with_capacity(row_count); col_count];
    for row in &rows {
        for (i, val) in row.iter().enumerate() {
            if i < col_count {
                col_values[i].push(val.clone());
            }
        }
    }

    let col_is_numeric: Vec<bool> = col_values
        .iter()
        .map(|col| column_is_numeric(col))
        .collect();

    let fields: Vec<Field> = headers
        .iter()
        .enumerate()
        .map(|(i, name)| {
            if col_is_numeric[i] {
                Field::new(name, DataType::Float64, true)
            } else {
                Field::new(name, DataType::Utf8, true)
            }
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(col_count);
    for (i, col) in col_values.iter().enumerate() {
        if col_is_numeric[i] {
            let mut b = Float64Builder::with_capacity(row_count);
            for v in col {
                if v.trim().is_empty() {
                    b.append_null();
                } else {
                    b.append_value(v.parse::<f64>().unwrap_or(0.0));
                }
            }
            columns.push(Arc::new(b.finish()));
        } else {
            let mut b = StringBuilder::with_capacity(row_count, row_count * 8);
            for v in col {
                b.append_value(v);
            }
            columns.push(Arc::new(b.finish()));
        }
    }

    RecordBatch::try_new(schema, columns).map_err(|e| e.to_string())
}

/// Write an Arrow IPC sidecar for an on-disk CSV (§G.8).
pub fn write_csv_arrow_sidecar(csv_path: &str, arrow_path: &PathBuf) -> Result<usize, String> {
    let batch = csv_to_batch(csv_path)?;
    let row_count = batch.num_rows();
    if let Some(parent) = arrow_path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    write_ipc_file(&batch, arrow_path)?;
    Ok(row_count)
}

/// Write an Arrow IPC sidecar for a simulation JSONL log (§G.8).
pub fn write_simulation_log_arrow_sidecar(
    log_path: &str,
    arrow_path: &PathBuf,
) -> Result<usize, String> {
    let content = fs::read_to_string(log_path).map_err(|e| e.to_string())?;
    let entries: Vec<DayLogEntry> = content.lines().filter_map(parse_day_log_line).collect();
    if entries.is_empty() {
        return Err("No simulation log entries found".to_string());
    }
    let batch = simulation_entries_to_batch(&entries)?;
    let row_count = batch.num_rows();
    if let Some(parent) = arrow_path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    write_ipc_file(&batch, arrow_path)?;
    Ok(row_count)
}

pub fn export_batch(
    batch: &RecordBatch,
    label: &str,
    start: Instant,
) -> Result<ArrowIpcFile, String> {
    let path = arrow_temp_path(label)?;
    write_ipc_file(batch, &path)?;
    let rust_ms = start.elapsed().as_millis() as u64;
    Ok(ArrowIpcFile {
        path: path.to_string_lossy().to_string(),
        row_count: batch.num_rows(),
        column_count: batch.num_columns(),
        rust_ms,
    })
}

#[tauri::command]
pub fn csv_to_arrow_ipc(path: String) -> Result<ArrowIpcFile, String> {
    let start = Instant::now();
    let batch = csv_to_batch(&path)?;
    export_batch(&batch, "csv", start)
}

#[tauri::command]
pub fn simulation_log_to_arrow_ipc(path: String) -> Result<ArrowIpcFile, String> {
    let start = Instant::now();
    let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let entries: Vec<DayLogEntry> = content.lines().filter_map(parse_day_log_line).collect();
    if entries.is_empty() {
        return Err("No simulation log entries found".to_string());
    }
    let batch = simulation_entries_to_batch(&entries)?;
    export_batch(&batch, "sim", start)
}

#[tauri::command]
pub fn benchmark_arrow_pipeline(csv_path: String) -> Result<ArrowPipelineBenchmark, String> {
    let start = Instant::now();
    let batch = csv_to_batch(&csv_path)?;
    let path = arrow_temp_path("bench")?;
    write_ipc_file(&batch, &path)?;
    Ok(ArrowPipelineBenchmark {
        row_count: batch.num_rows(),
        column_count: batch.num_columns(),
        rust_ms: start.elapsed().as_millis() as u64,
        ipc_path: path.to_string_lossy().to_string(),
    })
}

#[tauri::command]
pub fn read_binary_file(path: String) -> Result<Vec<u8>, String> {
    fs::read(&path).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn path_exists(path: String) -> bool {
    std::path::Path::new(&path).is_file()
}
