use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::commands::process::resolve_python;

const PARQUET_SCRIPT: &str = r#"
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
df.to_parquet(sys.argv[2], engine="pyarrow", index=False)
"#;

fn run_parquet_export(
    project_root: &str,
    csv_path: &str,
    output_path: &str,
) -> Result<String, String> {
    let python = resolve_python(project_root, None);
    if let Some(parent) = Path::new(output_path).parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let output = Command::new(&python)
        .args(["-c", PARQUET_SCRIPT, csv_path, output_path])
        .output()
        .map_err(|e| format!("Failed to run Python for Parquet export: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "Parquet export failed (is pyarrow installed?): {stderr}"
        ));
    }

    Ok(output_path.to_string())
}

/// Convert an on-disk CSV file to Parquet via pandas/pyarrow (§G.7).
#[tauri::command]
pub fn export_csv_to_parquet(
    project_root: String,
    csv_path: String,
    output_path: String,
) -> Result<String, String> {
    if !Path::new(&csv_path).is_file() {
        return Err(format!("CSV file not found: {csv_path}"));
    }
    run_parquet_export(&project_root, &csv_path, &output_path)
}

fn write_table_csv(
    headers: &[String],
    rows: &[HashMap<String, serde_json::Value>],
    path: &Path,
) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path).map_err(|e| e.to_string())?;
    writer.write_record(headers).map_err(|e| e.to_string())?;
    for row in rows {
        let record: Vec<String> = headers
            .iter()
            .map(|h| match row.get(h) {
                Some(serde_json::Value::Number(n)) => n.to_string(),
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Bool(b)) => b.to_string(),
                Some(serde_json::Value::Null) | None => String::new(),
                Some(other) => other.to_string(),
            })
            .collect();
        writer.write_record(&record).map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Export in-memory tabular data to Parquet (writes a temp CSV first) (§G.7).
#[tauri::command]
pub fn export_table_parquet(
    project_root: String,
    headers: Vec<String>,
    rows: Vec<HashMap<String, serde_json::Value>>,
    output_path: String,
) -> Result<String, String> {
    let temp_dir = std::env::temp_dir().join("wsmart-studio-parquet");
    fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
    let temp_csv = temp_dir.join(format!(
        "table_{}.csv",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    write_table_csv(&headers, &rows, &temp_csv)?;
    let result = run_parquet_export(&project_root, temp_csv.to_str().unwrap(), &output_path);
    let _ = fs::remove_file(&temp_csv);
    result
}
