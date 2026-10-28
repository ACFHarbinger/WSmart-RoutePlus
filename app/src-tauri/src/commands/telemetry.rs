/// Policy telemetry SQLite queries (§A.3 Option C).
use serde::{Deserialize, Serialize};
use std::process::Command;

use super::process::resolve_python;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PolicyTelemetryTrendRow {
    pub id: i32,
    pub log_path: String,
    pub run_label: Option<String>,
    pub run_created_at: String,
    pub policy: String,
    pub sample_idx: i32,
    pub day: i32,
    pub policy_type: String,
    pub step_count: i32,
    pub final_metric: Option<f64>,
    pub metric_name: Option<String>,
    pub emitted_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PolicyTelemetryTrends {
    pub db_path: String,
    pub rows: Vec<PolicyTelemetryTrendRow>,
    pub policy_types: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PolicyTrajectorySeries {
    pub id: i32,
    pub label: String,
    pub run_label: Option<String>,
    pub policy: String,
    pub day: i32,
    pub policy_type: String,
    pub metric_name: String,
    pub x: Vec<i32>,
    pub y: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PolicyTrajectoryTrends {
    pub db_path: String,
    pub series: Vec<PolicyTrajectorySeries>,
}

const QUERY_TRENDS_SCRIPT: &str = r#"
import json, sys
from logic.src.tracking.logging.modules.policy_telemetry_db import query_policy_telemetry_trends
policy_type = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
run_label = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 500
print(json.dumps(query_policy_telemetry_trends(
    policy_type=policy_type or None,
    run_label=run_label or None,
    limit=limit,
)))
"#;

const QUERY_TRAJECTORIES_SCRIPT: &str = r#"
import json, sys
from logic.src.tracking.logging.modules.policy_telemetry_db import query_policy_trajectory_series
policy = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
policy_type = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
run_label = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
limit = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 12
print(json.dumps(query_policy_trajectory_series(
    policy=policy or None,
    policy_type=policy_type or None,
    run_label=run_label or None,
    limit=limit,
)))
"#;

fn run_python_json(
    python: &str,
    working_dir: &str,
    script: &str,
    args: &[&str],
) -> Result<serde_json::Value, String> {
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .args(args)
        .current_dir(working_dir)
        .output()
        .map_err(|e| format!("Failed to run Python: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

    if !output.status.success() && stdout.is_empty() {
        return Err(if stderr.is_empty() {
            format!("Python exited with code {:?}", output.status.code())
        } else {
            stderr
        });
    }

    serde_json::from_str(&stdout).map_err(|e| format!("Invalid JSON from Python: {e}\n{stdout}"))
}

/// Load cross-run policy telemetry rows from ``assets/telemetry.db``.
#[tauri::command]
pub fn load_policy_telemetry_trends(
    project_root: String,
    python_executable: Option<String>,
    policy_type: Option<String>,
    run_label: Option<String>,
    limit: Option<i32>,
) -> Result<PolicyTelemetryTrends, String> {
    let python = resolve_python(&project_root, python_executable);
    let type_arg = policy_type.unwrap_or_default();
    let run_arg = run_label.unwrap_or_default();
    let limit_arg = limit.unwrap_or(500).to_string();
    let value = run_python_json(
        &python,
        &project_root,
        QUERY_TRENDS_SCRIPT,
        &[&type_arg, &run_arg, &limit_arg],
    )?;
    serde_json::from_value(value).map_err(|e| e.to_string())
}

/// Load cross-run improvement trajectories from persisted ring-buffer JSON.
#[tauri::command]
pub fn load_policy_trajectory_trends(
    project_root: String,
    python_executable: Option<String>,
    policy: Option<String>,
    policy_type: Option<String>,
    run_label: Option<String>,
    limit: Option<i32>,
) -> Result<PolicyTrajectoryTrends, String> {
    let python = resolve_python(&project_root, python_executable);
    let policy_arg = policy.unwrap_or_default();
    let type_arg = policy_type.unwrap_or_default();
    let run_arg = run_label.unwrap_or_default();
    let limit_arg = limit.unwrap_or(12).to_string();
    let value = run_python_json(
        &python,
        &project_root,
        QUERY_TRAJECTORIES_SCRIPT,
        &[&policy_arg, &type_arg, &run_arg, &limit_arg],
    )?;
    serde_json::from_value(value).map_err(|e| e.to_string())
}
