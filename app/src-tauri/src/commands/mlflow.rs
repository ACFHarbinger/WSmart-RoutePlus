/// MLflow experiment tracking commands — query runs and metric history via Python subprocess.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;

use super::process::resolve_python;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MlflowRun {
    pub run_id: String,
    pub run_name: String,
    pub experiment_id: String,
    pub status: String,
    pub start_time: Option<i64>,
    pub end_time: Option<i64>,
    pub artifact_uri: String,
    pub params: HashMap<String, String>,
    pub metrics: HashMap<String, f64>,
    pub tags: HashMap<String, String>,
}

const LIST_RUNS_SCRIPT: &str = r#"
import json, sys, os
try:
    import mlflow
except ImportError:
    print(json.dumps({"error": "mlflow not installed"}))
    sys.exit(0)

tracking_uri = sys.argv[1]
experiment_name = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else ""
project_root = sys.argv[3]

if not tracking_uri.startswith(("http://", "https://", "file://")):
    tracking_uri = os.path.join(project_root, tracking_uri)

mlflow.set_tracking_uri(tracking_uri)
kwargs = {"output_format": "list", "max_results": 200}
if experiment_name:
    kwargs["experiment_names"] = [experiment_name]
try:
    runs = mlflow.search_runs(**kwargs)
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(0)

out = []
for run in runs:
    out.append({
        "run_id": run.info.run_id,
        "run_name": run.info.run_name or "",
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "artifact_uri": run.info.artifact_uri or "",
        "params": dict(run.data.params),
        "metrics": {k: float(v) for k, v in run.data.metrics.items()},
        "tags": dict(run.data.tags),
    })
print(json.dumps(out))
"#;

const LIST_METRIC_KEYS_SCRIPT: &str = r#"
import json, sys, os
try:
    from mlflow.tracking import MlflowClient
except ImportError:
    print(json.dumps({"error": "mlflow not installed"}))
    sys.exit(0)

run_id = sys.argv[1]
tracking_uri = sys.argv[2]
project_root = sys.argv[3]

if not tracking_uri.startswith(("http://", "https://", "file://")):
    tracking_uri = os.path.join(project_root, tracking_uri)

try:
    client = MlflowClient(tracking_uri)
    run = client.get_run(run_id)
    keys = sorted(run.data.metrics.keys())
    print(json.dumps(keys))
except Exception as e:
    print(json.dumps({"error": str(e)}))
"#;

const LOAD_METRIC_HISTORY_SCRIPT: &str = r#"
import json, sys, os
try:
    from mlflow.tracking import MlflowClient
except ImportError:
    print(json.dumps({"error": "mlflow not installed"}))
    sys.exit(0)

run_ids = json.loads(sys.argv[1])
metric_key = sys.argv[2]
tracking_uri = sys.argv[3]
project_root = sys.argv[4]

if not tracking_uri.startswith(("http://", "https://", "file://")):
    tracking_uri = os.path.join(project_root, tracking_uri)

out = {}
try:
    client = MlflowClient(tracking_uri)
    for run_id in run_ids:
        history = client.get_metric_history(run_id, metric_key)
        out[run_id] = [{"step": m.step, "value": float(m.value)} for m in history]
    print(json.dumps(out))
except Exception as e:
    print(json.dumps({"error": str(e)}))
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

    let value: serde_json::Value = serde_json::from_str(&stdout)
        .map_err(|e| format!("Invalid JSON from Python: {e}\n{stdout}"))?;

    if let Some(err) = value.get("error").and_then(|v| v.as_str()) {
        return Err(err.to_string());
    }

    Ok(value)
}

/// List MLflow runs from a tracking URI (local path or remote server).
#[tauri::command]
pub fn list_mlflow_runs(
    tracking_uri: String,
    experiment_name: Option<String>,
    project_root: String,
    python_executable: Option<String>,
) -> Result<Vec<MlflowRun>, String> {
    let python = resolve_python(&project_root, python_executable);
    let exp = experiment_name.unwrap_or_default();
    let value = run_python_json(
        &python,
        &project_root,
        LIST_RUNS_SCRIPT,
        &[&tracking_uri, &exp, &project_root],
    )?;
    serde_json::from_value(value).map_err(|e| e.to_string())
}

/// List metric keys logged for a single MLflow run.
#[tauri::command]
pub fn list_mlflow_metric_keys(
    run_id: String,
    tracking_uri: String,
    project_root: String,
    python_executable: Option<String>,
) -> Result<Vec<String>, String> {
    let python = resolve_python(&project_root, python_executable);
    let value = run_python_json(
        &python,
        &project_root,
        LIST_METRIC_KEYS_SCRIPT,
        &[&run_id, &tracking_uri, &project_root],
    )?;
    serde_json::from_value(value).map_err(|e| e.to_string())
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MlflowMetricPoint {
    pub step: i64,
    pub value: f64,
}

/// Load step-indexed metric history for one or more MLflow runs (for overlay comparison).
#[tauri::command]
pub fn load_mlflow_metric_history(
    run_ids: Vec<String>,
    metric_key: String,
    tracking_uri: String,
    project_root: String,
    python_executable: Option<String>,
) -> Result<HashMap<String, Vec<MlflowMetricPoint>>, String> {
    let python = resolve_python(&project_root, python_executable);
    let run_ids_json =
        serde_json::to_string(&run_ids).map_err(|e| format!("Failed to encode run IDs: {e}"))?;
    let value = run_python_json(
        &python,
        &project_root,
        LOAD_METRIC_HISTORY_SCRIPT,
        &[&run_ids_json, &metric_key, &tracking_uri, &project_root],
    )?;
    serde_json::from_value(value).map_err(|e| e.to_string())
}
