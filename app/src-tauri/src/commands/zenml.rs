/// ZenML pipeline tracking commands — list runs and step metadata via Python subprocess.
use serde::{Deserialize, Serialize};
use std::process::Command;

use super::process::resolve_python;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ZenmlPipelineRun {
    pub id: String,
    pub pipeline: String,
    pub status: String,
    pub created: String,
    pub updated: String,
    pub stack: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ZenmlPipelineStep {
    pub name: String,
    pub status: String,
    pub created: String,
    pub updated: String,
    pub duration_seconds: Option<f64>,
}

const LIST_RUNS_SCRIPT: &str = r#"
import json, sys
try:
    from zenml.client import Client
except ImportError:
    print(json.dumps({"error": "zenml not installed"}))
    sys.exit(0)
try:
    client = Client()
    runs = client.list_pipeline_runs(sort_by="desc:created", size=50)
    out = []
    for r in runs:
        out.append({
            "id": str(r.id),
            "pipeline": r.pipeline.name if r.pipeline else "",
            "status": str(r.status),
            "created": str(r.created) if r.created else "",
            "updated": str(r.updated) if r.updated else "",
            "stack": r.stack.name if r.stack else "",
        })
    print(json.dumps(out))
except Exception as e:
    print(json.dumps({"error": str(e)}))
"#;

const LOAD_STEPS_SCRIPT: &str = r#"
import json, sys
try:
    from zenml.client import Client
except ImportError:
    print(json.dumps({"error": "zenml not installed"}))
    sys.exit(0)
run_id = sys.argv[1]
try:
    client = Client()
    run = client.get_pipeline_run(run_id)
    out = []
    for step_name, step in run.steps.items():
        duration_s = None
        if step.created and step.updated:
            try:
                duration_s = (step.updated - step.created).total_seconds()
            except Exception:
                pass
        out.append({
            "name": step_name,
            "status": str(step.status),
            "created": str(step.created) if step.created else "",
            "updated": str(step.updated) if step.updated else "",
            "duration_seconds": duration_s,
        })
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

/// List recent ZenML pipeline runs (requires `zenml` package in the project venv).
#[tauri::command]
pub fn list_zenml_pipeline_runs(
    project_root: String,
    python_executable: Option<String>,
) -> Result<Vec<ZenmlPipelineRun>, String> {
    let python = resolve_python(&project_root, python_executable);
    let value = run_python_json(&python, &project_root, LIST_RUNS_SCRIPT, &[])?;
    serde_json::from_value(value).map_err(|e| e.to_string())
}

/// Load step-level metadata for a ZenML pipeline run.
#[tauri::command]
pub fn load_zenml_run_steps(
    run_id: String,
    project_root: String,
    python_executable: Option<String>,
) -> Result<Vec<ZenmlPipelineStep>, String> {
    let python = resolve_python(&project_root, python_executable);
    let value = run_python_json(&python, &project_root, LOAD_STEPS_SCRIPT, &[&run_id])?;
    serde_json::from_value(value).map_err(|e| e.to_string())
}
