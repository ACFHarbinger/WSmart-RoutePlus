//! Catalog and execution-trace bridge. Both commands run `python main.py` and
//! read the JSON file the CLI writes. The process stdout is only that path.
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use super::process::resolve_python;
use crate::commands::cloud::cli::resolve_project_root;

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

struct TempJson(PathBuf);

impl Drop for TempJson {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn next_temp_json(kind: &str) -> PathBuf {
    let n = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("wsmart-{kind}-{}-{n}.json", std::process::id()))
}

fn run_main_py_json(
    project_root: &str,
    mut args: Vec<String>,
) -> Result<serde_json::Value, String> {
    let dir = resolve_project_root(Some(project_root));
    let dir_str = dir
        .to_str()
        .ok_or_else(|| "project root is not valid UTF-8".to_string())?;
    let python = resolve_python(dir_str, None);
    let output_path = next_temp_json(args.get(1).map(String::as_str).unwrap_or("main"));
    let cleanup = TempJson(output_path.clone());

    args.push("--output".to_string());
    args.push(output_path.to_string_lossy().into_owned());

    let output = Command::new(&python)
        .args(&args)
        .current_dir(&dir)
        .output()
        .map_err(|err| format!("Failed to run '{python} main.py': {err}"))?;

    if !output.status.success() {
        return Err(command_failure(&output));
    }

    let text = std::fs::read_to_string(&cleanup.0).map_err(|err| {
        format!(
            "failed to read {}: {err}; {}",
            cleanup.0.display(),
            command_failure(&output)
        )
    })?;
    serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse JSON from {}: {err}", cleanup.0.display()))
}

fn command_failure(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let detail = if !stderr.trim().is_empty() {
        stderr.trim()
    } else {
        stdout.trim()
    };
    format!(
        "main.py exited with status {:?}: {detail}",
        output.status.code()
    )
}

fn push_opt(args: &mut Vec<String>, flag: &str, value: &Option<String>) {
    if let Some(value) = value {
        args.push(flag.to_string());
        args.push(value.clone());
    }
}

/// `python main.py list_policy_modules --output <temp.json>`.
#[tauri::command(rename_all = "snake_case")]
pub fn list_policy_modules(project_root: String) -> Result<serde_json::Value, String> {
    run_main_py_json(
        &project_root,
        vec!["main.py".to_string(), "list_policy_modules".to_string()],
    )
}

/// `python main.py trace_policy` with `--routing alns`. Optional flags are omitted when `None`.
#[tauri::command(rename_all = "snake_case")]
pub fn trace_policy(
    project_root: String,
    problem: String,
    graph_size: u32,
    seed: u32,
    acceptance: String,
    packing: Option<String>,
    route_improvement: Option<String>,
    max_iterations: u32,
) -> Result<serde_json::Value, String> {
    let mut args = vec![
        "main.py".to_string(),
        "trace_policy".to_string(),
        "--problem".to_string(),
        problem,
        "--graph-size".to_string(),
        graph_size.to_string(),
        "--seed".to_string(),
        seed.to_string(),
        "--acceptance".to_string(),
        acceptance,
        "--routing".to_string(),
        "alns".to_string(),
        "--max-iterations".to_string(),
        max_iterations.to_string(),
    ];
    push_opt(&mut args, "--packing", &packing);
    push_opt(&mut args, "--route-improvement", &route_improvement);
    run_main_py_json(&project_root, args)
}
