//! Subprocess wrapper around the `python main.py cloud` CLI.

use serde::de::DeserializeOwned;
use std::path::PathBuf;
use tokio::process::Command;

use crate::commands::process::resolve_python;

/// Locate the project root containing `main.py`.
pub fn resolve_project_root(project_root: Option<&str>) -> PathBuf {
    if let Some(r) = project_root.filter(|s| !s.trim().is_empty()) {
        let p = PathBuf::from(r);
        if p.exists() {
            return p;
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("main.py").exists() {
            return cwd;
        }
        if let Some(parent) = cwd.parent() {
            if parent.join("main.py").exists() {
                return parent.to_path_buf();
            }
        }
    }

    PathBuf::from(".")
}

/// Run a cloud sub-command with `--json` and deserialize the result.
pub async fn run_cloud_cli<T: DeserializeOwned>(
    project_root: Option<&str>,
    root: Option<&str>,
    python_executable: Option<&str>,
    subcmd_args: &[String],
) -> Result<T, String> {
    let project_dir = resolve_project_root(project_root);
    let project_dir_str = project_dir.to_str().unwrap_or(".");
    let python = resolve_python(project_dir_str, python_executable.map(str::to_string));

    let root_arg = root.unwrap_or("outputs/cloud");
    let mut full_args = vec![
        "main.py".to_string(),
        "cloud".to_string(),
        "--root".to_string(),
        root_arg.to_string(),
        "--json".to_string(),
    ];
    full_args.extend_from_slice(subcmd_args);

    let mut cmd = Command::new(&python);
    cmd.args(&full_args)
        .current_dir(&project_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);

    // Ensure logic packages are found by Python interpreter
    if let Some(existing_pythonpath) = std::env::var_os("PYTHONPATH") {
        let mut paths = std::env::split_paths(&existing_pythonpath).collect::<Vec<_>>();
        if !paths.contains(&project_dir) {
            paths.insert(0, project_dir.clone());
        }
        if let Ok(new_pythonpath) = std::env::join_paths(paths) {
            cmd.env("PYTHONPATH", new_pythonpath);
        }
    } else {
        cmd.env("PYTHONPATH", project_dir_str);
    }

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to execute Python process '{python}': {e}"))?;

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let stderr_str = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        let err_body = stderr_str.trim();
        let fallback_body = stdout_str.trim();
        let message = if !err_body.is_empty() {
            err_body
        } else if !fallback_body.is_empty() {
            fallback_body
        } else {
            "unspecified error occurred in cloud CLI"
        };
        return Err(format!(
            "cloud command failed (exit code {:?}): {}",
            output.status.code(),
            message
        ));
    }

    let trimmed_stdout = stdout_str.trim();
    serde_json::from_str::<T>(trimmed_stdout).map_err(|e| {
        format!(
            "failed to parse JSON from cloud command output: {e}; stdout: '{trimmed_stdout}'; stderr: '{}'",
            stderr_str.trim()
        )
    })
}
