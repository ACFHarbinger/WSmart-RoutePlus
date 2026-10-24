use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::commands::process::resolve_python;

/// Used by ConfigEditor and OutputBrowser to display file contents without a CSV parser.
#[tauri::command]
pub fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| format!("{e}: {path}"))
}

/// Minimal RFC 4648 base64 decoder (standard alphabet, `=` padding).
fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Result<u32, String> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a') as u32 + 26),
            b'0'..=b'9' => Ok((c - b'0') as u32 + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("invalid base64 byte {c}")),
        }
    }
    let clean: Vec<u8> = input
        .bytes()
        .filter(|b| !b" \t\r\n".contains(b))
        .collect();
    let stripped: &[u8] = {
        let mut end = clean.len();
        while end > 0 && clean[end - 1] == b'=' {
            end -= 1;
        }
        &clean[..end]
    };
    let mut out = Vec::with_capacity(stripped.len() * 3 / 4);
    for chunk in stripped.chunks(4) {
        let mut acc: u32 = 0;
        for (i, &c) in chunk.iter().enumerate() {
            acc |= val(c)? << (18 - 6 * i);
        }
        let n_bytes = match chunk.len() {
            4 => 3,
            3 => 2,
            2 => 1,
            _ => return Err("truncated base64 quantum".into()),
        };
        for i in 0..n_bytes {
            out.push(((acc >> (16 - 8 * i)) & 0xFF) as u8);
        }
    }
    Ok(out)
}

/// Write (overwrite) a binary file from a base64 payload — used by the native
/// Report Studio exporters (PNG figures, PPTX/DOCX/XLSX artefacts; §H).
#[tauri::command]
pub fn write_binary_file(path: String, base64: String) -> Result<(), String> {
    let bytes = base64_decode(&base64).map_err(|e| format!("base64: {e}"))?;
    if let Some(parent) = Path::new(&path).parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }
    fs::write(&path, bytes).map_err(|e| format!("{e}: {path}"))
}

/// Recursively list files under `root` whose name matches `prefix`/`suffix`
/// filters — used by the native Report Studio output-tree parser (§H.1).
#[tauri::command]
pub fn list_files_recursive(
    root: String,
    prefix: Option<String>,
    suffix: Option<String>,
) -> Result<Vec<String>, String> {
    let base = PathBuf::from(&root);
    if !base.is_dir() {
        return Ok(vec![]);
    }
    let mut out: Vec<String> = Vec::new();
    let mut stack = vec![base];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let pre_ok = prefix.as_deref().is_none_or(|p| name.starts_with(p));
                let suf_ok = suffix.as_deref().is_none_or(|s| name.ends_with(s));
                if pre_ok && suf_ok {
                    out.push(path.to_string_lossy().to_string());
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Write (overwrite) any text file — used by ConfigEditor to save changes to YAML configs.
#[tauri::command]
pub fn write_text_file(path: String, content: String) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }
    std::fs::write(&path, &content).map_err(|e| format!("{e}: {path}"))
}

/// List all files (non-recursive) inside a directory, returning name + extension + size.
#[derive(Debug, Serialize, Deserialize)]
pub struct DirEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub size_bytes: u64,
    pub extension: String,
}

#[tauri::command]
pub fn list_dir(path: String) -> Result<Vec<DirEntry>, String> {
    let base = Path::new(&path);
    if !base.exists() {
        return Ok(vec![]);
    }
    let mut entries: Vec<DirEntry> = std::fs::read_dir(base)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| {
            let p = e.path();
            let meta = std::fs::metadata(&p).ok();
            DirEntry {
                name: e.file_name().to_string_lossy().to_string(),
                path: p.to_string_lossy().to_string(),
                is_dir: p.is_dir(),
                size_bytes: meta.map(|m| m.len()).unwrap_or(0),
                extension: p
                    .extension()
                    .map(|x| x.to_string_lossy().to_string())
                    .unwrap_or_default(),
            }
        })
        .collect();
    entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
    Ok(entries)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DatasetPreviewStats {
    pub num_instances: usize,
    pub num_nodes: Option<usize>,
    pub demand_mean: Option<f64>,
    pub demand_std: Option<f64>,
    pub demand_histogram: Vec<f64>,
    pub distance_mean: Option<f64>,
    pub file_size_bytes: u64,
}

const PREVIEW_DATASET_SCRIPT: &str = r#"
import json, sys, pickle, os
path = sys.argv[1]
size = os.path.getsize(path)
stats = {
    "num_instances": 0, "num_nodes": None, "demand_mean": None,
    "demand_std": None, "demand_histogram": [], "distance_mean": None,
    "file_size_bytes": size,
}
try:
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)
except Exception:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(0)

instances = data if isinstance(data, list) else [data]
stats["num_instances"] = len(instances)

def _tensor_stats(t):
    import torch
    if hasattr(t, "detach"):
        flat = t.detach().float().flatten()
        if flat.numel() == 0:
            return None, None
        return float(flat.mean()), float(flat.std())
    return None, None

first = instances[0] if instances else None
if isinstance(first, dict):
    for k in ["demand", "demands", "capacity"]:
        if k in first:
            m, s = _tensor_stats(first[k])
            if m is not None:
                stats["demand_mean"] = m
                stats["demand_std"] = s
                break
    for k in ["loc", "coords", "nodes", "graph"]:
        if k in first and hasattr(first[k], "shape"):
            stats["num_nodes"] = first[k].shape[0] if len(first[k].shape) > 0 else None
            break

print(json.dumps(stats))
"#;

/// Fast Python preview for `.pt` / `.pkl` VRPLIB or generated datasets (§G.2).
#[tauri::command]
pub fn preview_dataset_stats(
    path: String,
    project_root: String,
    python_executable: Option<String>,
) -> Result<DatasetPreviewStats, String> {
    if !Path::new(&path).is_file() {
        return Err(format!("File not found: {path}"));
    }
    let python = resolve_python(&project_root, python_executable);
    let output = Command::new(&python)
        .arg("-c")
        .arg(PREVIEW_DATASET_SCRIPT)
        .arg(&path)
        .current_dir(&project_root)
        .output()
        .map_err(|e| format!("Failed to run preview script: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        return Err("Preview script produced no output".to_string());
    }

    let value: serde_json::Value =
        serde_json::from_str(&stdout).map_err(|e| format!("Invalid JSON: {e}"))?;
    if let Some(err) = value.get("error").and_then(|v| v.as_str()) {
        return Err(err.to_string());
    }
    serde_json::from_value(value).map_err(|e| e.to_string())
}
