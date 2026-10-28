use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use zip::write::SimpleFileOptions;
use zip::{ZipArchive, ZipWriter};

use crate::commands::arrow;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WsrouteBundleFile {
    pub path: String,
    pub size_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WsrouteBundleInfo {
    pub path: String,
    pub version: Option<String>,
    pub created_at: Option<String>,
    pub arrow_sidecars: Option<u32>,
    pub files: Vec<WsrouteBundleFile>,
}

/// Inspect a `.wsroute` zip bundle produced by `logic/gen/export_for_studio.py`.
#[tauri::command]
pub fn inspect_wsroute_bundle(path: String) -> Result<WsrouteBundleInfo, String> {
    let file = File::open(&path).map_err(|e| format!("Failed to open bundle: {e}"))?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid zip bundle: {e}"))?;

    let mut files = Vec::new();
    let mut version = None;
    let mut created_at = None;
    let mut arrow_sidecars = None;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = entry.name().to_string();
        let size = entry.size();

        if name == "manifest.json" {
            let mut buf = String::new();
            if entry.read_to_string(&mut buf).is_ok() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&buf) {
                    version = v
                        .get("version")
                        .and_then(|x| x.as_str())
                        .map(str::to_string);
                    created_at = v
                        .get("created_at")
                        .and_then(|x| x.as_str())
                        .map(str::to_string);
                    arrow_sidecars = v
                        .get("arrow_sidecars")
                        .and_then(|x| x.as_u64())
                        .map(|n| n as u32);
                }
            }
        }

        files.push(WsrouteBundleFile {
            path: name,
            size_bytes: size,
        });
    }

    files.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(WsrouteBundleInfo {
        path,
        version,
        created_at,
        arrow_sidecars,
        files,
    })
}

const BUNDLE_EXTENSIONS: &[&str] = &[
    "arrow", "csv", "json", "jsonl", "yaml", "yml", "npz", "pkl", "pt", "parquet",
];

fn is_bundle_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| BUNDLE_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn collect_bundle_files(
    dir: &Path,
    base: &Path,
    out: &mut Vec<(String, PathBuf)>,
) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            collect_bundle_files(&path, base, out)?;
        } else if is_bundle_extension(&path) {
            let rel = path.strip_prefix(base).map_err(|e| e.to_string())?;
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            out.push((rel_str, path));
        }
    }
    Ok(())
}

fn iso_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

/// Package a run output directory into a `.wsroute` zip bundle (§G.8).
#[tauri::command]
pub fn create_wsroute_bundle(
    source_dir: String,
    output_path: String,
    include_arrow: Option<bool>,
) -> Result<WsrouteBundleInfo, String> {
    let source = Path::new(&source_dir);
    if !source.is_dir() {
        return Err(format!("Source directory not found: {source_dir}"));
    }

    let mut files: Vec<(String, PathBuf)> = Vec::new();
    collect_bundle_files(source, source, &mut files)?;
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut arrow_sidecars: Vec<(String, PathBuf)> = Vec::new();
    if include_arrow.unwrap_or(false) {
        let staging = std::env::temp_dir().join(format!(
            "wsroute_arrow_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&staging).map_err(|e| e.to_string())?;

        for (rel, abs) in &files {
            let ext = abs.extension().and_then(|e| e.to_str()).unwrap_or("");
            let arrow_rel = Path::new(rel)
                .with_extension("arrow")
                .to_string_lossy()
                .replace('\\', "/");
            if files.iter().any(|(r, _)| r == &arrow_rel) {
                continue;
            }
            let arrow_abs = staging.join(&arrow_rel);
            let path_str = abs.to_string_lossy();
            match ext {
                "csv" => arrow::write_csv_arrow_sidecar(&path_str, &arrow_abs)?,
                "jsonl" => arrow::write_simulation_log_arrow_sidecar(&path_str, &arrow_abs)?,
                _ => continue,
            };
            arrow_sidecars.push((arrow_rel, arrow_abs));
        }
    }

    let mut packaged: Vec<(String, PathBuf)> = files.clone();
    packaged.extend(arrow_sidecars.clone());
    packaged.sort_by(|a, b| a.0.cmp(&b.0));

    let rel_paths: Vec<String> = packaged.iter().map(|(r, _)| r.clone()).collect();
    let manifest = serde_json::json!({
        "version": "1",
        "created_at": iso_timestamp(),
        "source": source_dir,
        "file_count": rel_paths.len(),
        "arrow_sidecars": arrow_sidecars.len(),
        "files": rel_paths,
    });

    if let Some(parent) = Path::new(&output_path).parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let out_file = File::create(&output_path).map_err(|e| e.to_string())?;
    let mut zip = ZipWriter::new(out_file);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("manifest.json", options)
        .map_err(|e| e.to_string())?;
    zip.write_all(manifest.to_string().as_bytes())
        .map_err(|e| e.to_string())?;

    for (rel, abs) in &packaged {
        zip.start_file(rel, options).map_err(|e| e.to_string())?;
        let mut f = File::open(abs).map_err(|e| e.to_string())?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        zip.write_all(&buf).map_err(|e| e.to_string())?;
    }

    zip.finish().map_err(|e| e.to_string())?;
    if let Some((_, path)) = arrow_sidecars.first() {
        if let Some(parent) = path.parent() {
            let _ = fs::remove_dir_all(parent);
        }
    }
    inspect_wsroute_bundle(output_path)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WsrouteExtractResult {
    pub dest_dir: String,
    pub extracted_files: Vec<String>,
    pub log_path: Option<String>,
}

/// Extract a `.wsroute` bundle to a destination directory; returns first `.jsonl` path if found.
#[tauri::command]
pub fn extract_wsroute_bundle(
    path: String,
    dest_dir: String,
) -> Result<WsrouteExtractResult, String> {
    fs::create_dir_all(&dest_dir).map_err(|e| e.to_string())?;
    let dest = Path::new(&dest_dir);

    let file = File::open(&path).map_err(|e| format!("Failed to open bundle: {e}"))?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid zip bundle: {e}"))?;

    let mut extracted = Vec::new();
    let mut log_path: Option<String> = None;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = entry.name().to_string();
        if name.ends_with('/') {
            continue;
        }

        let out_path = dest.join(&name);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }

        let mut out_file = File::create(&out_path).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut out_file).map_err(|e| e.to_string())?;

        let out_str = out_path.to_string_lossy().to_string();
        extracted.push(out_str.clone());

        if log_path.is_none() && name.ends_with(".jsonl") {
            log_path = Some(out_str);
        }
    }

    extracted.sort();

    Ok(WsrouteExtractResult {
        dest_dir,
        extracted_files: extracted,
        log_path,
    })
}
