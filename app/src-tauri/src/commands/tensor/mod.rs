/// TensorDict / NumPy archive inspection and slice streaming (§G.5.1).
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Float64Builder, Int32Builder, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};
use zip::ZipArchive;

use crate::commands::arrow::{export_batch, ArrowIpcFile};

pub mod npy;
pub mod td;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub const MMAP_THRESHOLD_BYTES: u64 = 64;
#[cfg(not(test))]
pub const MMAP_THRESHOLD_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NpzArrayInfo {
    pub key: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NpzArchiveInfo {
    pub path: String,
    pub arrays: Vec<NpzArrayInfo>,
    pub total_bytes: u64,
    pub used_memmap: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NpzVectorData {
    pub key: String,
    pub values: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TensorSlicePreview {
    pub key: String,
    pub full_shape: Vec<usize>,
    pub slice_shape: Vec<usize>,
    pub indices: Vec<usize>,
    pub values: Vec<Vec<f64>>,
    pub min: f64,
    pub max: f64,
    pub rust_ms: u64,
    pub used_memmap: bool,
    #[serde(default)]
    pub used_decompress_slice: bool,
}

#[tauri::command]
pub fn inspect_npz_archive(
    path: String,
    project_root: Option<String>,
    python_executable: Option<String>,
) -> Result<NpzArchiveInfo, String> {
    let p = Path::new(&path);
    if !p.exists() {
        return Err(format!("File not found: {path}"));
    }

    if td::is_td_path(p) {
        let root = project_root.filter(|s| !s.is_empty()).ok_or_else(|| {
            "Project root required to inspect TensorDict (.td) archives".to_string()
        })?;
        return td::inspect_td_archive(&path, &root, python_executable);
    }

    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if ext == "npy" {
        let file_meta = std::fs::metadata(p).map_err(|e| e.to_string())?;
        let mut file = File::open(p).map_err(|e| e.to_string())?;
        let mut header_buf = vec![0u8; 256];
        let n = file.read(&mut header_buf).map_err(|e| e.to_string())?;
        header_buf.truncate(n);
        let (shape, dtype, _) = npy::parse_npy_header(&header_buf)?;
        let key = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("array")
            .to_string();
        return Ok(NpzArchiveInfo {
            path,
            arrays: vec![NpzArrayInfo {
                key,
                shape,
                dtype,
                size_bytes: file_meta.len(),
            }],
            total_bytes: file_meta.len(),
            used_memmap: file_meta.len() > MMAP_THRESHOLD_BYTES,
        });
    }

    let file = File::open(p).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid NPZ: {e}"))?;
    let mut arrays = Vec::new();
    let mut total_bytes = 0u64;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = entry.name().to_string();
        if !name.ends_with(".npy") {
            continue;
        }
        let key = name.trim_end_matches(".npy").to_string();
        let (shape, dtype, payload) = npy::read_zip_npy_entry(&mut entry)?;
        let size_bytes = payload.len() as u64;
        total_bytes += size_bytes;
        arrays.push(NpzArrayInfo {
            key,
            shape,
            dtype,
            size_bytes,
        });
    }

    arrays.sort_by(|a, b| a.key.cmp(&b.key));
    let used_memmap = arrays.iter().any(|a| a.size_bytes > MMAP_THRESHOLD_BYTES);
    Ok(NpzArchiveInfo {
        path,
        arrays,
        total_bytes,
        used_memmap,
    })
}

#[tauri::command]
pub fn load_tensor_slice(
    path: String,
    key: Option<String>,
    indices: Option<Vec<usize>>,
    max_dim: Option<usize>,
    project_root: Option<String>,
    python_executable: Option<String>,
) -> Result<TensorSlicePreview, String> {
    let start = Instant::now();
    let max_dim = max_dim.unwrap_or(64).clamp(8, 256);
    let indices = indices.unwrap_or_default();

    let p = Path::new(&path);
    if td::is_td_path(p) {
        let root = project_root
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "Project root required to load TensorDict (.td) slices".to_string())?;
        let k = key.ok_or("TensorDict (.td) requires a key")?;
        return td::load_td_slice(&path, &k, &indices, max_dim, &root, python_executable);
    }

    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let resolved_key = if ext == "npy" {
        key.unwrap_or_else(|| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("array")
                .to_string()
        })
    } else {
        key.ok_or("NPZ requires a key")?
    };

    let meta = std::fs::metadata(p).map_err(|e| e.to_string())?;
    let use_npy_mmap = ext == "npy" && meta.len() > MMAP_THRESHOLD_BYTES;

    let (full_shape, plane_shape, flat, used_memmap, used_decompress_slice) = if use_npy_mmap {
        let (plane_shape, flat, mmap_used) = npy::load_npy_plane_mmap(p, &indices)?;
        let mut header_buf = vec![0u8; 256];
        let mut file = File::open(p).map_err(|e| e.to_string())?;
        let n = file.read(&mut header_buf).map_err(|e| e.to_string())?;
        header_buf.truncate(n);
        let (shape, _, _) = npy::parse_npy_header(&header_buf)?;
        (shape, plane_shape, flat, mmap_used, false)
    } else if ext == "npz" {
        match npy::load_npz_plane_mmap(p, &resolved_key, &indices) {
            Ok((shape, plane_shape, flat, mmap_used)) => {
                (shape, plane_shape, flat, mmap_used, false)
            }
            Err(_) => match npy::load_npz_plane_decompress(p, &resolved_key, &indices) {
                Ok((shape, plane_shape, flat, decomp_used)) => {
                    (shape, plane_shape, flat, false, decomp_used)
                }
                Err(_) => {
                    let (shape, arr) = npy::open_npz_array(&path, &resolved_key)?;
                    let (plane_shape, flat) = npy::select_2d_slice(&arr, &indices)?;
                    (shape, plane_shape, flat, false, false)
                }
            },
        }
    } else {
        let arr = npy::load_array_from_path(p)?;
        let shape = arr.shape().to_vec();
        let (plane_shape, flat) = npy::select_2d_slice(&arr, &indices)?;
        (shape, plane_shape, flat, false, false)
    };

    let rows = plane_shape[0];
    let cols = plane_shape.get(1).copied().unwrap_or(1);
    let (values, slice_shape) = npy::downsample_2d(&flat, rows, cols, max_dim);
    let (min, max) = npy::min_max(&values);

    Ok(TensorSlicePreview {
        key: resolved_key,
        full_shape,
        slice_shape,
        indices,
        values,
        min,
        max,
        rust_ms: start.elapsed().as_millis() as u64,
        used_memmap,
        used_decompress_slice,
    })
}

#[tauri::command]
pub fn tensor_slice_to_arrow_ipc(
    path: String,
    key: Option<String>,
    indices: Option<Vec<usize>>,
    max_dim: Option<usize>,
    project_root: Option<String>,
    python_executable: Option<String>,
) -> Result<ArrowIpcFile, String> {
    let start = Instant::now();
    let preview = load_tensor_slice(path, key, indices, max_dim, project_root, python_executable)?;
    let rows = preview.slice_shape[0];
    let cols = preview.slice_shape.get(1).copied().unwrap_or(1);

    let n = rows * cols;
    let mut row_b = Int32Builder::with_capacity(n);
    let mut col_b = Int32Builder::with_capacity(n);
    let mut val_b = Float64Builder::with_capacity(n);

    for (r, row) in preview.values.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            row_b.append_value(r as i32);
            col_b.append_value(c as i32);
            val_b.append_value(v);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("row", DataType::Int32, false),
        Field::new("col", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(row_b.finish()),
            Arc::new(col_b.finish()),
            Arc::new(val_b.finish()),
        ],
    )
    .map_err(|e| e.to_string())?;

    export_batch(&batch, "tensor", start)
}

/// Load 1-D (or scalar) arrays from an NPZ archive — used for θ axes and BPC markers (§G.5.2).
#[tauri::command]
pub fn load_npz_vectors(path: String, keys: Vec<String>) -> Result<Vec<NpzVectorData>, String> {
    let mut out = Vec::new();
    for key in keys {
        let (shape, arr) = npy::open_npz_array(&path, &key)?;
        let flat: Vec<f64> = arr.iter().copied().collect();
        if flat.is_empty() {
            return Err(format!("Key '{key}' is empty"));
        }
        if shape.len() > 1 {
            return Err(format!(
                "Key '{key}' has rank {} — expected 0-D or 1-D vector",
                shape.len()
            ));
        }
        out.push(NpzVectorData { key, values: flat });
    }
    Ok(out)
}

/// Load an NPZ array of any rank as a flattened vector — used by the native
/// Report Studio dataset statistics engine (§H.1 raw waste matrices).
#[tauri::command]
pub fn load_npz_flat(path: String, key: String) -> Result<Vec<f64>, String> {
    let (_, arr) = npy::open_npz_array(&path, &key)?;
    Ok(arr.iter().copied().collect())
}

/// Memory-map probe: reports whether a `.npy` or stored `.npz` entry is large enough for mmap.
#[tauri::command]
pub fn probe_npy_mmap(path: String) -> Result<bool, String> {
    let p = Path::new(&path);
    let meta = std::fs::metadata(p).map_err(|e| e.to_string())?;
    if meta.len() < 128 {
        return Ok(false);
    }
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext == "npz" {
        let file = File::open(p).map_err(|e| e.to_string())?;
        let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;
        for i in 0..archive.len() {
            let entry = archive.by_index(i).map_err(|e| e.to_string())?;
            if !entry.name().ends_with(".npy") {
                continue;
            }
            let loc = npy::npz_entry_loc(&entry);
            if loc.uncompressed_size > MMAP_THRESHOLD_BYTES {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    let mut file = File::open(p).map_err(|e| e.to_string())?;
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic).map_err(|e| e.to_string())?;
    Ok(magic == *b"\x93NUMPY" && meta.len() > MMAP_THRESHOLD_BYTES)
}
