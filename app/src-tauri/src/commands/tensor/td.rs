use std::path::Path;
use std::process::Command;

use crate::commands::process::resolve_python;
use super::{NpzArchiveInfo, TensorSlicePreview};

const INSPECT_TD_SCRIPT: &str = r#"
import json, sys
path = sys.argv[1]
try:
    import torch
except ImportError as e:
    print(json.dumps({"error": f"PyTorch required for .td files: {e}"}))
    sys.exit(0)
try:
    td = torch.load(path, map_location="cpu", weights_only=False)
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(0)
if not hasattr(td, "keys"):
    print(json.dumps({"error": "File does not contain a TensorDict-like object"}))
    sys.exit(0)
arrays = []
total_bytes = 0
for key in sorted(td.keys()):
    t = td[key]
    if not hasattr(t, "shape"):
        continue
    shape = list(t.shape)
    dtype = str(getattr(t, "dtype", "float32")).replace("torch.", "")
    nbytes = int(t.numel() * t.element_size())
    arrays.append({"key": key, "shape": shape, "dtype": dtype, "size_bytes": nbytes})
    total_bytes += nbytes
print(json.dumps({
    "path": path,
    "arrays": arrays,
    "total_bytes": total_bytes,
    "used_memmap": False,
}))
"#;

const LOAD_TD_SLICE_SCRIPT: &str = r#"
import json, sys
path, key, indices_json, max_dim = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
indices = json.loads(indices_json)
try:
    import torch
    import numpy as np
except ImportError as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(0)
try:
    td = torch.load(path, map_location="cpu", weights_only=False)
    arr = td[key].detach().float().cpu().numpy()
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(0)
shape = list(arr.shape)
if len(shape) == 0:
    print(json.dumps({"error": "Empty tensor"}))
    sys.exit(0)
if len(shape) == 1:
    plane = arr.reshape(-1, 1)
elif len(shape) == 2:
    plane = arr
else:
    leading = len(shape) - 2
    if len(indices) < leading:
        print(json.dumps({"error": f"Need {leading} leading indices, got {len(indices)}"}))
        sys.exit(0)
    rows, cols = shape[-2], shape[-1]
    prefix = indices[:leading]
    out = []
    for r in range(rows):
        for c in range(cols):
            idx = tuple(prefix + [r, c])
            out.append(float(arr[idx]))
    plane = np.array(out, dtype=np.float64).reshape(rows, cols)
rows, cols = plane.shape
stride_r = max(1, rows // max_dim)
stride_c = max(1, cols // max_dim)
grid = plane[::stride_r, ::stride_c]
values = grid.tolist()
flat = [v for row in values for v in row]
vmin = float(np.min(flat)) if flat else 0.0
vmax = float(np.max(flat)) if flat else 0.0
print(json.dumps({
    "key": key,
    "full_shape": shape,
    "slice_shape": [len(values), len(values[0]) if values else 0],
    "indices": indices,
    "values": values,
    "min": vmin,
    "max": vmax,
}))
"#;

pub fn is_td_path(path: &Path) -> bool {
    path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("td"))
        .unwrap_or(false)
}

pub fn inspect_td_archive(
    path: &str,
    project_root: &str,
    python_executable: Option<String>,
) -> Result<NpzArchiveInfo, String> {
    let python = resolve_python(project_root, python_executable);
    let output = Command::new(&python)
        .arg("-c")
        .arg(INSPECT_TD_SCRIPT)
        .arg(path)
        .current_dir(project_root)
        .output()
        .map_err(|e| format!("Failed to inspect .td archive: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("TensorDict inspect produced no output: {stderr}"));
    }

    let value: serde_json::Value =
        serde_json::from_str(&stdout).map_err(|e| format!("Invalid inspect JSON: {e}"))?;
    if let Some(err) = value.get("error").and_then(|v| v.as_str()) {
        return Err(err.to_string());
    }
    serde_json::from_value(value).map_err(|e| e.to_string())
}

pub fn load_td_slice(
    path: &str,
    key: &str,
    indices: &[usize],
    max_dim: usize,
    project_root: &str,
    python_executable: Option<String>,
) -> Result<TensorSlicePreview, String> {
    let python = resolve_python(project_root, python_executable);
    let indices_json = serde_json::to_string(indices).map_err(|e| e.to_string())?;

    let output = Command::new(&python)
        .arg("-c")
        .arg(LOAD_TD_SLICE_SCRIPT)
        .arg(path)
        .arg(key)
        .arg(&indices_json)
        .arg(max_dim.to_string())
        .current_dir(project_root)
        .output()
        .map_err(|e| format!("Failed to load .td slice: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("TensorDict slice produced no output: {stderr}"));
    }

    let value: serde_json::Value =
        serde_json::from_str(&stdout).map_err(|e| format!("Invalid slice JSON: {e}"))?;
    if let Some(err) = value.get("error").and_then(|v| v.as_str()) {
        return Err(err.to_string());
    }

    let key = value
        .get("key")
        .and_then(|v| v.as_str())
        .unwrap_or(key)
        .to_string();
    let full_shape: Vec<usize> = serde_json::from_value(value["full_shape"].clone())
        .map_err(|e| format!("Invalid full_shape: {e}"))?;
    let slice_shape: Vec<usize> = serde_json::from_value(value["slice_shape"].clone())
        .map_err(|e| format!("Invalid slice_shape: {e}"))?;
    let values: Vec<Vec<f64>> = serde_json::from_value(value["values"].clone())
        .map_err(|e| format!("Invalid values: {e}"))?;
    let min = value.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let max = value.get("max").and_then(|v| v.as_f64()).unwrap_or(0.0);

    Ok(TensorSlicePreview {
        key,
        full_shape,
        slice_shape,
        indices: indices.to_vec(),
        values,
        min,
        max,
        rust_ms: 0,
        used_memmap: false,
        used_decompress_slice: false,
    })
}
