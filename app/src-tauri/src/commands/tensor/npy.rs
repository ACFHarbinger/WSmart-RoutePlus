use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;
use memmap2::Mmap;
use ndarray::{ArrayD, IxDyn};
use ndarray_npy::ReadNpyExt;
use zip::read::ZipFile;
use zip::{CompressionMethod, ZipArchive};

use super::MMAP_THRESHOLD_BYTES;

pub fn itemsize_from_descr(descr: &str) -> Result<usize, String> {
    let d = descr.trim_matches(|c| c == '\'' || c == '"');
    if d.ends_with("f8") || d.ends_with("i8") || d.ends_with("u8") || d.ends_with("l") {
        Ok(8)
    } else if d.ends_with("f4") || d.ends_with("i4") || d.ends_with("u4") || d.ends_with("f") {
        Ok(4)
    } else if d.ends_with("f2") || d.ends_with("i2") || d.ends_with("u2") {
        Ok(2)
    } else if d.ends_with("i1") || d.ends_with("u1") || d.ends_with("b1") {
        Ok(1)
    } else {
        Err(format!("Unsupported NPY dtype for mmap slice: {descr}"))
    }
}

pub fn c_order_offset(shape: &[usize], indices: &[usize]) -> Result<usize, String> {
    if shape.len() < 2 {
        return Ok(0);
    }
    let leading = shape.len() - 2;
    if indices.len() < leading {
        return Err(format!(
            "Need {} leading indices for rank-{} tensor, got {}",
            leading,
            shape.len(),
            indices.len()
        ));
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let mut offset = 0usize;
    for i in 0..leading {
        offset += indices[i] * strides[i];
    }
    Ok(offset)
}

pub fn read_f64_plane(bytes: &[u8], itemsize: usize) -> Result<Vec<f64>, String> {
    if bytes.len() % itemsize != 0 {
        return Err("Mmap plane byte length is not aligned to dtype width".to_string());
    }
    let count = bytes.len() / itemsize;
    let mut out = Vec::with_capacity(count);
    match itemsize {
        8 => {
            for chunk in bytes.chunks_exact(8) {
                out.push(f64::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        4 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes(chunk.try_into().unwrap()) as f64);
            }
        }
        2 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(f16_to_f64(u16::from_le_bytes(chunk.try_into().unwrap())));
            }
        }
        1 => {
            for &b in bytes {
                out.push(b as f64);
            }
        }
        _ => return Err(format!("Unsupported itemsize {itemsize}")),
    }
    Ok(out)
}

pub fn f16_to_f64(bits: u16) -> f64 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;
    if exp == 0 {
        if frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = (frac as f64) / 1024.0 * 2.0_f64.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 0x1f {
        return if frac == 0 {
            if sign == 1 { f64::NEG_INFINITY } else { f64::INFINITY }
        } else {
            f64::NAN
        };
    }
    let val = (1.0 + (frac as f64) / 1024.0) * 2.0_f64.powi(exp as i32 - 15);
    if sign == 1 { -val } else { val }
}

pub struct NpzNpyEntryLoc {
    pub data_start: u64,
    pub compression: CompressionMethod,
    pub uncompressed_size: u64,
}

pub fn resolve_npz_npy_key(archive: &mut ZipArchive<File>, key: &str) -> Result<String, String> {
    let npy_name = format!("{key}.npy");
    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|e| e.name().to_string()))
        .collect();
    if names.iter().any(|n| n == &npy_name) {
        Ok(npy_name)
    } else if names.iter().any(|n| n == key) {
        Ok(key.to_string())
    } else {
        Err(format!("Key '{key}' not found in NPZ"))
    }
}

pub fn resolve_npz_npy_entry(path: &Path, key: &str) -> Result<NpzNpyEntryLoc, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid NPZ: {e}"))?;
    let resolved = resolve_npz_npy_key(&mut archive, key)?;
    let entry = archive
        .by_name(&resolved)
        .map_err(|e| format!("Key '{key}' not found in NPZ: {e}"))?;
    Ok(npz_entry_loc(&entry))
}

pub fn npz_entry_loc(entry: &ZipFile<'_>) -> NpzNpyEntryLoc {
    NpzNpyEntryLoc {
        data_start: entry.data_start(),
        compression: entry.compression(),
        uncompressed_size: entry.size(),
    }
}

pub fn load_plane_from_npy_bytes(
    raw: &[u8],
    indices: &[usize],
) -> Result<(Vec<usize>, Vec<usize>, Vec<f64>), String> {
    let (shape, dtype, data_offset) = parse_npy_header(raw)?;
    let itemsize = itemsize_from_descr(&dtype)?;
    let rows = *shape.get(shape.len().saturating_sub(2)).unwrap_or(&0);
    let cols = shape.last().copied().unwrap_or(0);
    if rows == 0 || cols == 0 {
        return Err("Cannot slice empty tensor plane".to_string());
    }
    let elem_offset = c_order_offset(&shape, indices)?;
    let plane_elems = rows * cols;
    let byte_start = data_offset + elem_offset * itemsize;
    let byte_end = byte_start + plane_elems * itemsize;
    if byte_end > raw.len() {
        return Err("NPY plane slice exceeds buffer bounds".to_string());
    }
    let flat = read_f64_plane(&raw[byte_start..byte_end], itemsize)?;
    Ok((shape, vec![rows, cols], flat))
}

pub fn load_npz_plane_decompress(
    path: &Path,
    key: &str,
    indices: &[usize],
) -> Result<(Vec<usize>, Vec<usize>, Vec<f64>, bool), String> {
    let loc = resolve_npz_npy_entry(path, key)?;
    if loc.uncompressed_size <= MMAP_THRESHOLD_BYTES {
        return Err("NPZ entry below decompress-slice threshold".to_string());
    }
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid NPZ: {e}"))?;
    let resolved = resolve_npz_npy_key(&mut archive, key)?;
    let mut entry = archive
        .by_name(&resolved)
        .map_err(|e| format!("Key '{key}' not found in NPZ: {e}"))?;
    let mut raw = Vec::new();
    entry.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let (shape, plane_shape, flat) = load_plane_from_npy_bytes(&raw, indices)?;
    Ok((shape, plane_shape, flat, true))
}

pub fn load_npz_plane_mmap(
    path: &Path,
    key: &str,
    indices: &[usize],
) -> Result<(Vec<usize>, Vec<usize>, Vec<f64>, bool), String> {
    let loc = resolve_npz_npy_entry(path, key)?;
    if loc.compression != CompressionMethod::Stored {
        return Err("NPZ entry is compressed — mmap slice unavailable".to_string());
    }
    if loc.uncompressed_size <= MMAP_THRESHOLD_BYTES {
        return Err("NPZ entry below mmap threshold".to_string());
    }
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap failed: {e}"))? };
    let npy_start = loc.data_start as usize;
    if npy_start >= mmap.len() {
        return Err("NPZ data offset out of bounds".to_string());
    }
    let (shape, dtype, data_offset) = parse_npy_header(&mmap[npy_start..])?;
    let itemsize = itemsize_from_descr(&dtype)?;
    let rows = *shape.get(shape.len().saturating_sub(2)).unwrap_or(&0);
    let cols = shape.last().copied().unwrap_or(0);
    if rows == 0 || cols == 0 {
        return Err("Cannot mmap-slice empty tensor plane".to_string());
    }
    let elem_offset = c_order_offset(&shape, indices)?;
    let plane_elems = rows * cols;
    let byte_start = npy_start + data_offset + elem_offset * itemsize;
    let byte_end = byte_start + plane_elems * itemsize;
    if byte_end > mmap.len() {
        return Err("Mmap NPZ slice exceeds file bounds".to_string());
    }
    let flat = read_f64_plane(&mmap[byte_start..byte_end], itemsize)?;
    Ok((shape, vec![rows, cols], flat, true))
}

pub fn load_npy_plane_mmap(
    path: &Path,
    indices: &[usize],
) -> Result<(Vec<usize>, Vec<f64>, bool), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap failed: {e}"))? };
    let (shape, dtype, data_offset) = parse_npy_header(&mmap)?;
    let itemsize = itemsize_from_descr(&dtype)?;
    let rows = *shape.get(shape.len().saturating_sub(2)).unwrap_or(&0);
    let cols = shape.last().copied().unwrap_or(0);
    if rows == 0 || cols == 0 {
        return Err("Cannot mmap-slice empty tensor plane".to_string());
    }
    let elem_offset = c_order_offset(&shape, indices)?;
    let plane_elems = rows * cols;
    let byte_start = data_offset + elem_offset * itemsize;
    let byte_end = byte_start + plane_elems * itemsize;
    if byte_end > mmap.len() {
        return Err("Mmap slice exceeds file bounds".to_string());
    }
    let flat = read_f64_plane(&mmap[byte_start..byte_end], itemsize)?;
    Ok((vec![rows, cols], flat, true))
}

pub fn parse_npy_header(raw: &[u8]) -> Result<(Vec<usize>, String, usize), String> {
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        return Err("Invalid NPY magic header".to_string());
    }
    let header_len = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let header_start = 10;
    let header_end = header_start + header_len;
    if raw.len() < header_end {
        return Err("Truncated NPY header".to_string());
    }
    let header = String::from_utf8_lossy(&raw[header_start..header_end]);
    let shape = parse_shape_from_header(&header)?;
    let dtype = parse_descr_from_header(&header).unwrap_or_else(|| "unknown".to_string());
    Ok((shape, dtype, header_end))
}

pub fn parse_shape_from_header(header: &str) -> Result<Vec<usize>, String> {
    let start = header.find("'shape'").ok_or("shape missing in NPY header")?;
    let rest = &header[start..];
    let open = rest.find('(').ok_or("shape tuple missing")?;
    let close = rest[open..].find(')').ok_or("shape tuple unclosed")?;
    let inner = &rest[open + 1..open + close];
    if inner.trim().is_empty() {
        return Ok(vec![]);
    }
    Ok(inner
        .split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect())
}

pub fn parse_descr_from_header(header: &str) -> Option<String> {
    let start = header.find("'descr'")?;
    let rest = &header[start + "'descr'".len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('\'') {
        return None;
    }
    let value = &after[1..];
    let end = value.find('\'')?;
    Some(value[..end].to_string())
}

pub fn read_zip_npy_entry(
    entry: &mut zip::read::ZipFile<'_>,
) -> Result<(Vec<usize>, String, Vec<u8>), String> {
    let mut raw = Vec::new();
    entry.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let (shape, dtype, offset) = parse_npy_header(&raw)?;
    Ok((shape, dtype, raw[offset..].to_vec()))
}

pub fn load_array_from_bytes(bytes: &[u8]) -> Result<ArrayD<f64>, String> {
    let mut cursor = Cursor::new(bytes);
    ArrayD::read_npy(&mut cursor).map_err(|e| format!("NPY parse error: {e}"))
}

pub fn load_array_from_path(path: &Path) -> Result<ArrayD<f64>, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    ArrayD::read_npy(&mut file).map_err(|e| format!("NPY read error: {e}"))
}

pub fn select_2d_slice(arr: &ArrayD<f64>, indices: &[usize]) -> Result<(Vec<usize>, Vec<f64>), String> {
    let shape = arr.shape().to_vec();
    if shape.is_empty() {
        return Err("Empty tensor".to_string());
    }
    if shape.len() == 1 {
        let n = shape[0];
        return Ok((vec![n, 1], arr.iter().copied().collect()));
    }
    if shape.len() == 2 {
        return Ok((shape.clone(), arr.iter().copied().collect()));
    }

    let leading = shape.len() - 2;
    if indices.len() < leading {
        return Err(format!(
            "Need {} leading indices for rank-{} tensor, got {}",
            leading,
            shape.len(),
            indices.len()
        ));
    }

    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let mut idx: Vec<usize> = indices[..leading].to_vec();
            idx.push(r);
            idx.push(c);
            let v = arr
                .get(IxDyn(&idx))
                .copied()
                .ok_or_else(|| format!("Index out of bounds: {idx:?}"))?;
            out.push(v);
        }
    }
    Ok((vec![rows, cols], out))
}

pub fn downsample_2d(
    flat: &[f64],
    rows: usize,
    cols: usize,
    max_dim: usize,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    if rows == 0 || cols == 0 {
        return (vec![], vec![0, 0]);
    }
    let stride_r = (rows / max_dim).max(1);
    let stride_c = (cols / max_dim).max(1);
    let out_rows = (rows + stride_r - 1) / stride_r;
    let out_cols = (cols + stride_c - 1) / stride_c;
    let mut grid = vec![vec![0.0; out_cols]; out_rows];
    for (oi, r) in (0..rows).step_by(stride_r).enumerate() {
        for (oj, c) in (0..cols).step_by(stride_c).enumerate() {
            grid[oi][oj] = flat[r * cols + c];
        }
    }
    (grid, vec![out_rows, out_cols])
}

pub fn min_max(grid: &[Vec<f64>]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for row in grid {
        for &v in row {
            if v.is_finite() {
                min = min.min(v);
                max = max.max(v);
            }
        }
    }
    if !min.is_finite() {
        min = 0.0;
    }
    if !max.is_finite() {
        max = 0.0;
    }
    (min, max)
}

pub fn open_npz_array(
    path: &str,
    key: &str,
) -> Result<(Vec<usize>, ArrayD<f64>), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid NPZ: {e}"))?;
    let npy_name = format!("{key}.npy");
    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|e| e.name().to_string()))
        .collect();
    let resolved = if names.iter().any(|n| n == &npy_name) {
        npy_name
    } else if names.iter().any(|n| n == key) {
        key.to_string()
    } else {
        return Err(format!("Key '{key}' not found in NPZ"));
    };
    let mut entry = archive
        .by_name(&resolved)
        .map_err(|e| format!("Key '{key}' not found in NPZ: {e}"))?;
    let (shape, _dtype, payload) = read_zip_npy_entry(&mut entry)?;
    let arr = load_array_from_bytes(&payload)?;
    Ok((shape, arr))
}
