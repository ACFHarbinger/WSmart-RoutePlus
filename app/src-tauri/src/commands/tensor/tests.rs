use super::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use zip::CompressionMethod;

fn write_test_npy(path: &Path, shape: &[usize], values: &[f64]) {
    let shape_str = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({shape_str},), }}");
    let pad = (16 - (10 + header.len()) % 16) % 16;
    let header_padded = format!("{header}{}", " ".repeat(pad));
    let header_len = header_padded.len() as u16;
    let mut file = File::create(path).unwrap();
    file.write_all(b"\x93NUMPY\x01\x00").unwrap();
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(header_padded.as_bytes()).unwrap();
    for v in values {
        file.write_all(&v.to_le_bytes()).unwrap();
    }
}

#[test]
fn mmap_plane_reads_trailing_2d_slice() {
    let dir = std::env::temp_dir().join(format!("wsroute_mmap_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tensor.npy");
    // shape (2, 3, 4) — plane at index 1 is 3×4
    let mut vals = Vec::new();
    for b in 0..2 {
        for r in 0..3 {
            for c in 0..4 {
                vals.push((b * 100 + r * 10 + c) as f64);
            }
        }
    }
    write_test_npy(&path, &[2, 3, 4], &vals);

    let (plane_shape, flat, used) = npy::load_npy_plane_mmap(&path, &[1]).unwrap();
    assert!(used);
    assert_eq!(plane_shape, vec![3, 4]);
    assert_eq!(flat.len(), 12);
    assert!((flat[0] - 100.0).abs() < 1e-9);
    assert!((flat[11] - 123.0).abs() < 1e-9);

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn npz_mmap_plane_reads_trailing_2d_slice() {
    use zip::write::SimpleFileOptions;
    use zip::ZipWriter;

    let dir = std::env::temp_dir().join(format!("wsroute_npz_mmap_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let npy_path = dir.join("tensor.npy");
    let mut vals = Vec::new();
    for b in 0..2 {
        for r in 0..3 {
            for c in 0..4 {
                vals.push((b * 100 + r * 10 + c) as f64);
            }
        }
    }
    write_test_npy(&npy_path, &[2, 3, 4], &vals);

    let npz_path = dir.join("tensor.npz");
    let npy_bytes = std::fs::read(&npy_path).unwrap();
    let file = File::create(&npz_path).unwrap();
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file("tensor.npy", options).unwrap();
    zip.write_all(&npy_bytes).unwrap();
    zip.finish().unwrap();

    let (shape, plane_shape, flat, used) =
        npy::load_npz_plane_mmap(&npz_path, "tensor", &[1]).unwrap();
    assert!(used);
    assert_eq!(shape, vec![2, 3, 4]);
    assert_eq!(plane_shape, vec![3, 4]);
    assert_eq!(flat.len(), 12);
    assert!((flat[0] - 100.0).abs() < 1e-9);

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn npz_decompress_plane_reads_trailing_2d_slice() {
    use zip::write::SimpleFileOptions;
    use zip::ZipWriter;

    let dir = std::env::temp_dir().join(format!("wsroute_npz_deflate_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let npy_path = dir.join("tensor.npy");
    let mut vals = Vec::new();
    for b in 0..2 {
        for r in 0..3 {
            for c in 0..4 {
                vals.push((b * 100 + r * 10 + c) as f64);
            }
        }
    }
    write_test_npy(&npy_path, &[2, 3, 4], &vals);

    let npz_path = dir.join("tensor_deflate.npz");
    let npy_bytes = std::fs::read(&npy_path).unwrap();
    let file = File::create(&npz_path).unwrap();
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    zip.start_file("tensor.npy", options).unwrap();
    zip.write_all(&npy_bytes).unwrap();
    zip.finish().unwrap();

    let (shape, plane_shape, flat, used) =
        npy::load_npz_plane_decompress(&npz_path, "tensor", &[1]).unwrap();
    assert!(used);
    assert_eq!(shape, vec![2, 3, 4]);
    assert_eq!(plane_shape, vec![3, 4]);
    assert_eq!(flat.len(), 12);
    assert!((flat[11] - 123.0).abs() < 1e-9);

    let _ = std::fs::remove_dir_all(dir);
}
