use super::*;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn write_sample_run(dir: &Path) -> std::io::Result<()> {
    let log = dir.join("sim_log.jsonl");
    let mut f = File::create(log)?;
    writeln!(
        f,
        r#"GUI_DAY_LOG_START:test_policy,0,1,{{"profit":10.0,"km":5.0,"overflows":0,"kg":42.0,"kg/km":8.4,"cost":-32.0,"ncol":10,"kg_lost":0.0}}"#,
    )?;
    writeln!(
        f,
        r#"GUI_DAY_LOG_START:test_policy,0,2,{{"profit":11.5,"km":5.2,"overflows":1,"kg":40.0,"kg/km":7.7,"cost":-28.5,"ncol":9,"kg_lost":0.5}}"#,
    )?;
    Ok(())
}

fn write_sample_csv(dir: &Path) -> std::io::Result<()> {
    let csv = dir.join("metrics.csv");
    let mut f = File::create(csv)?;
    writeln!(f, "policy,profit,overflows")?;
    writeln!(f, "gurobi,12.5,0")?;
    writeln!(f, "alns,10.1,2")?;
    Ok(())
}

#[test]
fn parse_training_health_line_valid() {
    let line = r#"TRAINING_HEALTH_START:{"code":"grad_norm_explosion","severity":"critical","epoch":2,"step":50,"message":"test","details":{"grad_norm":150.0}}"#;
    let entry = parse_training_health_line(line).expect("parse");
    assert_eq!(entry.code, "grad_norm_explosion");
    assert_eq!(entry.severity, "critical");
    assert_eq!(entry.epoch, 2);
    assert_eq!(entry.step, 50);
}

#[test]
fn parse_training_health_line_ignores_policy_viz() {
    let line = r#"POLICY_VIZ_START:ALNS,0,1,alns,{"iteration":[0]}"#;
    assert!(parse_training_health_line(line).is_none());
}

#[test]
fn parse_attention_viz_line_valid() {
    let line = r#"ATTENTION_VIZ_START:{"phase":"val","epoch":1,"step":10,"snapshots":[{"layer":0,"head":0,"decode_step":0,"n_nodes":3,"matrix":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]}]}"#;
    let entry = parse_attention_viz_line(line).expect("parse");
    assert_eq!(entry.phase, "val");
    assert_eq!(entry.epoch, 1);
    assert_eq!(entry.step, 10);
    assert_eq!(entry.snapshots.len(), 1);
    assert_eq!(entry.snapshots[0].n_nodes, 3);
}

#[test]
fn parse_attention_viz_line_ignores_training_health() {
    let line = r#"TRAINING_HEALTH_START:{"code":"grad_norm_explosion","severity":"critical","epoch":0,"step":0,"message":"x","details":{}}"#;
    assert!(parse_attention_viz_line(line).is_none());
}

#[test]
fn wsroute_bundle_round_trip_preserves_jsonl() {
    let base = std::env::temp_dir().join(format!(
        "wsroute_test_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let source = base.join("source");
    let bundle_path = base.join("run.wsroute");
    let extract_dir = base.join("extracted");

    fs::create_dir_all(&source).expect("create source");
    write_sample_run(&source).expect("write sample run");

    let info = create_wsroute_bundle(
        source.to_string_lossy().to_string(),
        bundle_path.to_string_lossy().to_string(),
        None,
    )
    .expect("create bundle");
    assert!(info.files.iter().any(|f| f.path.ends_with(".jsonl")));

    let extracted = extract_wsroute_bundle(
        bundle_path.to_string_lossy().to_string(),
        extract_dir.to_string_lossy().to_string(),
    )
    .expect("extract bundle");

    let log_path = extracted.log_path.expect("jsonl path");
    let content = fs::read_to_string(&log_path).expect("read extracted log");
    assert!(content.contains("GUI_DAY_LOG_START:test_policy"));

    let entries = load_simulation_log(log_path).expect("parse extracted log");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].policy, "test_policy");
    assert_eq!(entries[0].day, 1);

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn simulation_arrow_sidecar_row_parity() {
    let base = std::env::temp_dir().join(format!(
        "sim_arrow_parity_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&base).expect("create base");
    write_sample_run(&base).expect("write sample run");

    let log_path = base.join("sim_log.jsonl");
    let arrow_path = base.join("sim_log.arrow");
    let entries = load_simulation_log(log_path.to_string_lossy().to_string())
        .expect("load simulation log");
    let row_count =
        crate::commands::arrow::write_simulation_log_arrow_sidecar(&log_path.to_string_lossy(), &arrow_path)
            .expect("write simulation arrow sidecar");
    assert_eq!(row_count, entries.len());

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn wsroute_bundle_with_arrow_sidecars() {
    let base = std::env::temp_dir().join(format!(
        "wsroute_arrow_test_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let source = base.join("source");
    let bundle_path = base.join("run.wsroute");

    fs::create_dir_all(&source).expect("create source");
    write_sample_run(&source).expect("write sample run");
    write_sample_csv(&source).expect("write sample csv");

    let info = create_wsroute_bundle(
        source.to_string_lossy().to_string(),
        bundle_path.to_string_lossy().to_string(),
        Some(true),
    )
    .expect("create bundle with arrow");

    assert_eq!(info.arrow_sidecars, Some(2));
    assert!(info.files.iter().any(|f| f.path.ends_with("metrics.arrow")));
    assert!(info.files.iter().any(|f| f.path.ends_with("sim_log.arrow")));

    let _ = fs::remove_dir_all(&base);
}
