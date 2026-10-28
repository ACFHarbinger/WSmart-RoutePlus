use super::scan_roots;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("wsr_corpus_{name}_{}_{nanos}", std::process::id()));
        fs::create_dir_all(&path).expect("temp dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn scan_simulation_directory_keeps_two_json_files() {
    let scratch = Scratch::new("run");
    let run = scratch.path().join("run");
    fs::create_dir(&run).unwrap();
    fs::write(
        run.join("one.json"),
        r#"{
            "day": 1,
            "policy": "alns",
            "city": "Porto",
            "profit": 10,
            "kg_lost": 1.5,
            "all_bin_coords": [
                {"id": 1, "lat": 10, "lng": 20},
                {"id": 2, "lat": 30, "lng": 40}
            ],
            "bin_state_c": [0.25, 0.5]
        }"#,
    )
    .unwrap();
    fs::write(
        run.join("two.json"),
        r#"[{"day": 2, "policy_name": "hgs", "city_name": "Lisbon", "kg/km": 3}]"#,
    )
    .unwrap();

    let result = scan_roots(vec![run.to_string_lossy().to_string()]);
    assert_eq!(result.kept, 2);
    assert_eq!(result.skipped, 0);
    assert!(result.failed.is_empty());
    assert_eq!(result.files.len(), 2);
    assert!(result.files.iter().all(|file| !file.partial));

    let first = result
        .files
        .iter()
        .find(|file| file.path.ends_with("one.json"))
        .unwrap();
    let day = &first.days[0];
    assert_eq!(day.policy.as_deref(), Some("alns"));
    assert_eq!(day.city.as_deref(), Some("Porto"));
    assert_eq!(
        day.metrics.get("profit").and_then(serde_json::Value::as_i64),
        Some(10)
    );
    assert!(day.metrics.get("km").is_none());
    assert_eq!(day.kg_lost, Some(1.5));
    assert_eq!(day.lat, Some(20.0));
    assert_eq!(day.lon, Some(30.0));
    assert_eq!(day.bins.len(), 2);
    assert_eq!(day.bins[0].bin_id, "1");
    assert_eq!(day.bins[0].fill, Some(0.25));
    let encoded = serde_json::to_value(day).unwrap();
    assert_eq!(encoded["kgLost"], 1.5);
    assert!(encoded.get("kg_lost").is_none());
    assert_eq!(encoded["bins"][0]["binId"], "1");

    let second = result
        .files
        .iter()
        .find(|file| file.path.ends_with("two.json"))
        .unwrap();
    assert_eq!(second.days[0].policy.as_deref(), Some("hgs"));
    assert_eq!(second.days[0].city.as_deref(), Some("Lisbon"));
    assert_eq!(
        second.days[0]
            .metrics
            .get("kg_per_km")
            .and_then(serde_json::Value::as_i64),
        Some(3)
    );
    assert!(second.days[0].metrics.get("km").is_none());
}

#[test]
fn scan_simulation_file_root_is_partial() {
    let scratch = Scratch::new("partial");
    let file = scratch.path().join("one.json");
    fs::write(&file, r#"{"day": 4, "full_policy": "bpc", "cost": 2}"#).unwrap();
    let result = scan_roots(vec![file.to_string_lossy().to_string()]);
    assert_eq!(result.files.len(), 1);
    assert!(result.files[0].partial);
    assert_eq!(result.kept, 1);
    assert_eq!(result.files[0].days[0].policy.as_deref(), Some("bpc"));
    assert_eq!(
        result.files[0].days[0]
            .metrics
            .get("cost")
            .and_then(serde_json::Value::as_i64),
        Some(2)
    );
}

#[test]
fn scan_simulation_day_log_reads_profit_only() {
    let scratch = Scratch::new("jsonl");
    let file = scratch.path().join("days.jsonl");
    fs::write(
        &file,
        "GUI_DAY_LOG_START:greedy,0,3,{\"profit\":12.5,\"kg_lost\":1.5}\n",
    )
    .unwrap();
    let result = scan_roots(vec![file.to_string_lossy().to_string()]);
    assert_eq!(result.kept, 1);
    assert!(result.files[0].partial);
    let day = &result.files[0].days[0];
    assert_eq!(day.policy.as_deref(), Some("greedy"));
    assert_eq!(day.day.as_ref().and_then(serde_json::Number::as_i64), Some(3));
    assert_eq!(
        day.metrics.get("profit").and_then(serde_json::Value::as_f64),
        Some(12.5)
    );
    assert!(day.metrics.get("km").is_none());
    assert_eq!(day.metrics.len(), 1);
    assert_eq!(day.kg_lost, Some(1.5));
    assert!(day.bins.is_empty());
    assert!(day.tour.is_empty());
    assert!(day.tour_indices.is_empty());
    assert!(day.route_bins.is_empty());
}

#[test]
fn scan_simulation_broken_json_keeps_good_file() {
    let scratch = Scratch::new("mixed");
    let good = scratch.path().join("good.json");
    let bad = scratch.path().join("bad.json");
    fs::write(&good, r#"{"policy": "alns", "day": 1, "kg": 4}"#).unwrap();
    fs::write(&bad, "{").unwrap();
    let result = scan_roots(vec![scratch.path().to_string_lossy().to_string()]);
    assert_eq!(result.files.len(), 1);
    assert!(result.files[0].path.ends_with("good.json"));
    assert!(!result.files[0].partial);
    assert_eq!(result.failed.len(), 1);
    assert_eq!(result.failed[0].path, bad.to_string_lossy());
    assert_eq!(result.failed[0].reason, "invalid json");
    assert_eq!(result.kept, 1);
}

#[test]
fn scan_simulation_txt_is_skipped() {
    let scratch = Scratch::new("txt");
    fs::write(scratch.path().join("notes.txt"), "hello").unwrap();
    let result = scan_roots(vec![scratch.path().to_string_lossy().to_string()]);
    assert!(result.files.is_empty());
    assert_eq!(result.skipped, 1);
    assert_eq!(result.kept, 0);
    assert!(result.failed.is_empty());
}

#[test]
fn scan_simulation_empty_directory() {
    let scratch = Scratch::new("empty");
    let result = scan_roots(vec![scratch.path().to_string_lossy().to_string()]);
    assert!(result.files.is_empty());
    assert_eq!(result.kept, 0);
    assert_eq!(result.skipped, 0);
    assert!(result.failed.is_empty());
    assert_eq!(result.roots.len(), 1);
}

#[test]
fn scan_simulation_reads_tour_and_route_bins() {
    let scratch = Scratch::new("route");
    let run = scratch.path().join("run");
    fs::create_dir(&run).unwrap();
    fs::write(
        run.join("tour.json"),
        r#"{
            "day": 1,
            "profit": 8,
            "tour": [0, 2, 0],
            "tour_indices": [2],
            "all_bin_coords": [{"id": 1, "lat": 41.1, "lng": -8.6}]
        }"#,
    )
    .unwrap();
    fs::write(
        run.join("shapes.json"),
        r#"{
            "profit": 1,
            "tour": [{"id": 0}, {"bin_id": -1}, {"binId": "4"}],
            "tourIndices": [1, -1],
            "bins": [
                {"id": "4", "latitude": 10, "longitude": 20},
                {"bin_id": 5, "fill": 0.2}
            ]
        }"#,
    )
    .unwrap();
    fs::write(run.join("profit.json"), r#"{"profit": 4}"#).unwrap();
    fs::write(
        run.join("days.jsonl"),
        "GUI_DAY_LOG_START:alns,0,2,{\"tour\":[0,-1,0],\"all_bin_coords\":[{\"id\":-1,\"lat\":41.1,\"lng\":-8.6}]}\n",
    )
    .unwrap();

    let result = scan_roots(vec![run.to_string_lossy().to_string()]);
    assert_eq!(result.kept, 4);
    assert!(result.failed.is_empty());

    let tour_day = &result
        .files
        .iter()
        .find(|file| file.path.ends_with("tour.json"))
        .unwrap()
        .days[0];
    assert_eq!(tour_day.tour, vec![0, 2, 0]);
    assert_eq!(tour_day.tour_indices, vec![2]);
    assert_eq!(tour_day.route_bins.len(), 1);
    assert_eq!(tour_day.route_bins[0].id, 1);
    assert_eq!(tour_day.route_bins[0].lat, Some(41.1));
    assert_eq!(tour_day.route_bins[0].lon, Some(-8.6));
    let encoded = serde_json::to_value(tour_day).unwrap();
    assert_eq!(encoded["tour"], serde_json::json!([0, 2, 0]));
    assert_eq!(encoded["tourIndices"], serde_json::json!([2]));
    assert_eq!(encoded["routeBins"][0]["lat"].as_f64(), Some(41.1));
    assert_eq!(encoded["routeBins"][0]["lon"].as_f64(), Some(-8.6));
    assert!(encoded.get("route_bins").is_none());
    assert!(encoded.get("tour_indices").is_none());

    let shapes = &result
        .files
        .iter()
        .find(|file| file.path.ends_with("shapes.json"))
        .unwrap()
        .days[0];
    assert_eq!(shapes.tour, vec![0, -1, 4]);
    assert_eq!(shapes.tour_indices, vec![1, -1]);
    assert_eq!(shapes.route_bins.len(), 1);
    assert_eq!(shapes.route_bins[0].id, 4);
    assert_eq!(shapes.route_bins[0].lat, Some(10.0));
    assert_eq!(shapes.route_bins[0].lon, Some(20.0));

    let profit = &result
        .files
        .iter()
        .find(|file| file.path.ends_with("profit.json"))
        .unwrap()
        .days[0];
    assert!(profit.tour.is_empty());
    assert!(profit.tour_indices.is_empty());
    assert!(profit.route_bins.is_empty());
    assert_eq!(
        profit.metrics.get("profit").and_then(serde_json::Value::as_i64),
        Some(4)
    );

    let logged = &result
        .files
        .iter()
        .find(|file| file.path.ends_with("days.jsonl"))
        .unwrap()
        .days[0];
    assert_eq!(logged.tour, vec![0, -1, 0]);
    assert_eq!(logged.route_bins.len(), 1);
    assert_eq!(logged.route_bins[0].id, -1);
    assert_eq!(logged.route_bins[0].lat, Some(41.1));
    assert_eq!(logged.route_bins[0].lon, Some(-8.6));
}
