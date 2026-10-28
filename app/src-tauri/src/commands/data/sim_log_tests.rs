use std::path::Path;

use super::*;

#[test]
fn test_extract_policy_name_from_filename() {
    assert_eq!(
        extract_policy_name_from_filename("log_last_minute_cf90_bpc_custom_ftsp_1N.json"),
        "last_minute_cf90_bpc_custom_ftsp"
    );
    assert_eq!(
        extract_policy_name_from_filename("log_alns_custom_10N.json"),
        "alns_custom"
    );
    assert_eq!(
        extract_policy_name_from_filename("log_greedy.json"),
        "greedy"
    );
    assert_eq!(
        extract_policy_name_from_filename("log_realtime_emp_1N.jsonl"),
        "realtime_emp"
    );
}

#[test]
fn test_extract_sim_hierarchy_metadata() {
    let p = Path::new("outputs/simulations-cptp/90days/figueiradafoz350_plastic/gamma3/lm_ftsp/log_last_minute_cf90_bpc_custom_ftsp_1N.json");
    let meta = extract_sim_hierarchy_metadata(p, None);
    assert_eq!(meta.days, Some(90));
    assert_eq!(meta.region.as_deref(), Some("figueiradafoz"));
    assert_eq!(meta.num_bins, Some(350));
    assert_eq!(meta.waste_type.as_deref(), Some("plastic"));
    assert_eq!(meta.distribution.as_deref(), Some("gamma3"));
    assert_eq!(meta.run_name.as_deref(), Some("lm_ftsp"));
    assert_eq!(meta.policy, "last_minute_cf90_bpc_custom_ftsp");
    assert_eq!(meta.num_instances, Some(1));
}

#[test]
fn test_extract_sim_hierarchy_metadata_no_distribution() {
    let p = Path::new("outputs/simulations/30days/riomaior100_all/run_greedy/log_greedy_1N.json");
    let meta = extract_sim_hierarchy_metadata(p, None);
    assert_eq!(meta.days, Some(30));
    assert_eq!(meta.region.as_deref(), Some("riomaior"));
    assert_eq!(meta.num_bins, Some(100));
    assert_eq!(meta.waste_type.as_deref(), Some("all"));
    assert_eq!(meta.distribution, None);
    assert_eq!(meta.run_name.as_deref(), Some("run_greedy"));
    assert_eq!(meta.policy, "greedy");
    assert_eq!(meta.num_instances, Some(1));
}

#[test]
fn test_parse_simulation_json_log_mock() {
    let json_text = r#"{
            "samples": { "0": { "profit": 100.0 } },
            "daily": {
                "0": {
                    "day": [1, 2],
                    "overflows": [0.0, 1.0],
                    "kg": [50.0, 75.0],
                    "km": [10.0, 15.0],
                    "profit": [20.0, 30.0],
                    "tour": [[0, 1, 2], [0, 2]]
                }
            }
        }"#;

    let entries = parse_simulation_json_log(json_text, Some(Path::new("log_alns_1N.json"))).unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].policy, "alns");
    assert_eq!(entries[0].sample_id, 0);
    assert_eq!(entries[0].day, 1);
    assert_eq!(entries[0].data["kg"], 50.0);
    assert_eq!(entries[0].data["tour_indices"], serde_json::json!([0, 1]));

    assert_eq!(entries[1].day, 2);
    assert_eq!(entries[1].data["kg"], 75.0);
    assert_eq!(entries[1].data["tour_indices"], serde_json::json!([1]));
}
