use std::path::Path;

/// Known stochastic waste fill rate distributions used in the simulator.
pub fn is_known_distribution(s: &str) -> bool {
    let lower = s.to_lowercase();
    matches!(
        lower.as_str(),
        "emp"
            | "empirical"
            | "gamma1"
            | "gamma2"
            | "gamma3"
            | "gamma"
            | "poisson"
            | "normal"
            | "gaussian"
            | "uniform"
            | "beta"
            | "weibull"
            | "lognormal"
    )
}

/// Extract clean policy name from a log filename (e.g. `log_last_minute_cf90_bpc_custom_ftsp_1N.json`
/// -> `last_minute_cf90_bpc_custom_ftsp`).
pub fn extract_policy_name_from_filename(filename: &str) -> String {
    let mut stem = filename;
    if let Some(pos) = stem.rfind('.') {
        stem = &stem[..pos];
    }
    if let Some(stripped) = stem.strip_prefix("log_") {
        stem = stripped;
    }
    // Strip trailing _<digits>N (e.g. _1N, _10N)
    if let Some(idx) = stem.rfind('_') {
        let suffix = &stem[idx + 1..];
        if suffix.ends_with('N') && suffix[..suffix.len() - 1].chars().all(|c| c.is_ascii_digit()) {
            stem = &stem[..idx];
        }
    }
    if stem.is_empty() {
        "simulation_policy".to_string()
    } else {
        stem.to_string()
    }
}

/// Extract sample instance count N from filename (e.g. `log_alns_10N.json` -> 10).
pub fn extract_instance_count_from_filename(filename: &str) -> Option<i32> {
    let mut stem = filename;
    if let Some(pos) = stem.rfind('.') {
        stem = &stem[..pos];
    }
    if let Some(idx) = stem.rfind('_') {
        let suffix = &stem[idx + 1..];
        if let Some(num_str) = suffix.strip_suffix('N') {
            if let Ok(n) = num_str.parse::<i32>() {
                return Some(n);
            }
        }
    }
    None
}

/// Check if a path corresponds to a simulation log file.
pub fn is_simulation_log_file(path: &Path) -> bool {
    let filename = match path.file_name().and_then(|f| f.to_str()) {
        Some(f) => f.to_lowercase(),
        None => return false,
    };
    if filename.starts_with('.') {
        return false;
    }
    if filename.ends_with(".jsonl") {
        return true;
    }
    if filename.ends_with(".json") && (filename.starts_with("log_") || filename.contains("_n.json")) {
        return true;
    }
    false
}
