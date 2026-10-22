/// Policy registry commands — load registered simulator policy names from Hydra config.
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Default policies used when `test_sim.yaml` cannot be parsed (mirrors controller defaults).
const FALLBACK_POLICIES: &[&str] = &[
    "aco_hh", "alns", "bpc", "hgs", "pg_clns", "psoma", "sans", "swc_tcf",
];

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SimPolicyEntry {
    pub id: String,
    pub config_key: String,
}

/// Parse `logic/configs/tasks/test_sim.yaml` and extract policy IDs from lines like:
/// `- /policies/routing/hyper_heuristics@p.aco_hh: policy_aco_hh`
#[tauri::command]
pub fn list_sim_policies(project_root: String) -> Result<Vec<SimPolicyEntry>, String> {
    let yaml_path = Path::new(&project_root).join("logic/configs/tasks/test_sim.yaml");
    if !yaml_path.exists() {
        return Ok(fallback_entries());
    }

    let content = std::fs::read_to_string(&yaml_path).map_err(|e| e.to_string())?;
    let mut policies: Vec<SimPolicyEntry> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        // e.g. "- /policies/routing/hyper_heuristics@p.aco_hh: policy_aco_hh"
        // (the config group is the policy's category folder; older files use "- /policies@p.").
        if !trimmed.starts_with("- /policies") {
            continue;
        }
        let Some(at_idx) = trimmed.find("@p.") else {
            continue;
        };
        let after_prefix = &trimmed[at_idx + "@p.".len()..];
        let colon_idx = after_prefix.find(':').ok_or_else(|| {
            format!("Malformed policy line in test_sim.yaml: {trimmed}")
        })?;
        let id = after_prefix[..colon_idx].trim().to_string();
        let config_key = after_prefix[colon_idx + 1..].trim().to_string();
        if !id.is_empty() {
            policies.push(SimPolicyEntry { id, config_key });
        }
    }

    if policies.is_empty() {
        return Ok(fallback_entries());
    }

    policies.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(policies)
}

fn fallback_entries() -> Vec<SimPolicyEntry> {
    FALLBACK_POLICIES
        .iter()
        .map(|id| SimPolicyEntry {
            id: (*id).to_string(),
            config_key: format!("policy_{id}"),
        })
        .collect()
}
