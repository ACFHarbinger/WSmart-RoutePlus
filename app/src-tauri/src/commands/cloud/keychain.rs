//! OS Keychain credential management for WSmart-Route Cloud Runner.
//!
//! # Credential Storage & Naming Scheme
//!
//! Credentials (such as cloud provider service account JSON keys, API tokens, or Gurobi licenses)
//! are stored securely in the operating system's native keychain using the `keyring` crate:
//!
//! - **Service Name:** `wsmart-route.cloud`
//! - **Account Name:** `{provider}/{profile}`
//!   (e.g., `gcp/default`, `aws/dev`, `azure/prod`)
//!
//! ## CR-08 (Python-side) Integration Contract
//!
//! Python-side lookup by profile name (CR-08 / Claude) reads the identical entry using the
//! standard `keyring` Python library:
//!
//! ```python
//! import keyring
//!
//! # Look up secret by provider and profile name:
//! secret = keyring.get_password("wsmart-route.cloud", f"{provider.lower()}/{profile}")
//! ```
//!
//! ## Security Invariants
//! - Secrets are **never** logged to stdout, stderr, or logfiles.
//! - Secrets are **never** returned to the Tauri frontend once stored.
//! - Secrets are **never** written to unencrypted disk storage outside the OS keychain.

pub const KEYCHAIN_SERVICE: &str = "wsmart-route.cloud";

/// Construct the canonical account identifier for a profile and provider.
pub fn make_account_key(provider: &str, profile: &str) -> String {
    format!("{}/{}", provider.trim().to_lowercase(), profile.trim())
}

/// Store a secret credential securely in the OS keychain.
pub fn store_credential(profile: &str, provider: &str, secret: &str) -> Result<(), String> {
    let p_clean = profile.trim();
    let prov_clean = provider.trim();

    if p_clean.is_empty() {
        return Err("Credential profile name cannot be empty".to_string());
    }
    if prov_clean.is_empty() {
        return Err("Cloud provider cannot be empty".to_string());
    }
    if secret.trim().is_empty() {
        return Err("Secret credential payload cannot be empty".to_string());
    }

    let account = make_account_key(prov_clean, p_clean);
    let entry = keyring::Entry::new(KEYCHAIN_SERVICE, &account)
        .map_err(|e| format!("Failed to initialize keyring entry for account '{account}': {e}"))?;

    entry
        .set_password(secret)
        .map_err(|e| format!("Failed to write credential to OS keychain: {e}"))?;

    Ok(())
}

/// Delete a credential from the OS keychain.
pub fn delete_credential(profile: &str, provider: &str) -> Result<(), String> {
    let p_clean = profile.trim();
    let prov_clean = provider.trim();

    if p_clean.is_empty() {
        return Err("Credential profile name cannot be empty".to_string());
    }
    if prov_clean.is_empty() {
        return Err("Cloud provider cannot be empty".to_string());
    }

    let account = make_account_key(prov_clean, p_clean);
    let entry = keyring::Entry::new(KEYCHAIN_SERVICE, &account)
        .map_err(|e| format!("Failed to initialize keyring entry for account '{account}': {e}"))?;

    match entry.delete_credential() {
        Ok(()) => Ok(()),
        Err(keyring::Error::NoEntry) => Ok(()), // Already absent, idempotent success
        Err(e) => Err(format!("Failed to delete credential from OS keychain: {e}")),
    }
}

/// Check if a credential exists in the OS keychain without exposing its secret.
pub fn has_credential(profile: &str, provider: &str) -> Result<bool, String> {
    let p_clean = profile.trim();
    let prov_clean = provider.trim();

    if p_clean.is_empty() || prov_clean.is_empty() {
        return Ok(false);
    }

    let account = make_account_key(prov_clean, p_clean);
    let entry = keyring::Entry::new(KEYCHAIN_SERVICE, &account)
        .map_err(|e| format!("Failed to initialize keyring entry for account '{account}': {e}"))?;

    match entry.get_password() {
        Ok(_) => Ok(true),
        Err(keyring::Error::NoEntry) => Ok(false),
        Err(e) => Err(format!("Failed to inspect OS keychain: {e}")),
    }
}
