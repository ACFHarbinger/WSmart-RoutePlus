//! Host environment cleanup before GTK and WebKit start.
//!
//! A desktop session or a Snap-packaged IDE exports `/snap/core20/...` on
//! `LD_LIBRARY_PATH`. WebKitGTK then starts `WebKitNetworkProcess` with that
//! path, the helper loads Snap's `libpthread`, and the load fails with
//! `__libc_pthread_init` / `GLIBC_PRIVATE`. The view reports
//! "WebKit encountered an internal error". The same environment makes Glycin
//! believe it is inside a Snap, and points GTK at Snap modules so
//! `colorreload-gtk-module` and `window-decorations-gtk-module` do not load.

use std::path::Path;

const PATH_VARS: &[&str] = &[
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LIBRARY_PATH",
    "GTK_PATH",
    "GTK_IM_MODULE_FILE",
    "GDK_PIXBUF_MODULE_FILE",
    "GDK_PIXBUF_MODULEDIR",
    "GIO_MODULE_DIR",
    "GIO_EXTRA_MODULES",
    "XDG_DATA_DIRS",
];

const SNAP_IDENTITY: &[&str] = &[
    "SNAP",
    "SNAP_ARCH",
    "SNAP_COMMON",
    "SNAP_CONTEXT",
    "SNAP_COOKIE",
    "SNAP_DATA",
    "SNAP_INSTANCE_NAME",
    "SNAP_LIBRARY_PATH",
    "SNAP_NAME",
    "SNAP_REEXEC",
    "SNAP_REVISION",
    "SNAP_USER_COMMON",
    "SNAP_USER_DATA",
    "SNAP_VERSION",
];

/// Drop empty entries and any entry that points into a Snap mount.
pub fn without_snap_paths(value: &str) -> Option<String> {
    let kept: Vec<&str> = value
        .split(':')
        .filter(|part| {
            let part = part.trim();
            !part.is_empty() && !part.contains("/snap/")
        })
        .collect();
    if kept.is_empty() {
        None
    } else {
        Some(kept.join(":"))
    }
}

fn scrub_path_var(key: &str) {
    let Ok(value) = std::env::var(key) else {
        return;
    };
    match without_snap_paths(&value) {
        Some(cleaned) if cleaned != value => std::env::set_var(key, cleaned),
        None => std::env::remove_var(key),
        Some(_) => {}
    }
}

fn running_from_snap() -> bool {
    std::env::current_exe()
        .ok()
        .is_some_and(|path| path.starts_with("/snap"))
}

/// Apply the cleanup. A binary that itself lives under `/snap` is left alone.
pub fn prepare() {
    if running_from_snap() {
        return;
    }
    for key in PATH_VARS {
        scrub_path_var(key);
    }
    if let Ok(prefix) = std::env::var("GTK_EXE_PREFIX") {
        if prefix.contains("/snap/") {
            std::env::remove_var("GTK_EXE_PREFIX");
        }
    }
    for key in SNAP_IDENTITY {
        std::env::remove_var(key);
    }
    // Host GTK modules are installed, but a Snap `GTK_PATH` hides them.
    // Re-check after the path scrub and drop names the host cannot open.
    if let Ok(modules) = std::env::var("GTK_MODULES") {
        if let Some(kept) = installed_gtk_modules(&modules) {
            if kept != modules {
                std::env::set_var("GTK_MODULES", kept);
            }
        } else {
            std::env::remove_var("GTK_MODULES");
        }
    }
}

fn installed_gtk_modules(value: &str) -> Option<String> {
    let kept: Vec<&str> = value
        .split(':')
        .map(str::trim)
        .filter(|name| !name.is_empty() && gtk3_module_present(name))
        .collect();
    if kept.is_empty() {
        None
    } else {
        Some(kept.join(":"))
    }
}

fn gtk3_module_present(name: &str) -> bool {
    let file = format!("lib{name}.so");
    let mut dirs = vec![
        Path::new("/usr/lib/x86_64-linux-gnu/gtk-3.0/modules").to_path_buf(),
        Path::new("/usr/lib/gtk-3.0/modules").to_path_buf(),
        Path::new("/usr/lib64/gtk-3.0/modules").to_path_buf(),
    ];
    if let Ok(gtk_path) = std::env::var("GTK_PATH") {
        for part in gtk_path.split(':') {
            if part.is_empty() || part.contains("/snap/") {
                continue;
            }
            dirs.push(Path::new(part).join("gtk-3.0/modules"));
            dirs.push(Path::new(part).to_path_buf());
        }
    }
    dirs.iter().any(|dir| dir.join(&file).is_file())
}

#[cfg(test)]
mod tests {
    use super::without_snap_paths;

    #[test]
    fn snap_library_entries_are_removed() {
        let cleaned = without_snap_paths(
            "/snap/core20/current/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/snap/code/current/usr/lib",
        );
        assert_eq!(cleaned.as_deref(), Some("/usr/lib/x86_64-linux-gnu"));
    }

    #[test]
    fn a_snap_only_path_is_cleared() {
        assert_eq!(
            without_snap_paths("/snap/core20/current/lib/x86_64-linux-gnu"),
            None
        );
    }

    #[test]
    fn host_paths_stay() {
        assert_eq!(
            without_snap_paths("/usr/local/lib:/usr/lib").as_deref(),
            Some("/usr/local/lib:/usr/lib")
        );
    }
}
