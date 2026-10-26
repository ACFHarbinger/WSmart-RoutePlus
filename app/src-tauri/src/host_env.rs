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
    silence_appindicator_deprecation();

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

unsafe extern "C" fn noop_glib_log_handler(
    _log_domain: *const std::ffi::c_char,
    _log_level: i32,
    _message: *const std::ffi::c_char,
    _user_data: *mut std::ffi::c_void,
) {
}

/// Suppress the runtime deprecation warning emitted by `libayatana-appindicator`:
/// "libayatana-appindicator is deprecated. Please use libayatana-appindicator-glib in newly written code."
pub fn silence_appindicator_deprecation() {
    type GLogFunc = unsafe extern "C" fn(
        *const std::ffi::c_char,
        i32,
        *const std::ffi::c_char,
        *mut std::ffi::c_void,
    );
    type GLogSetHandler = unsafe extern "C" fn(
        *const std::ffi::c_char,
        i32,
        GLogFunc,
        *mut std::ffi::c_void,
    ) -> u32;

    unsafe {
        if let Ok(glib) = libloading::Library::new("libglib-2.0.so.0") {
            if let Ok(set_handler) = glib.get::<GLogSetHandler>(b"g_log_set_handler\0") {
                let domain = b"libayatana-appindicator\0";
                // 0xFF covers all log levels including G_LOG_LEVEL_WARNING (16)
                let log_levels = 0xFF;
                set_handler(
                    domain.as_ptr().cast(),
                    log_levels,
                    noop_glib_log_handler,
                    std::ptr::null_mut(),
                );
                std::mem::forget(glib);
            }
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
