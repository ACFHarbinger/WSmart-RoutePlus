//! Background mode: closing the main window hides it instead of quitting, so
//! simulations and training runs launched from the Studio keep running.
//!
//! A tray icon brings the window back or quits the app, and launching the
//! Studio again shows the hidden window (single-instance plugin, wired in
//! `lib.rs`). If the tray icon cannot be created, closing the window quits as
//! before, so the window is never hidden with no way back.

use std::sync::atomic::{AtomicBool, Ordering};

use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, Runtime, State, Window, WindowEvent};
use tauri_plugin_notification::NotificationExt;

/// Label of the Studio window (the default window in `tauri.conf.json`).
const MAIN_WINDOW: &str = "main";

/// Whether closing the main window hides it, and whether the user was told once.
pub struct CloseToBackground {
    enabled: AtomicBool,
    notified: AtomicBool,
}

impl Default for CloseToBackground {
    fn default() -> Self {
        Self {
            enabled: AtomicBool::new(true),
            notified: AtomicBool::new(false),
        }
    }
}

/// Show, restore and focus the main window.
pub fn show_main_window<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window(MAIN_WINDOW) {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// Create the tray icon with Show and Quit; on failure, closing the window quits.
pub fn setup_tray<R: Runtime>(app: &AppHandle<R>) {
    #[cfg(target_os = "linux")]
    crate::host_env::silence_appindicator_deprecation();

    #[cfg(target_os = "linux")]
    if !appindicator_available() {
        return disable_background(app, "libayatana-appindicator3 is not installed");
    }
    if let Err(err) = build_tray(app) {
        disable_background(app, &err.to_string());
    }
}

fn disable_background<R: Runtime>(app: &AppHandle<R>, reason: &str) {
    eprintln!("WSmart-Route Studio: no tray icon ({reason}); closing the window will quit");
    app.state::<CloseToBackground>()
        .enabled
        .store(false, Ordering::SeqCst);
}

/// Whether Linux has an AppIndicator library: without one the tray backend panics, which aborts the app.
#[cfg(target_os = "linux")]
fn appindicator_available() -> bool {
    ["libayatana-appindicator3.so.1", "libappindicator3.so.1"]
        .iter()
        // SAFETY: loading a system shared library; it is dropped (unloaded) right away.
        .any(|name| unsafe { libloading::Library::new(name) }.is_ok())
}

fn build_tray<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, "show", "Show WSmart-Route Studio", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit WSmart-Route Studio", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &quit])?;
    let mut tray = TrayIconBuilder::with_id("studio")
        .tooltip("WSmart-Route Studio")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => show_main_window(app),
            "quit" => app.exit(0),
            _ => {}
        })
        // Linux tray icons (AppIndicator) emit no click events; the menu covers them.
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main_window(tray.app_handle());
            }
        });
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(app)?;
    Ok(())
}

/// Hide the main window on close while background mode is on.
pub fn on_window_event<R: Runtime>(window: &Window<R>, event: &WindowEvent) {
    let WindowEvent::CloseRequested { api, .. } = event else {
        return;
    };
    if window.label() != MAIN_WINDOW {
        return;
    }
    let state = window.state::<CloseToBackground>();
    if !state.enabled.load(Ordering::SeqCst) {
        return;
    }
    api.prevent_close();
    let _ = window.hide();
    if !state.notified.swap(true, Ordering::SeqCst) {
        let _ = window
            .app_handle()
            .notification()
            .builder()
            .title("WSmart-Route Studio is still running")
            .body("Running jobs continue. Use the tray icon to reopen the window or quit.")
            .show();
    }
}

/// Turn background mode on or off (the Settings toggle). Refused while there is no tray icon.
#[tauri::command]
pub fn set_close_to_background(
    app: AppHandle,
    state: State<'_, CloseToBackground>,
    enabled: bool,
) -> bool {
    let has_tray = app.tray_by_id("studio").is_some();
    let effective = enabled && has_tray;
    state.enabled.store(effective, Ordering::SeqCst);
    effective
}
