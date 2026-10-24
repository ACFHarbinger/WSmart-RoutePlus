/**
 * Background mode: whether closing the Studio window hides it instead of quitting.
 */
import { invoke } from "@tauri-apps/api/core";

/**
 * Tell the Tauri backend whether the window's X hides the Studio.
 *
 * Resolves to the mode actually in effect: false when the backend has no tray
 * icon (closing then quits) or when there is no Tauri backend at all.
 */
export async function syncCloseToBackground(enabled: boolean): Promise<boolean> {
  try {
    return await invoke<boolean>("set_close_to_background", { enabled });
  } catch {
    return false;
  }
}
