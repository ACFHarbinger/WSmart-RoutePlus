/**
 * Settings card for background mode: whether the window's X hides the Studio
 * (jobs keep running, tray icon to reopen or quit) or quits it.
 */
import { useEffect, useState } from "react";
import { useAppStore } from "../../store/app";
import { syncCloseToBackground } from "../../utils/app/backgroundMode";

export function BackgroundModeCard() {
  const closeToBackground = useAppStore((s) => s.closeToBackground);
  const setCloseToBackground = useAppStore((s) => s.setCloseToBackground);
  // null until the backend answers; false when it has no tray icon
  const [effective, setEffective] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    void syncCloseToBackground(closeToBackground).then((value) => {
      if (!cancelled) setEffective(value);
    });
    return () => {
      cancelled = true;
    };
  }, [closeToBackground]);

  const unavailable = closeToBackground && effective === false;

  return (
    <div className="card space-y-3">
      <h2 className="text-sm font-semibold text-gray-200">Window</h2>
      <label className="flex items-start gap-3 cursor-pointer">
        <input
          type="checkbox"
          checked={closeToBackground}
          onChange={(e) => setCloseToBackground(e.target.checked)}
          className="mt-0.5"
        />
        <span className="text-sm text-gray-300">
          Keep running in the background when the window is closed
          <span className="block text-xs text-canvas-muted">
            Running simulations and training continue; use the tray icon to reopen the window or
            quit. When off, closing the window quits the Studio.
          </span>
        </span>
      </label>
      {unavailable && (
        <p className="text-xs text-amber-400">
          No tray icon is available, so closing the window quits the Studio. On Linux, install
          libayatana-appindicator3 and restart the Studio.
        </p>
      )}
    </div>
  );
}
