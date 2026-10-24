import { useEffect, useRef, useState } from "react";

/**
 * Throttles rapid upstream store updates to a bounded UI frame rate (default 2 Hz / 500 ms)
 * and suspends updates when `document.visibilityState === "hidden"`.
 */
export function useSimThrottle<T>(value: T, intervalMs = 500): T {
  const [throttled, setThrottled] = useState<T>(value);
  const latestRef = useRef<T>(value);
  const lastUpdateRef = useRef<number>(Date.now());
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    latestRef.current = value;

    if (typeof document !== "undefined" && document.visibilityState === "hidden") {
      return;
    }

    const now = Date.now();
    const elapsed = now - lastUpdateRef.current;

    if (elapsed >= intervalMs) {
      lastUpdateRef.current = now;
      setThrottled(value);
      return;
    }

    if (!timerRef.current) {
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        if (typeof document === "undefined" || document.visibilityState !== "hidden") {
          lastUpdateRef.current = Date.now();
          setThrottled(latestRef.current);
        }
      }, intervalMs - elapsed);
    }
  }, [value, intervalMs]);

  useEffect(() => {
    if (typeof document === "undefined") return;
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        lastUpdateRef.current = Date.now();
        setThrottled(latestRef.current);
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, []);

  return throttled;
}
