import { useEffect, useState } from "react";

type Status = "loading" | "ready" | "error";

export interface JsonState<T> {
  data: T | null;
  status: Status;
  error: string | null;
}

/**
 * Fetch a JSON document from /data/ with loading and error state. Data is
 * fetched lazily on mount; callers can gate the request with `enabled`.
 */
export function useJson<T>(path: string, enabled = true): JsonState<T> {
  const [state, setState] = useState<JsonState<T>>({
    data: null,
    status: "loading",
    error: null,
  });

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setState({ data: null, status: "loading", error: null });
    fetch(path)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
        return res.json() as Promise<T>;
      })
      .then((data) => {
        if (!cancelled) setState({ data, status: "ready", error: null });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setState({
            data: null,
            status: "error",
            error: err instanceof Error ? err.message : String(err),
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [path, enabled]);

  return state;
}

export function formatNumber(value: number, decimals = 0): string {
  return value.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}
