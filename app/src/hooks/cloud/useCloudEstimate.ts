/**
 * Debounced pre-launch cost and ETA estimate of the Launcher's request (CR-10).
 */
import { useEffect, useState } from "react";
import type { CloudJobEstimate, CloudSubmitRequest } from "../../types/cloud";
import { getCloudCostEstimate } from "../../utils/cloud";

/**
 * Re-estimates `request` `delayMs` after it last changed; null while unavailable.
 *
 * `request` is compared by its JSON form, so callers can rebuild the object on
 * every render without re-running the estimate.
 */
export function useCloudEstimate(
  request: CloudSubmitRequest | null,
  options: { projectRoot?: string; root?: string },
  delayMs = 600
): CloudJobEstimate | null {
  const [estimate, setEstimate] = useState<CloudJobEstimate | null>(null);
  const key = request ? JSON.stringify(request) : "";
  const { projectRoot, root } = options;

  useEffect(() => {
    if (!key) {
      setEstimate(null);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      getCloudCostEstimate(JSON.parse(key) as CloudSubmitRequest, { projectRoot, root })
        .then((result) => {
          if (!cancelled) setEstimate(result);
        })
        .catch(() => {
          if (!cancelled) setEstimate(null);
        });
    }, delayMs);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [key, projectRoot, root, delayMs]);

  return estimate;
}
