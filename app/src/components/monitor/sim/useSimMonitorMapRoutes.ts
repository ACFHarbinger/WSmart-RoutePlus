import { useEffect, useMemo } from "react";
import { hexToRgb } from "../../../utils/charts/colors";
import { POLICY_COLORS } from "./SimMonitorPanels";
import type { DayLogEntry, PendingMapCompare } from "../../../types";

export function useSimMonitorMapRoutes({
  entries,
  displayDay,
  selectedSample,
  activeMapPolicies,
  policies,
  pendingMapCompare,
  setPendingMapCompare,
  setShowRouteMap,
  setMapPolicies,
  setMapLayout,
  setRouteMapMode,
}: {
  entries: DayLogEntry[];
  displayDay: number;
  selectedSample: number | null;
  activeMapPolicies: string[];
  policies: string[];
  pendingMapCompare: PendingMapCompare | null;
  setPendingMapCompare: (p: PendingMapCompare | null) => void;
  setShowRouteMap: (show: boolean) => void;
  setMapPolicies: (p: string[]) => void;
  setMapLayout: (l: "overlay" | "split") => void;
  setRouteMapMode: (m: "echarts" | "deckgl") => void;
}) {
  useEffect(() => {
    if (!pendingMapCompare) return;
    setShowRouteMap(true);
    setMapPolicies(pendingMapCompare.policies);
    setMapLayout(pendingMapCompare.layout);
    if (pendingMapCompare.mapMode) setRouteMapMode(pendingMapCompare.mapMode);
    setPendingMapCompare(null);
  }, [pendingMapCompare, setPendingMapCompare, setShowRouteMap, setMapPolicies, setMapLayout, setRouteMapMode]);

  const mapRoutes = useMemo(() => {
    const dayEntries = entries.filter(
      (e) =>
        e.day === displayDay &&
        (selectedSample === null || e.sample_id === selectedSample) &&
        activeMapPolicies.includes(e.policy)
    );
    return dayEntries
      .filter((e) => e.data.all_bin_coords?.length)
      .map((e) => {
        const idx = policies.indexOf(e.policy);
        return {
          id: `${e.policy}-${e.sample_id}`,
          label: e.policy,
          data: e.data,
          color: hexToRgb(POLICY_COLORS[idx % POLICY_COLORS.length]),
        };
      });
  }, [entries, displayDay, selectedSample, activeMapPolicies, policies]);

  const mapHasFailureData = useMemo(
    () =>
      mapRoutes.some(
        (r) =>
          r.data.failure_analysis?.has_failure &&
          ((r.data.failure_analysis.overflow_bins?.length ?? 0) > 0 ||
            (r.data.failure_analysis.skipped_high_fill_bins?.length ?? 0) > 0)
      ),
    [mapRoutes]
  );

  return { mapRoutes, mapHasFailureData };
}
