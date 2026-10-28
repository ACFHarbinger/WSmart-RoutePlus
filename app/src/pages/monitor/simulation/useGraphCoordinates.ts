import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DayLogEntry } from "../../../types";
import {
  enrichEntriesWithGraphCoords,
  GRAPH_PRESETS,
  guessGraphPreset,
  loadGraphCoordinates,
} from "../../../utils/graph/graphCoords";
import { toast } from "sonner";

export function useGraphCoordinates({
  activeLogPath,
  entries,
  projectRoot,
  selectedSample,
  loadEntries,
  displayEntry,
  setRouteMapMode,
}: {
  activeLogPath: string | null;
  entries: DayLogEntry[];
  projectRoot: string;
  selectedSample: number | null;
  loadEntries: (entries: DayLogEntry[]) => void;
  displayEntry: DayLogEntry | null;
  setRouteMapMode: (mode: "echarts" | "deckgl") => void;
}) {
  const [graphPreset, setGraphPreset] = useState(GRAPH_PRESETS[0].id);
  const [loadingGraphCoords, setLoadingGraphCoords] = useState(false);
  const lastEnrichedPathRef = useRef<string | null>(null);

  useEffect(() => {
    if (!activeLogPath || !entries.length) return;
    const guessed = guessGraphPreset(activeLogPath, entries);
    if (guessed) setGraphPreset(guessed);
  }, [activeLogPath, entries]);

  // Attempt auto-enrich if projectRoot is set and coordinates not yet loaded
  useEffect(() => {
    if (!projectRoot || !entries.length || !activeLogPath) return;
    if (lastEnrichedPathRef.current === activeLogPath) return;

    const day1 = entries.find((e) => e.day === 1);
    const hasAnyCoords = day1?.data.all_bin_coords?.some((b) => b.lat != null && b.lng != null);
    if (hasAnyCoords) {
      lastEnrichedPathRef.current = activeLogPath;
      return;
    }

    const presetId = guessGraphPreset(activeLogPath, entries) ?? graphPreset;
    loadGraphCoordinates(projectRoot, presetId, selectedSample ?? 0)
      .then((table) => {
        loadEntries(enrichEntriesWithGraphCoords(entries, table));
        setRouteMapMode("deckgl");
        lastEnrichedPathRef.current = activeLogPath;
      })
      .catch((err) => {
        console.warn("Auto-enrich graph coordinates skipped:", err);
      });
  }, [projectRoot, activeLogPath, entries, graphPreset, selectedSample, loadEntries]);

  const applyGraphCoords = useCallback(async () => {
    if (!projectRoot) {
      toast.error("Set project root in Settings to load graph coordinates");
      return;
    }
    if (!entries.length) return;
    setLoadingGraphCoords(true);
    try {
      const sampleIndex = selectedSample ?? 0;
      const table = await loadGraphCoordinates(projectRoot, graphPreset, sampleIndex);
      loadEntries(enrichEntriesWithGraphCoords(entries, table));
      setRouteMapMode("deckgl");
      toast.success("Graph coordinates loaded", {
        description: GRAPH_PRESETS.find((p) => p.id === graphPreset)?.label,
      });
    } catch (err) {
      toast.error("Failed to load graph coordinates", { description: String(err) });
    } finally {
      setLoadingGraphCoords(false);
    }
  }, [projectRoot, entries, graphPreset, selectedSample, loadEntries]);

  const hasGeoCoords = useMemo(
    () => displayEntry?.data.all_bin_coords?.some((b) => b.lat != null && b.lng != null) ?? false,
    [displayEntry]
  );

  const hasBinCoords = useMemo(
    () => (displayEntry?.data.all_bin_coords?.length ?? 0) > 0,
    [displayEntry]
  );

  return {
    graphPreset,
    setGraphPreset,
    loadingGraphCoords,
    applyGraphCoords,
    hasGeoCoords,
    hasBinCoords,
  };
}
