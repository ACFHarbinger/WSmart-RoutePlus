import { useMemo } from "react";
import { create } from "zustand";

import type {
  CorpusFilters,
  CorpusHighlight,
  CorpusMetric,
  CorpusScanResult,
  CorpusView,
} from "../types/corpus";
import { scanSimulationCorpus } from "../utils/corpus/scanClient";
import { buildCorpusView, groupCorpusRuns } from "../utils/corpus/select";

export interface SimulationCorpusState {
  roots: string[];
  scan: CorpusScanResult | null;
  loading: boolean;
  error: string | null;
  /** Empty means every run that is not hidden. A scan fills this with every run id. */
  selectedRunIds: string[];
  hiddenRunIds: string[];
  metric: CorpusMetric;
  dayMin: number | null;
  dayMax: number | null;
  policies: string[];
  cities: string[];
  playbackDay: number | null;
  playbackPlaying: boolean;
  baselineMode: "run" | "value";
  pinnedRunId: string | null;
  typedBaseline: number | null;
  highlight: CorpusHighlight | null;
  addRoots: (paths: string[]) => Promise<void>;
  reload: () => Promise<void>;
  hideRun: (id: string) => void;
  setMetric: (metric: CorpusMetric) => void;
  setDayRange: (dayMin: number | null, dayMax: number | null) => void;
  setPolicies: (policies: string[]) => void;
  setCities: (cities: string[]) => void;
  setPlaybackDay: (playbackDay: number | null) => void;
  setPlaybackPlaying: (playbackPlaying: boolean) => void;
  setBaselineMode: (baselineMode: "run" | "value") => void;
  setPinnedRunId: (pinnedRunId: string | null) => void;
  setTypedBaseline: (typedBaseline: number | null) => void;
  setHighlight: (highlight: CorpusHighlight | null) => void;
  setSelectedRunIds: (selectedRunIds: string[]) => void;
}

function rootKey(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/+$/, "");
}

function uniqueRoots(current: readonly string[], paths: readonly string[]): string[] {
  const next = [...current];
  for (const path of paths) {
    if (!path || next.some((existing) => rootKey(existing) === rootKey(path))) continue;
    next.push(path);
  }
  return next;
}

function errorMessage(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  if (typeof err === "string" && err) return err;
  return "Corpus scan failed";
}

function corpusFilters(state: Pick<SimulationCorpusState, keyof CorpusFilters>): CorpusFilters {
  return {
    selectedRunIds: state.selectedRunIds,
    hiddenRunIds: state.hiddenRunIds,
    metric: state.metric,
    dayMin: state.dayMin,
    dayMax: state.dayMax,
    policies: state.policies,
    cities: state.cities,
    playbackDay: state.playbackDay,
    baselineMode: state.baselineMode,
    pinnedRunId: state.pinnedRunId,
    typedBaseline: state.typedBaseline,
  };
}

let scanTicket = 0;

export const useSimulationCorpusStore = create<SimulationCorpusState>((set, get) => {
  async function rescan(): Promise<void> {
    const ticket = ++scanTicket;
    const roots = get().roots;
    set({ loading: true, error: null });
    try {
      const scan = await scanSimulationCorpus(roots);
      if (ticket !== scanTicket) return;
      const runs = groupCorpusRuns(scan.files);
      const ids = new Set(runs.map((run) => run.id));
      set({
        scan,
        loading: false,
        error: null,
        selectedRunIds: runs.map((run) => run.id),
        hiddenRunIds: get().hiddenRunIds.filter((id) => ids.has(id)),
      });
    } catch (err) {
      if (ticket !== scanTicket) return;
      set({ loading: false, error: errorMessage(err) });
    }
  }

  return {
    roots: [],
    scan: null,
    loading: false,
    error: null,
    selectedRunIds: [],
    hiddenRunIds: [],
    metric: "profit",
    dayMin: null,
    dayMax: null,
    policies: [],
    cities: [],
    playbackDay: null,
    playbackPlaying: false,
    baselineMode: "run",
    pinnedRunId: null,
    typedBaseline: null,
    highlight: null,
    addRoots: async (paths) => {
      set({ roots: uniqueRoots(get().roots, paths) });
      await rescan();
    },
    reload: rescan,
    hideRun: (id) =>
      set((state) => ({
        hiddenRunIds: state.hiddenRunIds.includes(id)
          ? state.hiddenRunIds
          : [...state.hiddenRunIds, id],
        selectedRunIds: state.selectedRunIds.filter((runId) => runId !== id),
        pinnedRunId: state.pinnedRunId === id ? null : state.pinnedRunId,
        highlight: state.highlight?.runId === id ? null : state.highlight,
      })),
    setMetric: (metric) => set({ metric }),
    setDayRange: (dayMin, dayMax) => set({ dayMin, dayMax }),
    setPolicies: (policies) => set({ policies }),
    setCities: (cities) => set({ cities }),
    setPlaybackDay: (playbackDay) => set({ playbackDay }),
    setPlaybackPlaying: (playbackPlaying) => set({ playbackPlaying }),
    setBaselineMode: (baselineMode) => set({ baselineMode }),
    setPinnedRunId: (pinnedRunId) => set({ pinnedRunId }),
    setTypedBaseline: (typedBaseline) => set({ typedBaseline }),
    setHighlight: (highlight) => set({ highlight }),
    setSelectedRunIds: (selectedRunIds) => set({ selectedRunIds }),
  };
});

/** Scene snapshot. `view` stays null until a scan exists. `playbackDay` also selects the route map day. */
export function useCorpusSceneState(): {
  view: CorpusView | null;
  metric: CorpusMetric;
  highlight: CorpusHighlight | null;
  setHighlight: (next: CorpusHighlight | null) => void;
} {
  const scan = useSimulationCorpusStore((state) => state.scan);
  const metric = useSimulationCorpusStore((state) => state.metric);
  const highlight = useSimulationCorpusStore((state) => state.highlight);
  const setHighlight = useSimulationCorpusStore((state) => state.setHighlight);
  const selectedRunIds = useSimulationCorpusStore((state) => state.selectedRunIds);
  const hiddenRunIds = useSimulationCorpusStore((state) => state.hiddenRunIds);
  const dayMin = useSimulationCorpusStore((state) => state.dayMin);
  const dayMax = useSimulationCorpusStore((state) => state.dayMax);
  const policies = useSimulationCorpusStore((state) => state.policies);
  const cities = useSimulationCorpusStore((state) => state.cities);
  const playbackDay = useSimulationCorpusStore((state) => state.playbackDay);
  const baselineMode = useSimulationCorpusStore((state) => state.baselineMode);
  const pinnedRunId = useSimulationCorpusStore((state) => state.pinnedRunId);
  const typedBaseline = useSimulationCorpusStore((state) => state.typedBaseline);

  const view = useMemo(() => {
    if (!scan) return null;
    return buildCorpusView(
      scan,
      corpusFilters({
        selectedRunIds,
        hiddenRunIds,
        metric,
        dayMin,
        dayMax,
        policies,
        cities,
        playbackDay,
        baselineMode,
        pinnedRunId,
        typedBaseline,
      })
    );
  }, [
    scan,
    metric,
    selectedRunIds,
    hiddenRunIds,
    dayMin,
    dayMax,
    policies,
    cities,
    playbackDay,
    baselineMode,
    pinnedRunId,
    typedBaseline,
  ]);

  return useMemo(
    () => ({ view, metric, highlight, setHighlight }),
    [view, metric, highlight, setHighlight]
  );
}
