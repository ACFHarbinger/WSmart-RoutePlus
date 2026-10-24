import { create } from "zustand";
import { policyVizDataLen } from "../utils/benchmark/policyTelemetry";
import type { DayLogEntry, PolicyVizEntry, SimBatchUpdate, SimFailureEntry } from "../types";

function dayKey(e: DayLogEntry): string {
  return `${e.policy}::${e.sample_id}::${e.day}`;
}

function vizKey(e: PolicyVizEntry): string {
  return `${e.policy}::${e.sample_id}::${e.day}::${e.policy_type}`;
}

function failureKey(e: SimFailureEntry): string {
  return `${e.policy}::${e.sample_id}::${e.day}`;
}

interface SimState {
  // All entries from the current log file
  entries: DayLogEntry[];
  policyVizEntries: PolicyVizEntry[];
  failureEntries: SimFailureEntry[];
  // O(1) index maps for deduplication
  dayIndexByKey: Record<string, number>;
  vizIndexByKey: Record<string, number>;
  failureIndexByKey: Record<string, number>;
  // Timestamp of most recent event update
  lastEventAt: number | null;
  // Selected filters (mirrors Streamlit sidebar controls)
  selectedPolicy: string | null;
  selectedSample: number | null;
  selectedDay: number | null;
  // Active log file being watched
  watchPath: string | null;
  isWatching: boolean;

  addEntry: (entry: DayLogEntry) => void;
  addEntries: (entries: DayLogEntry[]) => void;
  addPolicyVizEntry: (entry: PolicyVizEntry) => void;
  addPolicyVizEntries: (entries: PolicyVizEntry[]) => void;
  addFailureEntry: (entry: SimFailureEntry) => void;
  addFailureEntries: (entries: SimFailureEntry[]) => void;
  applyBatch: (batch: SimBatchUpdate) => void;
  loadEntries: (entries: DayLogEntry[]) => void;
  loadPolicyVizEntries: (entries: PolicyVizEntry[]) => void;
  loadFailureEntries: (entries: SimFailureEntry[]) => void;
  setSelectedPolicy: (policy: string | null) => void;
  setSelectedSample: (sample: number | null) => void;
  setSelectedDay: (day: number | null) => void;
  setWatchPath: (path: string | null) => void;
  setWatching: (watching: boolean) => void;
  reset: () => void;
}

export const useSimStore = create<SimState>((set, get) => ({
  entries: [],
  policyVizEntries: [],
  failureEntries: [],
  dayIndexByKey: {},
  vizIndexByKey: {},
  failureIndexByKey: {},
  lastEventAt: null,
  selectedPolicy: null,
  selectedSample: null,
  selectedDay: null,
  watchPath: null,
  isWatching: false,

  addEntries: (incoming) => {
    if (incoming.length === 0) return;
    set((s) => {
      let nextEntries: DayLogEntry[] | null = null;
      let nextIndex: Record<string, number> | null = null;
      for (const entry of incoming) {
        const k = dayKey(entry);
        const idxMap = nextIndex ?? s.dayIndexByKey;
        if (idxMap[k] !== undefined) continue;
        if (!nextEntries) {
          nextEntries = [...s.entries];
          nextIndex = { ...s.dayIndexByKey };
        }
        nextIndex![k] = nextEntries.length;
        nextEntries.push(entry);
      }
      if (!nextEntries || !nextIndex) return s;
      return {
        entries: nextEntries,
        dayIndexByKey: nextIndex,
        lastEventAt: Date.now(),
      };
    });
  },

  addEntry: (entry) => {
    get().addEntries([entry]);
  },

  addPolicyVizEntries: (incoming) => {
    if (incoming.length === 0) return;
    set((s) => {
      let nextViz: PolicyVizEntry[] | null = null;
      let nextIndex: Record<string, number> | null = null;
      for (const entry of incoming) {
        const k = vizKey(entry);
        const idxMap = nextIndex ?? s.vizIndexByKey;
        const existingIdx = idxMap[k];
        if (existingIdx === undefined) {
          if (!nextViz) {
            nextViz = [...s.policyVizEntries];
            nextIndex = { ...s.vizIndexByKey };
          }
          nextIndex![k] = nextViz.length;
          nextViz.push(entry);
        } else {
          const arr = nextViz ?? s.policyVizEntries;
          const existing = arr[existingIdx]!;
          if (policyVizDataLen(entry.data) >= policyVizDataLen(existing.data)) {
            if (!nextViz) {
              nextViz = [...s.policyVizEntries];
              nextIndex = { ...s.vizIndexByKey };
            }
            nextViz[existingIdx] = entry;
          }
        }
      }
      if (!nextViz || !nextIndex) return s;
      return {
        policyVizEntries: nextViz,
        vizIndexByKey: nextIndex,
        lastEventAt: Date.now(),
      };
    });
  },

  addPolicyVizEntry: (entry) => {
    get().addPolicyVizEntries([entry]);
  },

  addFailureEntries: (incoming) => {
    if (incoming.length === 0) return;
    set((s) => {
      let nextFailures: SimFailureEntry[] | null = null;
      let nextIndex: Record<string, number> | null = null;
      for (const entry of incoming) {
        const k = failureKey(entry);
        const idxMap = nextIndex ?? s.failureIndexByKey;
        if (idxMap[k] !== undefined) continue;
        if (!nextFailures) {
          nextFailures = [...s.failureEntries];
          nextIndex = { ...s.failureIndexByKey };
        }
        nextIndex![k] = nextFailures.length;
        nextFailures.push(entry);
      }
      if (!nextFailures || !nextIndex) return s;
      return {
        failureEntries: nextFailures,
        failureIndexByKey: nextIndex,
        lastEventAt: Date.now(),
      };
    });
  },

  addFailureEntry: (entry) => {
    get().addFailureEntries([entry]);
  },

  applyBatch: (batch) => {
    const state = get();
    if (batch.day_entries.length > 0) state.addEntries(batch.day_entries);
    if (batch.viz_entries.length > 0) state.addPolicyVizEntries(batch.viz_entries);
    if (batch.failure_entries.length > 0) state.addFailureEntries(batch.failure_entries);
  },

  loadEntries: (entries) => {
    const dayIndexByKey: Record<string, number> = {};
    entries.forEach((e, i) => {
      dayIndexByKey[dayKey(e)] = i;
    });
    set({ entries, dayIndexByKey, lastEventAt: Date.now() });
  },

  loadPolicyVizEntries: (entries) => {
    const vizIndexByKey: Record<string, number> = {};
    entries.forEach((e, i) => {
      vizIndexByKey[vizKey(e)] = i;
    });
    set({ policyVizEntries: entries, vizIndexByKey, lastEventAt: Date.now() });
  },

  loadFailureEntries: (entries) => {
    const failureIndexByKey: Record<string, number> = {};
    entries.forEach((e, i) => {
      failureIndexByKey[failureKey(e)] = i;
    });
    set({ failureEntries: entries, failureIndexByKey, lastEventAt: Date.now() });
  },

  setSelectedPolicy: (selectedPolicy) => set({ selectedPolicy }),
  setSelectedSample: (selectedSample) => set({ selectedSample }),
  setSelectedDay: (selectedDay) => set({ selectedDay }),
  setWatchPath: (watchPath) => set({ watchPath }),
  setWatching: (isWatching) => set({ isWatching }),

  reset: () =>
    set({
      entries: [],
      policyVizEntries: [],
      failureEntries: [],
      dayIndexByKey: {},
      vizIndexByKey: {},
      failureIndexByKey: {},
      lastEventAt: null,
      selectedPolicy: null,
      selectedSample: null,
      selectedDay: null,
      watchPath: null,
      isWatching: false,
    }),
}));

// Derived selectors
export function uniquePolicies(entries: DayLogEntry[]): string[] {
  return [...new Set(entries.map((e) => e.policy))].sort();
}

export function uniqueSamples(entries: DayLogEntry[]): number[] {
  return [...new Set(entries.map((e) => e.sample_id))].sort((a, b) => a - b);
}

export function filterEntries(
  entries: DayLogEntry[],
  policy: string | null,
  sampleId: number | null
): DayLogEntry[] {
  return entries.filter(
    (e) =>
      (policy === null || e.policy === policy) &&
      (sampleId === null || e.sample_id === sampleId)
  );
}
