import { create } from "zustand";
import type { DayLogEntry } from "../types";

export type LoadedRunSource = "viewing" | "live";

export interface LoadedRunPayload {
  path: string;
  label: string;
  area?: string;
  numLoc?: number;
  distribution?: string;
  days?: number;
  entries?: DayLogEntry[];
  source?: LoadedRunSource;
}

export interface LoadedRunState {
  path: string | null;
  label: string | null;
  area: string | null;
  numLoc: number | null;
  distribution: string | null;
  days: number | null;
  entries: DayLogEntry[];
  source: LoadedRunSource;
  loadedAt: number | null;
  setLoadedRun: (payload: LoadedRunPayload) => void;
  setSource: (source: LoadedRunSource) => void;
  toggleSource: () => void;
  clearLoadedRun: () => void;
}

/**
 * Parse simulation path metadata from standard WSmart-Route output folders, e.g.
 * `outputs/simulations/riomaior_204_lognormal/2026-10-24_12-00-00`
 */
export function parseRunMetadataFromPath(path: string): {
  label: string;
  area?: string;
  numLoc?: number;
  distribution?: string;
} {
  const clean = path.replace(/\\/g, "/").replace(/\/+$/, "");
  const parts = clean.split("/").filter(Boolean);
  const last = parts[parts.length - 1] ?? clean;
  const parent = parts[parts.length - 2] ?? "";

  const match =
    parent.match(/^([a-zA-Z0-9]+)_(\d+)_([a-zA-Z0-9]+)$/) ??
    last.match(/^([a-zA-Z0-9]+)_(\d+)_([a-zA-Z0-9]+)$/);

  if (match) {
    return {
      label: `${match[1]}_${match[2]}_${match[3]}`,
      area: match[1],
      numLoc: Number(match[2]),
      distribution: match[3],
    };
  }

  return { label: last || "loaded_run" };
}

export const useLoadedRunStore = create<LoadedRunState>((set) => ({
  path: null,
  label: null,
  area: null,
  numLoc: null,
  distribution: null,
  days: null,
  entries: [],
  source: "viewing",
  loadedAt: null,

  setLoadedRun: (payload) => {
    const inferred = parseRunMetadataFromPath(payload.path);
    const entries = payload.entries ?? [];
    const maxDay =
      entries.length > 0
        ? Math.max(...entries.map((e) => e.day ?? 0))
        : undefined;

    set({
      path: payload.path,
      label: payload.label || inferred.label,
      area: payload.area ?? inferred.area ?? null,
      numLoc: payload.numLoc ?? inferred.numLoc ?? null,
      distribution: payload.distribution ?? inferred.distribution ?? null,
      days: payload.days ?? maxDay ?? null,
      entries,
      source: payload.source ?? "viewing",
      loadedAt: Date.now(),
    });
  },

  setSource: (source) => set({ source }),

  toggleSource: () =>
    set((state) => ({
      source: state.source === "viewing" ? "live" : "viewing",
    })),

  clearLoadedRun: () =>
    set({
      path: null,
      label: null,
      area: null,
      numLoc: null,
      distribution: null,
      days: null,
      entries: [],
      source: "viewing",
      loadedAt: null,
    }),
}));
