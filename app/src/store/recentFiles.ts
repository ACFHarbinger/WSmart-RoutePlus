import { create } from "zustand";
import { persist } from "zustand/middleware";
import { portfolioRunLabel } from "../utils/duckdb/arrowPipeline";

export type RecentFileKind = "log" | "run" | "csv" | "training" | "checkpoint" | "config";

export interface RecentFile {
  path: string;
  label: string;
  kind: RecentFileKind;
  openedAt: number;
}

interface RecentFilesState {
  files: RecentFile[];
  pushRecent: (file: Omit<RecentFile, "openedAt">) => void;
  refreshRecentLabels: (projectRoot?: string | null) => void;
  removeRecent: (path: string) => void;
  clearRecent: () => void;
}

const MAX_RECENT = 12;

export const useRecentFilesStore = create<RecentFilesState>()(
  persist(
    (set) => ({
      files: [],
      pushRecent: (file) =>
        set((s) => {
          const openedAt = Date.now();
          const next = [
            { ...file, openedAt },
            ...s.files.filter((f) => f.path !== file.path),
          ].slice(0, MAX_RECENT);
          return { files: next };
        }),
      refreshRecentLabels: (projectRoot) =>
        set((s) => {
          if (s.files.length === 0) return s;
          const files = s.files.map((f) => ({
            ...f,
            label: portfolioRunLabel(f.path, f.label, projectRoot),
          }));
          const changed = files.some((f, i) => f.label !== s.files[i]!.label);
          return changed ? { files } : s;
        }),
      removeRecent: (path) =>
        set((s) => ({ files: s.files.filter((f) => f.path !== path) })),
      clearRecent: () => set({ files: [] }),
    }),
    { name: "wsmart-studio-recent-files" }
  )
);

export function recentFileLabel(path: string): string {
  return path.split("/").pop() ?? path;
}

/**
 * Classify a filesystem path into a recent-file kind for drop/open handoff
 * (§G.7 / §G.8 / §G.14 / §G.17 / §D.7).
 *
 * File extensions take priority. Bare directories use path heuristics for Lightning
 * training runs under ``logs/`` and simulation output runs under ``outputs/simulations`` (``assets/output`` before 2026-10-13).
 */
export function recentKindFromPath(path: string): RecentFileKind | null {
  const normalized = path.replace(/\\/g, "/");
  const base = (normalized.split("/").pop() ?? path).toLowerCase();
  if (base.endsWith(".jsonl") || base.endsWith(".log")) return "log";
  if (base.endsWith(".csv")) return "csv";
  if (/\.(pt|ckpt|pth)$/.test(base)) return "checkpoint";
  if (/\.(ya?ml|toml|cfg|ini)$/.test(base)) return "config";
  if (base.endsWith(".wsroute")) return null;

  // Directory heuristics (no recognised file extension)
  const hasFileExt = /\.[a-z0-9]{1,8}$/i.test(base);
  if (!hasFileExt) {
    if (/(^|\/)logs(\/|$)/.test(normalized) && !normalized.endsWith("/logs")) {
      return "training";
    }
    if (/(^|\/)assets\/output(\/|$)/.test(normalized)) {
      return "run";
    }
  }
  return null;
}
