/**
 * Cloud Analysis Store — UI preferences and selected cloud job state for CR-16 (§J.1).
 * Persisted to localStorage via Zustand `persist`.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";

export type AnalysisTab = "overview" | "tasks" | "cost" | "waterfall";

export interface CloudAnalysisState {
  selectedJobId: string | null;
  activeTab: AnalysisTab;
  taskFilter: string;
  searchQuery: string;
  sortBy: "task_id" | "cell_id" | "duration" | "state" | "cost";
  sortDirection: "asc" | "desc";
  root: string;

  setSelectedJobId: (id: string | null) => void;
  setActiveTab: (tab: AnalysisTab) => void;
  setTaskFilter: (filter: string) => void;
  setSearchQuery: (query: string) => void;
  setSort: (by: "task_id" | "cell_id" | "duration" | "state" | "cost", dir?: "asc" | "desc") => void;
  setRoot: (root: string) => void;
  reset: () => void;
}

const DEFAULT_STATE = {
  selectedJobId: null,
  activeTab: "overview" as AnalysisTab,
  taskFilter: "all",
  searchQuery: "",
  sortBy: "task_id" as const,
  sortDirection: "asc" as const,
  root: "",
};

export const useCloudAnalysisStore = create<CloudAnalysisState>()(
  persist(
    (set) => ({
      ...DEFAULT_STATE,

      setSelectedJobId: (selectedJobId) => set({ selectedJobId }),
      setActiveTab: (activeTab) => set({ activeTab }),
      setTaskFilter: (taskFilter) => set({ taskFilter }),
      setSearchQuery: (searchQuery) => set({ searchQuery }),
      setSort: (sortBy, dir) =>
        set((state) => ({
          sortBy,
          sortDirection: dir ?? (state.sortBy === sortBy && state.sortDirection === "asc" ? "desc" : "asc"),
        })),
      setRoot: (root) => set({ root }),
      reset: () => set(DEFAULT_STATE),
    }),
    {
      name: "wsroute-cloud-analysis",
    }
  )
);
