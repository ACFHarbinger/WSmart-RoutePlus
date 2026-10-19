/**
 * Persisted state and UI preferences for the Cloud Monitor screen (CR-15 / §J.1).
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface CloudMonitorState {
  selectedJobId: string | null;
  stateFilter: string;
  searchQuery: string;
  autoRefresh: boolean;
  root: string;
  activeTab: "tasks" | "metrics" | "json";
  setSelectedJobId: (jobId: string | null) => void;
  setStateFilter: (filter: string) => void;
  setSearchQuery: (query: string) => void;
  setAutoRefresh: (auto: boolean) => void;
  setRoot: (root: string) => void;
  setActiveTab: (tab: "tasks" | "metrics" | "json") => void;
  patch: (updates: Partial<Omit<CloudMonitorState, "patch" | "setSelectedJobId" | "setStateFilter" | "setSearchQuery" | "setAutoRefresh" | "setRoot" | "setActiveTab">>) => void;
}

export const useCloudMonitorStore = create<CloudMonitorState>()(
  persist(
    (set) => ({
      selectedJobId: null,
      stateFilter: "all",
      searchQuery: "",
      autoRefresh: true,
      root: "outputs/cloud",
      activeTab: "tasks",
      setSelectedJobId: (selectedJobId) => set({ selectedJobId }),
      setStateFilter: (stateFilter) => set({ stateFilter }),
      setSearchQuery: (searchQuery) => set({ searchQuery }),
      setAutoRefresh: (autoRefresh) => set({ autoRefresh }),
      setRoot: (root) => set({ root }),
      setActiveTab: (activeTab) => set({ activeTab }),
      patch: (updates) => set(updates),
    }),
    { name: "wsroute-cloud-monitor" }
  )
);
