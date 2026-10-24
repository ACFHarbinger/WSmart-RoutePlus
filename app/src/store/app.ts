import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AppMode, BenchmarkLogRef, EvalAnalyticsRow, PendingMapCompare } from "../types";
import {
  applyDomTheme,
  resolveEffectiveTheme,
  type EffectiveTheme,
  type ThemePreference,
} from "../utils/charts/theme";
import { syncCloseToBackground } from "../utils/app/backgroundMode";

interface AppState {
  mode: AppMode;
  theme: ThemePreference;
  effectiveTheme: EffectiveTheme;
  reduceMotion: boolean;
  projectRoot: string;
  pythonPath: string;
  // Closing the window hides the Studio (tray icon) instead of quitting
  closeToBackground: boolean;
  // Ephemeral — set by TrainingMonitor checkpoint browser to pre-populate EvaluationRunner
  pendingCheckpoint: string | null;
  // Ephemeral — set by OutputBrowser to auto-load a log file in SimulationSummary
  pendingLogPath: string | null;
  // Ephemeral — set by EvaluationRunner to pre-load results in BenchmarkAnalysis
  pendingEvalResults: EvalAnalyticsRow[] | null;
  pendingBenchmarkLogs: BenchmarkLogRef[] | null;
  pendingRunPath: string | null;
  pendingCsvPath: string | null;
  pendingTrainingRunPath: string | null;
  pendingConfigPath: string | null;
  pendingMapCompare: PendingMapCompare | null;
  setMode: (mode: AppMode) => void;
  setTheme: (theme: ThemePreference) => void;
  setEffectiveTheme: (effective: EffectiveTheme) => void;
  setReduceMotion: (reduce: boolean) => void;
  setProjectRoot: (root: string) => void;
  setPythonPath: (path: string) => void;
  setCloseToBackground: (enabled: boolean) => void;
  setPendingCheckpoint: (path: string | null) => void;
  setPendingLogPath: (path: string | null) => void;
  setPendingEvalResults: (rows: EvalAnalyticsRow[] | null) => void;
  setPendingBenchmarkLogs: (logs: BenchmarkLogRef[] | null) => void;
  setPendingRunPath: (path: string | null) => void;
  setPendingCsvPath: (path: string | null) => void;
  setPendingTrainingRunPath: (path: string | null) => void;
  setPendingConfigPath: (path: string | null) => void;
  setPendingMapCompare: (compare: PendingMapCompare | null) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      mode: "simulation",
      theme: "dark",
      effectiveTheme: "dark",
      reduceMotion: false,
      projectRoot: "",
      pythonPath: "",
      closeToBackground: true,
      pendingCheckpoint: null,
      pendingLogPath: null,
      pendingEvalResults: null,
      pendingBenchmarkLogs: null,
      pendingRunPath: null,
      pendingCsvPath: null,
      pendingTrainingRunPath: null,
      pendingConfigPath: null,
      pendingMapCompare: null,
      setMode: (mode) => set({ mode }),
      setTheme: (theme) => {
        const effective = resolveEffectiveTheme(theme);
        applyDomTheme(effective);
        set({ theme, effectiveTheme: effective });
      },
      setEffectiveTheme: (effectiveTheme) => set({ effectiveTheme }),
      setReduceMotion: (reduceMotion) => {
        if (typeof document !== "undefined") {
          document.documentElement.classList.toggle("reduce-motion", reduceMotion);
        }
        set({ reduceMotion });
      },
      setProjectRoot: (projectRoot) => set({ projectRoot }),
      setPythonPath: (pythonPath) => set({ pythonPath }),
      setCloseToBackground: (closeToBackground) => {
        void syncCloseToBackground(closeToBackground);
        set({ closeToBackground });
      },
      setPendingCheckpoint: (pendingCheckpoint) => set({ pendingCheckpoint }),
      setPendingLogPath: (pendingLogPath) => set({ pendingLogPath }),
      setPendingEvalResults: (pendingEvalResults: EvalAnalyticsRow[] | null) =>
        set({ pendingEvalResults }),
      setPendingBenchmarkLogs: (pendingBenchmarkLogs: BenchmarkLogRef[] | null) =>
        set({ pendingBenchmarkLogs }),
      setPendingRunPath: (pendingRunPath) => set({ pendingRunPath }),
      setPendingCsvPath: (pendingCsvPath) => set({ pendingCsvPath }),
      setPendingTrainingRunPath: (pendingTrainingRunPath) => set({ pendingTrainingRunPath }),
      setPendingConfigPath: (pendingConfigPath) => set({ pendingConfigPath }),
      setPendingMapCompare: (pendingMapCompare) => set({ pendingMapCompare }),
    }),
    {
      name: "wsmart-studio-app",
      partialize: (s) => ({
        mode: s.mode,
        theme: s.theme,
        reduceMotion: s.reduceMotion,
        projectRoot: s.projectRoot,
        pythonPath: s.pythonPath,
        closeToBackground: s.closeToBackground,
      }),
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        const effective = resolveEffectiveTheme(state.theme);
        applyDomTheme(effective);
        state.effectiveTheme = effective;
        if (typeof document !== "undefined") {
          document.documentElement.classList.toggle("reduce-motion", Boolean(state.reduceMotion));
        }
        // The backend starts with background mode on; apply a saved "off".
        if (state.closeToBackground === false) void syncCloseToBackground(false);
      },
    }
  )
);
