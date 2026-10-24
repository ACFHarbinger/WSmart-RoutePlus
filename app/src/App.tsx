import { lazy, Suspense, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Toaster } from "sonner";
import { Layout } from "./components/layout/Layout";
import { useHashSync } from "./hooks/app/useHashSync";
import { useProcessMonitor } from "./hooks/process/useProcessMonitor";
import { useThemeSync } from "./hooks/app/useThemeSync";
import { useAppStore } from "./store/app";
import { useRecentFilesStore } from "./store/recentFiles";
import { useLaunchTriggerStore } from "./store/launchTrigger";
import { useLayoutStore } from "./store/layout";
import { useDuckDbInit } from "./hooks/app/useDuckDbInit";
import { prefetchPage } from "./utils/app/pagePrefetch";
import { markStartup } from "./utils/app/startupTiming";
import type { AppMode } from "./types";

// Lazy-loaded pages (§G.7 performance)
const SimulationMonitor = lazy(() =>
  import("./pages/monitor/SimulationMonitor").then((m) => ({ default: m.SimulationMonitor }))
);
const TrainingMonitor = lazy(() =>
  import("./pages/monitor/TrainingMonitor").then((m) => ({ default: m.TrainingMonitor }))
);
const ProcessMonitor = lazy(() =>
  import("./pages/monitor/ProcessMonitor").then((m) => ({ default: m.ProcessMonitor }))
);
const CloudMonitor = lazy(() =>
  import("./pages/cloud/CloudMonitor").then((m) => ({ default: m.CloudMonitor }))
);
const CloudLauncher = lazy(() =>
  import("./pages/cloud/CloudLauncher").then((m) => ({ default: m.CloudLauncher }))
);
const CloudAnalysis = lazy(() =>
  import("./pages/cloud/CloudAnalysis").then((m) => ({ default: m.CloudAnalysis }))
);
const SimulationSummary = lazy(() =>
  import("./pages/analysis/SimulationSummary").then((m) => ({ default: m.SimulationSummary }))
);
const BenchmarkAnalysis = lazy(() =>
  import("./pages/analysis/BenchmarkAnalysis").then((m) => ({ default: m.BenchmarkAnalysis }))
);
const CityComparison = lazy(() =>
  import("./pages/analysis/CityComparison").then((m) => ({ default: m.CityComparison }))
);
const OlapExplorer = lazy(() =>
  import("./pages/analysis/OlapExplorer").then((m) => ({ default: m.OlapExplorer }))
);
const DataExplorer = lazy(() =>
  import("./pages/analysis/DataExplorer").then((m) => ({ default: m.DataExplorer }))
);
const ExperimentTracker = lazy(() =>
  import("./pages/analysis/ExperimentTracker").then((m) => ({ default: m.ExperimentTracker }))
);
const AlgorithmComparison = lazy(() =>
  import("./pages/analysis/AlgorithmComparison").then((m) => ({ default: m.AlgorithmComparison }))
);
const HPOTracker = lazy(() =>
  import("./pages/analysis/HPOTracker").then((m) => ({ default: m.HPOTracker }))
);
const SimulationLauncher = lazy(() =>
  import("./pages/launch/SimulationLauncher").then((m) => ({ default: m.SimulationLauncher }))
);
const TrainingHub = lazy(() =>
  import("./pages/launch/TrainingHub").then((m) => ({ default: m.TrainingHub }))
);
const DataGeneration = lazy(() =>
  import("./pages/launch/DataGeneration").then((m) => ({ default: m.DataGeneration }))
);
const EvaluationRunner = lazy(() =>
  import("./pages/launch/EvaluationRunner").then((m) => ({ default: m.EvaluationRunner }))
);
const ConfigEditor = lazy(() =>
  import("./pages/files/ConfigEditor").then((m) => ({ default: m.ConfigEditor }))
);
const OutputBrowser = lazy(() =>
  import("./pages/files/OutputBrowser").then((m) => ({ default: m.OutputBrowser }))
);
const ReportStudio = lazy(() =>
  import("./pages/launch/ReportStudio").then((m) => ({ default: m.ReportStudio }))
);
const SystemTools = lazy(() =>
  import("./pages/files/SystemTools").then((m) => ({ default: m.SystemTools }))
);
const Settings = lazy(() =>
  import("./pages/app/Settings").then((m) => ({ default: m.Settings }))
);

function PageFallback() {
  return (
    <div className="flex items-center justify-center h-64 gap-2 text-canvas-muted text-sm">
      <RefreshCw size={14} className="animate-spin" />
      Loading…
    </div>
  );
}

const PERSISTENT_MODES: readonly AppMode[] = [
  "simulation",
  "training",
  "simulation_summary",
  "benchmark",
  "olap_explorer",
  "data_explorer",
];

function renderPage(mode: AppMode) {
  switch (mode) {
    case "simulation":
      return <SimulationMonitor />;
    case "training":
      return <TrainingMonitor />;
    case "simulation_summary":
      return <SimulationSummary />;
    case "benchmark":
      return <BenchmarkAnalysis />;
    case "city_comparison":
      return <CityComparison />;
    case "olap_explorer":
      return <OlapExplorer />;
    case "data_explorer":
      return <DataExplorer />;
    case "experiment_tracker":
      return <ExperimentTracker />;
    case "algorithms":
      return <AlgorithmComparison />;
    case "hpo_tracker":
      return <HPOTracker />;
    case "process_monitor":
      return <ProcessMonitor />;
    case "cloud_monitor":
      return <CloudMonitor />;
    case "cloud_launcher":
      return <CloudLauncher />;
    case "cloud_analysis":
      return <CloudAnalysis />;
    case "sim_launcher":
      return <SimulationLauncher />;
    case "training_hub":
      return <TrainingHub />;
    case "data_gen":
      return <DataGeneration />;
    case "config_editor":
      return <ConfigEditor />;
    case "output_browser":
      return <OutputBrowser />;
    case "eval_runner":
      return <EvaluationRunner />;
    case "report_studio":
      return <ReportStudio />;
    case "system_tools":
      return <SystemTools />;
    case "settings":
      return <Settings />;
    default:
      return <SimulationMonitor />;
  }
}

function ActivePage() {
  const mode = useAppStore((s) => s.mode);
  const [visited, setVisited] = useState<Set<AppMode>>(() => new Set([mode]));

  useEffect(() => {
    setVisited((prev) => {
      if (prev.has(mode)) return prev;
      const next = new Set(prev);
      next.add(mode);
      return next;
    });
  }, [mode]);

  const isCurrentPersistent = PERSISTENT_MODES.includes(mode);

  return (
    <Suspense fallback={<PageFallback />}>
      {/* Mounted persistent views (Slice V4 zero-flicker retention) */}
      {PERSISTENT_MODES.filter((m) => visited.has(m)).map((m) => (
        <div
          key={m}
          style={{ display: mode === m ? "block" : "none" }}
          className="h-full"
        >
          {renderPage(m)}
        </div>
      ))}

      {/* Ephemeral / launcher views rendered on-demand */}
      {!isCurrentPersistent && (
        <div key={mode} className="h-full">
          {renderPage(mode)}
        </div>
      )}
    </Suspense>
  );
}

// Keyboard shortcut map — digit keys (no modifiers) for quick navigation
const DIGIT_MODES: AppMode[] = [
  "simulation",         // 1
  "simulation_summary", // 2
  "training",           // 3
  "benchmark",          // 4
  "sim_launcher",       // 5
  "training_hub",       // 6
  "process_monitor",    // 7
  "settings",           // 8
];

export default function App() {
  const { setMode, mode, effectiveTheme, projectRoot } = useAppStore();
  const refreshRecentLabels = useRecentFilesStore((s) => s.refreshRecentLabels);
  useThemeSync();
  const commandPaletteOpen = useLayoutStore((s) => s.commandPaletteOpen);
  const setCommandPaletteOpen = useLayoutStore((s) => s.setCommandPaletteOpen);
  const setShortcutsOpen = useLayoutStore((s) => s.setShortcutsOpen);
  const setGuidedTourOpen = useLayoutStore((s) => s.setGuidedTourOpen);
  const setGuidedTourStep = useLayoutStore((s) => s.setGuidedTourStep);
  const guidedTourOpen = useLayoutStore((s) => s.guidedTourOpen);
  const { triggerSim, triggerTrain, triggerDataGen, triggerEval } = useLaunchTriggerStore();

  useHashSync();
  useDuckDbInit();

  useEffect(() => {
    refreshRecentLabels(projectRoot);
  }, [projectRoot, refreshRecentLabels]);

  // Defer lazy route + vendor chunk warming until after first paint / idle (Slice V4)
  useEffect(() => {
    const warmChunks = () => {
      void Promise.all([
        import("./pages/monitor/SimulationMonitor"),
        import("./pages/monitor/TrainingMonitor"),
        import("./pages/monitor/ProcessMonitor"),
        import("./pages/analysis/SimulationSummary"),
        import("./pages/analysis/BenchmarkAnalysis"),
        import("./pages/analysis/CityComparison"),
        import("./pages/analysis/AlgorithmComparison"),
        import("./pages/analysis/OlapExplorer"),
        import("./pages/analysis/DataExplorer"),
        import("./pages/analysis/ExperimentTracker"),
        import("./pages/analysis/HPOTracker"),
        import("./pages/launch/SimulationLauncher"),
        import("./pages/launch/TrainingHub"),
        import("./pages/launch/DataGeneration"),
        import("./pages/launch/EvaluationRunner"),
        import("./pages/launch/ReportStudio"),
        import("./pages/files/ConfigEditor"),
        import("./pages/files/OutputBrowser"),
        import("./pages/files/SystemTools"),
        import("./pages/app/Settings"),
        import("echarts-for-react"),
        import("./components/maps/DeckRouteMap"),
      ]).then(() => markStartup("prefetchDone"));
      const modes: AppMode[] = [
        "simulation",
        "training",
        "simulation_summary",
        "benchmark",
        "city_comparison",
        "algorithms",
        "olap_explorer",
        "data_explorer",
        "experiment_tracker",
        "hpo_tracker",
        "process_monitor",
        "sim_launcher",
        "training_hub",
        "data_gen",
        "eval_runner",
        "report_studio",
        "config_editor",
        "output_browser",
        "system_tools",
        "settings",
      ];
      for (const m of modes) prefetchPage(m);
    };

    if (typeof window !== "undefined" && "requestIdleCallback" in window) {
      const id = window.requestIdleCallback(warmChunks, { timeout: 1500 });
      return () => window.cancelIdleCallback(id);
    }
    const timer = setTimeout(warmChunks, 400);
    return () => clearTimeout(timer);
  }, []);

  // Global keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ctrl+K → command palette (§G.7) — works even inside text fields
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setCommandPaletteOpen(!commandPaletteOpen);
        return;
      }

      const target = e.target as HTMLElement;
      const inTextField =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        target.isContentEditable;

      // Escape → close overlays
      if (e.key === "Escape") {
        if (commandPaletteOpen) setCommandPaletteOpen(false);
        else if (guidedTourOpen) setGuidedTourOpen(false);
        else setShortcutsOpen(false);
        return;
      }

      // Ctrl+Shift+/ → guided tour (§G.19)
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "/") {
        e.preventDefault();
        setGuidedTourStep(0);
        setGuidedTourOpen(true);
        return;
      }

      if (inTextField) return;

      // ? → keyboard shortcuts help overlay (§G.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && e.key === "?") {
        e.preventDefault();
        setShortcutsOpen(true);
        return;
      }
      // Ctrl+, → Settings
      if ((e.ctrlKey || e.metaKey) && e.key === ",") {
        e.preventDefault();
        setMode("settings");
        return;
      }
      // Ctrl+Shift+P → Process Monitor
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "P") {
        e.preventDefault();
        setMode("process_monitor");
        return;
      }
      // G → Geospatial (placeholder — navigates to simulation monitor for now)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "g") {
        e.preventDefault();
        setMode("simulation");
        return;
      }
      // Q → HPO query tracker
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "q") {
        e.preventDefault();
        setMode("hpo_tracker");
        return;
      }
      // T → Training monitor (§D.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "t") {
        e.preventDefault();
        setMode("training");
        return;
      }
      // H → Training & HPO hub (§D.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "h") {
        e.preventDefault();
        setMode("training_hub");
        return;
      }
      // E → Experiment tracker (§D.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "e") {
        e.preventDefault();
        setMode("experiment_tracker");
        return;
      }
      // L → Simulation launcher (§D.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "l") {
        e.preventDefault();
        setMode("sim_launcher");
        return;
      }
      // D → Data generation wizard (§D.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "d") {
        e.preventDefault();
        setMode("data_gen");
        return;
      }
      // V → Evaluation runner (§D.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "v") {
        e.preventDefault();
        setMode("eval_runner");
        return;
      }
      // B → Benchmark analysis (§D.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "b") {
        e.preventDefault();
        setMode("benchmark");
        return;
      }
      // O → Output browser (§D.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "o") {
        e.preventDefault();
        setMode("output_browser");
        return;
      }
      // P → Process monitor (§G.7 / §D.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "p") {
        e.preventDefault();
        setMode("process_monitor");
        return;
      }
      // M → Map / simulation digital twin (§G.7 / §D.7)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.key === "m") {
        e.preventDefault();
        setMode("simulation");
        return;
      }
      // Ctrl+R → launch on active launcher page (§G.7)
      if ((e.ctrlKey || e.metaKey) && e.key === "r") {
        e.preventDefault();
        if (mode === "sim_launcher") triggerSim();
        else if (mode === "training_hub") triggerTrain();
        else if (mode === "data_gen") triggerDataGen();
        else if (mode === "eval_runner") triggerEval();
        return;
      }
      // Digit 1-8 → quick mode switch
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
        const n = parseInt(e.key);
        if (n >= 1 && n <= DIGIT_MODES.length) {
          e.preventDefault();
          setMode(DIGIT_MODES[n - 1]);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    setMode,
    mode,
    triggerSim,
    triggerTrain,
    triggerDataGen,
    triggerEval,
    setShortcutsOpen,
    setCommandPaletteOpen,
    commandPaletteOpen,
    guidedTourOpen,
    setGuidedTourOpen,
    setGuidedTourStep,
  ]);

  // Global process event listener
  useProcessMonitor();

  return (
    <>
      <Layout>
        <ActivePage />
      </Layout>
      <Toaster
        theme={effectiveTheme}
        position="bottom-right"
        toastOptions={{
          classNames: {
            toast: "bg-canvas-elevated border-canvas-border text-gray-100",
          },
        }}
      />
    </>
  );
}
