import { lazy, Suspense, useEffect } from "react";
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
const CloudLauncher = lazy(() =>
  import("./pages/cloud/CloudLauncher").then((m) => ({ default: m.CloudLauncher }))
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

function ActivePage() {
  const mode = useAppStore((s) => s.mode);

  let page: React.ReactNode;
  switch (mode) {
    case "simulation":
      page = <SimulationMonitor />;
      break;
    case "training":
      page = <TrainingMonitor />;
      break;
    case "simulation_summary":
      page = <SimulationSummary />;
      break;
    case "benchmark":
      page = <BenchmarkAnalysis />;
      break;
    case "city_comparison":
      page = <CityComparison />;
      break;
    case "olap_explorer":
      page = <OlapExplorer />;
      break;
    case "data_explorer":
      page = <DataExplorer />;
      break;
    case "experiment_tracker":
      page = <ExperimentTracker />;
      break;
    case "algorithms":
      page = <AlgorithmComparison />;
      break;
    case "hpo_tracker":
      page = <HPOTracker />;
      break;
    case "process_monitor":
      page = <ProcessMonitor />;
      break;
    case "cloud_monitor":
      page = <CloudMonitor />;
      break;
    case "cloud_launcher":
      page = <CloudLauncher />;
      break;
    case "cloud_analysis":
      page = <CloudAnalysis />;
      break;
    case "sim_launcher":
      page = <SimulationLauncher />;
      break;
    case "cloud_launcher":
      page = <CloudLauncher />;
      break;
    case "training_hub":
      page = <TrainingHub />;
      break;
    case "data_gen":
      page = <DataGeneration />;
      break;
    case "config_editor":
      page = <ConfigEditor />;
      break;
    case "output_browser":
      page = <OutputBrowser />;
      break;
    case "eval_runner":
      page = <EvaluationRunner />;
      break;
    case "report_studio":
      page = <ReportStudio />;
      break;
    case "system_tools":
      page = <SystemTools />;
      break;
    case "settings":
      page = <Settings />;
      break;
    default:
      page = <SimulationMonitor />;
  }

  return <Suspense fallback={<PageFallback />}>{page}</Suspense>;
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

  // Warm all lazy route + vendor chunks on startup (§G.7)
  useEffect(() => {
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
      import("maplibre-gl"),
      import("@deck.gl/react"),
      import("@monaco-editor/react"),
      import("@duckdb/duckdb-wasm"),
      import("sigma"),
      import("@react-three/fiber"),
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
    for (const mode of modes) prefetchPage(mode);
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
