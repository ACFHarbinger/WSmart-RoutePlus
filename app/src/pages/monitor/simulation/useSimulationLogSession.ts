import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { toast } from "sonner";
import { useLogPathRunLabelBrush } from "../../../hooks/brush/useLogPathRunLabelBrush";
import { useRecentHandoff } from "../../../hooks/files/useRecentHandoff";
import { useDuckDbStore } from "../../../store/duckdb";
import { useLoadedRunStore } from "../../../store/loadedRun";
import type {
  DayLogEntry,
  DiscoveredSimulationRun,
  PolicyVizEntry,
  SimFailureEntry,
  SimulationDirectoryCatalog,
} from "../../../types";
import { portfolioRunLabel, runSimulationArrowPipeline } from "../../../utils/duckdb/arrowPipeline";

/** Catalog `key` is the run directory. Older payloads only have each log's `path`. */
function simulationRunPath(run: DiscoveredSimulationRun, rootDir: string): string {
  if (run.key) return run.key;
  const logPath = run.logs.find((log) => log.path)?.path;
  if (logPath) return logPath;
  if (!rootDir) {
    throw new Error("No simulation log path to load");
  }
  return rootDir;
}

export function useSimulationLogSession({
  reset,
  loadEntries,
  loadPolicyVizEntries,
  loadFailureEntries,
  setWatchPath,
  pendingLogPath,
  setPendingLogPath,
}: {
  reset: () => void;
  loadEntries: (entries: DayLogEntry[]) => void;
  loadPolicyVizEntries: (entries: PolicyVizEntry[]) => void;
  loadFailureEntries: (entries: SimFailureEntry[]) => void;
  setWatchPath: (path: string | null) => void;
  pendingLogPath: string | null;
  setPendingLogPath: (path: string | null) => void;
}) {
  const setLoadedRun = useLoadedRunStore((s) => s.setLoadedRun);
  const { projectRoot, handoff } = useRecentHandoff();
  const {
    ready: duckdbReady,
    loading: duckdbLoading,
    lastPipeline,
    setLastPipeline,
    setLoading: setDuckdbLoading,
  } = useDuckDbStore();

  const [activeLogPath, setActiveLogPath] = useState<string | null>(null);
  useLogPathRunLabelBrush(activeLogPath);
  const sourceRunLabel = useMemo(
    () => (activeLogPath ? portfolioRunLabel(activeLogPath, undefined, projectRoot) : null),
    [activeLogPath, projectRoot]
  );

  const loadLogFile = useCallback(
    async (path: string, watch = true) => {
      if (!path) {
        throw new Error("No simulation log path to load");
      }
      reset();
      const [historical, vizHistorical, failureHistorical] = await Promise.all([
        invoke<DayLogEntry[]>("load_simulation_log", { path }),
        invoke<PolicyVizEntry[]>("load_policy_viz_log", { path }).catch(() => []),
        invoke<SimFailureEntry[]>("load_sim_failure_log", { path }).catch(() => []),
      ]);
      loadEntries(historical);
      loadPolicyVizEntries(vizHistorical);
      loadFailureEntries(failureHistorical);
      setActiveLogPath(path);
      if (watch) setWatchPath(path);
      setLoadedRun({
        path,
        label: portfolioRunLabel(path, undefined, projectRoot) ?? "simulation_log",
        entries: historical,
        source: watch ? "live" : "viewing",
      });
      handoff(path, "log", { navigate: false });

      if (duckdbReady) {
        setDuckdbLoading(true);
        runSimulationArrowPipeline(path, "monitor_sim", projectRoot)
          .then(setLastPipeline)
          .catch((err) => console.warn("Simulation Arrow pipeline:", err))
          .finally(() => setDuckdbLoading(false));
      }
    },
    [
      reset,
      loadEntries,
      loadPolicyVizEntries,
      loadFailureEntries,
      setWatchPath,
      setLoadedRun,
      handoff,
      duckdbReady,
      projectRoot,
      setLastPipeline,
      setDuckdbLoading,
    ]
  );

  const [catalog, setCatalog] = useState<SimulationDirectoryCatalog | null>(null);
  const [selectedRunKey, setSelectedRunKey] = useState<string | null>(null);

  const loadDirectory = useCallback(
    async (dirPath: string) => {
      try {
        const cat = await invoke<SimulationDirectoryCatalog>("scan_simulation_directory", {
          dirPath,
        });
        if (!cat.runs || cat.runs.length === 0) {
          toast.error("No simulation logs found in directory", { description: dirPath });
          return;
        }
        setCatalog(cat);
        const firstRun = cat.runs[0];
        const runPath = simulationRunPath(firstRun, cat.root_dir);
        setSelectedRunKey(runPath);
        await loadLogFile(runPath);
        toast.success(`Loaded catalog: ${cat.runs.length} run(s), ${cat.total_logs} log(s)`);
      } catch (err) {
        toast.error("Failed to scan simulation directory", { description: String(err) });
      }
    },
    [loadLogFile]
  );

  const openLog = useCallback(async () => {
    const path = (await open({
      filters: [
        { name: "Simulation Logs", extensions: ["json", "jsonl", "log", "txt"] },
        { name: "All Files", extensions: ["*"] },
      ],
    })) as string | null;
    if (!path) return;
    setCatalog(null);
    setSelectedRunKey(null);
    await loadLogFile(path);
  }, [loadLogFile]);

  const openDirectory = useCallback(async () => {
    const selected = (await open({
      directory: true,
      multiple: false,
    })) as string | null;
    if (!selected) return;
    await loadDirectory(selected);
  }, [loadDirectory]);

  const handleRunChange = useCallback(
    async (runKey: string) => {
      setSelectedRunKey(runKey);
      if (runKey === "__ALL__" && catalog) {
        await loadLogFile(catalog.root_dir);
      } else if (runKey) {
        await loadLogFile(runKey);
      }
    },
    [catalog, loadLogFile]
  );

  useEffect(() => {
    if (pendingLogPath) {
      const p = pendingLogPath;
      setPendingLogPath(null);
      invoke<boolean>("path_exists", { path: p }).then((exists) => {
        if (!exists) return;
        invoke<SimulationDirectoryCatalog>("scan_simulation_directory", { dirPath: p })
          .then((cat) => {
            if (cat.runs && cat.runs.length > 0) {
              setCatalog(cat);
              const runPath = simulationRunPath(cat.runs[0], cat.root_dir);
              setSelectedRunKey(runPath);
              void loadLogFile(runPath, false);
            } else {
              void loadLogFile(p, false);
            }
          })
          .catch(() => {
            void loadLogFile(p, false);
          });
      });
    }
  }, [pendingLogPath, setPendingLogPath, loadLogFile]);

  return {
    activeLogPath,
    sourceRunLabel,
    catalog,
    selectedRunKey,
    openLog,
    openDirectory,
    handleRunChange,
    projectRoot,
    duckdbReady,
    duckdbLoading,
    lastPipeline,
  };
}
