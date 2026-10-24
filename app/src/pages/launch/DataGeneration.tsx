/**
 * Data Generation Wizard — full Hydra form for `main.py gen_data` (§G.11).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Play, ChevronDown, ChevronUp, Terminal } from "lucide-react";
import { open } from "@tauri-apps/plugin-dialog";
import { toast } from "sonner";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { LauncherLivePanel } from "../../components/monitor/live/LauncherLivePanel";
import { ProcessIdFooter } from "../../components/monitor/process/ProcessIdFooter";
import { DataGenSourceAndPreview } from "../../components/launch/DataGenPreviewPanel";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useLaunchTriggerStore } from "../../store/launchTrigger";
import { TRAIN_PROBLEMS, useDataGenStore } from "../../store/launchers";
import { useProcessStore } from "../../store/process";
import { useSpawnProcess } from "../../hooks/process/useSpawnProcess";
import { useProcessRunLabelBrush } from "../../hooks/brush/useProcessRunLabelBrush";
import { dataExplorerPathFromGenData } from "../../utils/runs/genDataPath";
import { brushLogPathFromProcessLines, outputRunPathFromLogLines } from "../../utils/runs/outputRunPath";
import { dataGenLivePanelTitle, findRecentLauncherProcessId } from "../../utils/process/launcherProcess";
import type { DatasetPreviewStats } from "../../types";

const PROBLEMS = [...TRAIN_PROBLEMS, "all"] as const;
const DISTRIBUTIONS = ["gamma3", "emp"] as const;
const DATASET_TYPES = [
  { value: "test_simulator", label: "Test Simulator" },
  { value: "train", label: "Training" },
  { value: "train_time", label: "Training (timed)" },
] as const;
const AREAS = ["figueiradafoz", "riomaior"] as const;

const DIST_LABELS: Record<string, string> = {
  gamma3: "Gamma-3",
  emp: "Empirical",
};

export function DataGeneration() {
  const { projectRoot, pythonPath } = useAppStore();
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const { spawn, launching } = useSpawnProcess();

  const {
    dataSource,
    tsplibPath,
    sensorCsvPath,
    problem,
    distributions,
    datasetType,
    seed,
    overwrite,
    area,
    numLoc,
    nSamples,
    nDays,
    extraOverrides,
    patch,
  } = useDataGenStore();

  const setDataSource = (v: "synthetic" | "tsplib" | "sensor") =>
    patch({ dataSource: v });
  const setTsplibPath = (v: string) => patch({ tsplibPath: v });
  const setSensorCsvPath = (v: string) => patch({ sensorCsvPath: v });
  const setProblem = (v: string) => patch({ problem: v });
  const setDistributions = (v: string[]) => patch({ distributions: v });
  const setDatasetType = (v: string) => patch({ datasetType: v });
  const setSeed = (v: number) => patch({ seed: v });
  const setOverwrite = (v: boolean) => patch({ overwrite: v });
  const setArea = (v: string) => patch({ area: v });
  const setNumLoc = (v: number) => patch({ numLoc: v });
  const setNSamples = (v: number) => patch({ nSamples: v });
  const setNDays = (v: number) => patch({ nDays: v });
  const setExtraOverrides = (v: string) => patch({ extraOverrides: v });

  const [showAdvanced, setShowAdvanced] = useState(false);

  const [liveProcessId, setLiveProcessId] = useState<string | null>(null);
  const processes = useProcessStore((s) => s.processes);
  const displayProcessId = useMemo(
    () => liveProcessId ?? findRecentLauncherProcessId(processes, "data_gen"),
    [liveProcessId, processes]
  );
  const displayProc = displayProcessId ? processes[displayProcessId] : null;
  const runStatus = displayProc?.status ?? null;
  const [previewStats, setPreviewStats] = useState<DatasetPreviewStats | null>(
    null
  );
  const [previewDatasetPath, setPreviewDatasetPath] = useState("");
  const [previewLoading, setPreviewLoading] = useState(false);

  const toggleDist = (d: string) => {
    const next = distributions.includes(d)
      ? distributions.filter((x) => x !== d)
      : [...distributions, d];
    setDistributions(next);
  };

  const pickTsplibFile = async () => {
    const path = (await open({
      filters: [{ name: "TSPLIB / VRP", extensions: ["vrp", "tsp"] }],
    })) as string | null;
    if (path) setTsplibPath(path);
  };

  const pickSensorFile = async () => {
    const path = (await open({
      filters: [{ name: "Sensor CSV", extensions: ["csv"] }],
    })) as string | null;
    if (path) setSensorCsvPath(path);
  };

  const previewDataset = async () => {
    if (!projectRoot) return;
    const path = (await open({
      filters: [{ name: "Dataset", extensions: ["pkl", "pt"] }],
    })) as string | null;
    if (!path) return;
    setPreviewDatasetPath(path);
    setPreviewLoading(true);
    try {
      const stats = await invoke<DatasetPreviewStats>("preview_dataset_stats", {
        path,
        projectRoot,
        pythonExecutable: pythonPath || null,
      });
      setPreviewStats(stats);
    } catch (err) {
      toast.error("Preview failed", { description: String(err) });
      setPreviewStats(null);
    } finally {
      setPreviewLoading(false);
    }
  };

  const hydraArgs = useMemo(() => {
    const distList =
      distributions.length > 0 ? distributions.join(",") : "gamma3";
    const sourceArgs =
      dataSource === "tsplib" && tsplibPath
        ? [`data.source=tsplib`, `data.tsplib_instance=${tsplibPath}`]
        : dataSource === "sensor" && sensorCsvPath
          ? [`data.source=sensor`, `data.sensor_file=${sensorCsvPath}`]
          : [
              `data.graphs.0.area=${area}`,
              `data.graphs.0.num_loc=${numLoc}`,
              `data.graphs.0.n_samples=${nSamples}`,
              `data.graphs.0.n_days=${nDays}`,
            ];
    const args = [
      `data.problem=${problem}`,
      `data.data_distributions=[${distList}]`,
      `data.dataset_type=${datasetType}`,
      `data.overwrite=${overwrite}`,
      ...sourceArgs,
      `seed=${seed}`,
    ];
    const extra = extraOverrides
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    return [...args, ...extra];
  }, [
    problem,
    distributions,
    datasetType,
    overwrite,
    dataSource,
    tsplibPath,
    sensorCsvPath,
    area,
    numLoc,
    nSamples,
    nDays,
    seed,
    extraOverrides,
  ]);

  const commandPreview = `python main.py gen_data \\\n  ${hydraArgs.join(" \\\n  ")}`;

  const launch = useCallback(async () => {
    if (!projectRoot) return;
    if (dataSource === "tsplib" && !tsplibPath.trim()) return;
    if (dataSource === "sensor" && !sensorCsvPath.trim()) return;
    const procId = `gen_data_${Date.now()}`;
    setLiveProcessId(procId);
    await spawn({
      id: procId,
      pythonArgs: ["main.py", "gen_data", ...hydraArgs],
      workingDir: projectRoot,
    });
  }, [projectRoot, dataSource, tsplibPath, sensorCsvPath, hydraArgs, spawn]);

  const dataGenNonce = useLaunchTriggerStore((s) => s.dataGenNonce);
  useEffect(() => {
    if (dataGenNonce > 0) void launch();
  }, [dataGenNonce, launch]);

  const liveLogLines = displayProc?.logLines ?? [];
  const liveRunLabel = useProcessRunLabelBrush(displayProcessId, liveLogLines);
  const outputRunPath = useMemo(
    () => outputRunPathFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const dataExplorerCsvPath = useMemo(() => {
    if (sensorCsvPath.trim()) return sensorCsvPath.trim();
    if (displayProc?.command) {
      return dataExplorerPathFromGenData(displayProc.command, liveLogLines);
    }
    return dataExplorerPathFromGenData("", liveLogLines);
  }, [sensorCsvPath, displayProc?.command, liveLogLines]);
  const liveLogPath = useMemo(
    () => brushLogPathFromProcessLines(liveLogLines, "sim"),
    [liveLogLines]
  );

  const isDone = runStatus !== null && runStatus !== "running";

  return (
    <div className="space-y-4 max-w-2xl">
      <GlobalFilterBar
        showLogScale
        runLabels={liveRunLabel ? [liveRunLabel] : []}
      />

      <DataGenSourceAndPreview
        dataSource={dataSource}
        setDataSource={setDataSource}
        tsplibPath={tsplibPath}
        setTsplibPath={setTsplibPath}
        pickTsplibFile={() => void pickTsplibFile()}
        sensorCsvPath={sensorCsvPath}
        setSensorCsvPath={setSensorCsvPath}
        pickSensorFile={() => void pickSensorFile()}
        projectRoot={projectRoot}
        previewLoading={previewLoading}
        previewDataset={() => void previewDataset()}
        previewDatasetPath={previewDatasetPath}
        previewStats={previewStats}
        logScale={logScale}
      />

      {/* Problem + distribution */}
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-200">Dataset</h2>

        <div className="flex flex-wrap gap-6">
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Problem</label>
            <select
              className="select-base w-36"
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
            >
              {PROBLEMS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Dataset Type</label>
            <select
              className="select-base w-44"
              value={datasetType}
              onChange={(e) => setDatasetType(e.target.value)}
            >
              {DATASET_TYPES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Seed</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={seed}
              min={0}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>
        </div>

        <div>
          <label className="block text-xs text-canvas-muted mb-2">
            Distributions
          </label>
          <div className="flex gap-4">
            {DISTRIBUTIONS.map((d) => (
              <label
                key={d}
                className="flex items-center gap-2 cursor-pointer text-sm text-gray-300"
              >
                <input
                  type="checkbox"
                  className="accent-accent-primary"
                  checked={distributions.includes(d)}
                  onChange={() => toggleDist(d)}
                />
                {DIST_LABELS[d]}
              </label>
            ))}
          </div>
          {distributions.length === 0 && (
            <p className="text-xs text-accent-warning mt-1">
              Select at least one distribution.
            </p>
          )}
        </div>

        <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-300 w-fit">
          <input
            type="checkbox"
            className="accent-accent-primary"
            checked={overwrite}
            onChange={(e) => setOverwrite(e.target.checked)}
          />
          Overwrite existing files
        </label>
      </div>

      {dataSource === "synthetic" && (
        <div className="card space-y-4">
          <h2 className="text-sm font-semibold text-gray-200">Graph</h2>
          <div className="flex flex-wrap gap-4">
            <div className="flex flex-col gap-1.5">
              <label className="text-xs text-canvas-muted">Area</label>
              <select
                className="select-base w-44"
                value={area}
                onChange={(e) => setArea(e.target.value)}
              >
                {AREAS.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-xs text-canvas-muted">Locations</label>
              <input
                type="number"
                className="input-base font-mono text-sm w-24"
                value={numLoc}
                min={1}
                onChange={(e) => setNumLoc(Number(e.target.value))}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-xs text-canvas-muted">Samples</label>
              <input
                type="number"
                className="input-base font-mono text-sm w-24"
                value={nSamples}
                min={1}
                onChange={(e) => setNSamples(Number(e.target.value))}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-xs text-canvas-muted">Days</label>
              <input
                type="number"
                className="input-base font-mono text-sm w-24"
                value={nDays}
                min={1}
                onChange={(e) => setNDays(Number(e.target.value))}
              />
            </div>
          </div>

          <p className="text-xs text-canvas-muted">
            Configures <code>data.graphs[0]</code>. For multi-graph generation
            use Advanced Overrides.
          </p>
        </div>
      )}

      {/* Advanced overrides */}
      <div className="card">
        <button
          className="w-full flex items-center justify-between text-sm font-medium text-gray-300"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          <span>Extra Hydra Overrides</span>
          {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        {showAdvanced && (
          <textarea
            className="input-base w-full font-mono text-xs h-20 resize-y mt-3"
            value={extraOverrides}
            onChange={(e) => setExtraOverrides(e.target.value)}
            placeholder="data.penalty_factor=2.5"
            spellCheck={false}
          />
        )}
      </div>

      {/* Command preview */}
      <div className="card space-y-2">
        <div className="flex items-center gap-2 text-xs text-canvas-muted">
          <Terminal size={12} />
          Command preview
        </div>
        <pre className="font-mono text-xs text-accent-secondary whitespace-pre-wrap bg-canvas-bg rounded-lg p-3">
          {commandPreview}
        </pre>
      </div>

      {/* Launch */}
      <div className="flex items-center gap-3">
        {!projectRoot && (
          <p className="text-xs text-accent-warning">
            Configure Project Root in Settings first.
          </p>
        )}
        <button
          onClick={() => void launch()}
          disabled={
            launching ||
            !projectRoot ||
            distributions.length === 0 ||
            (dataSource === "tsplib" && !tsplibPath.trim()) ||
            (dataSource === "sensor" && !sensorCsvPath.trim())
          }
          className="btn-primary flex items-center gap-2"
        >
          <Play size={14} />
          {launching ? "Generating…" : "Generate Dataset"}
        </button>
      </div>

      {displayProcessId && (
        <LauncherLivePanel
          header={{
            status: isDone ? (runStatus ?? "running") : "running",
            title: dataGenLivePanelTitle({
              isRunning: !isDone,
              status: runStatus ?? undefined,
            }),
            runLabel: liveRunLabel,
            logPath: liveLogPath,
            projectRoot,
            navMesh: {
              kind: "data_gen",
              hideSelf: true,
              showPostRun: isDone && runStatus === "completed",
              showOutputBrowser: isDone && runStatus === "completed",
              outputRunPath,
              csvPath: dataExplorerCsvPath,
            },
          }}
          progress={
            !isDone && displayProcessId
              ? { processId: displayProcessId }
              : undefined
          }
          footer={
            <ProcessIdFooter
              processId={displayProcessId}
              logPath={liveLogPath}
              runLabel={liveRunLabel}
              projectRoot={projectRoot}
            />
          }
          logLines={liveLogLines}
          logTailWaiting={!isDone}
        />
      )}
    </div>
  );
}
