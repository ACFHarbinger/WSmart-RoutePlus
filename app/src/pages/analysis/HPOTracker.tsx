/**
 * HPO Tracker — Optuna study browser with ECharts & Orion Honeycomb visualizations (§G.18, Slice V6).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { open as openUrl } from "@tauri-apps/plugin-shell";
import { Copy, Download, ExternalLink, FolderOpen, RefreshCw } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { ProcessIdFooter } from "../../components/monitor/process/ProcessIdFooter";
import { TrainHpoLivePanel } from "../../components/monitor/live/TrainHpoLivePanel";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { toast } from "sonner";
import { useProcessRunLabelBrush } from "../../hooks/brush/useProcessRunLabelBrush";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useProcessStore } from "../../store/process";
import { collectAttentionVizFromLogLines } from "../../utils/graph/attentionViz";
import { collectTrainingHealthFromLogLines } from "../../utils/training/trainingHealth";
import { brushLogPathFromProcessLines, outputRunPathFromLogLines, resolveLocalProjectPath, sqliteStoragePathFromUrl } from "../../utils/runs/outputRunPath";
import { runLabelFromPath } from "../../utils/benchmark/policyTelemetryTrends";
import { trainingRunPathFromLogLines } from "../../utils/training/trainingRunPath";
import { trialLogDirFromUserAttrs } from "../../utils/training/trialRunPath";
import { collectTrainingMetricsFromLogLines } from "../../utils/training/trainingMetrics";
import { findRecentHpoProcessId, trainHpoLivePanelTitle } from "../../utils/training/trainingProcess";
import { buildCrossStudyHistoryOption, buildHistoryOption, buildParallelOption, HpoStudyVisualsSection } from "../../components/analysis/hpo/HpoChartsAndTable";
import type { HpoReportExportResult, OptunaStudyData, OptunaStudySummary, OptunaTrial } from "../../types";

const DEFAULT_STORAGE = "sqlite:///assets/hpo/study.db";

export function HPOTracker() {
  const { projectRoot, pythonPath, effectiveTheme } = useAppStore();
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const activeRunLabel = useGlobalFiltersStore((s) => s.runLabel);
  const processes = useProcessStore((s) => s.processes);
  const [storageUrl, setStorageUrl] = useState(DEFAULT_STORAGE);
  const [studies, setStudies] = useState<OptunaStudySummary[]>([]);
  const [selectedStudy, setSelectedStudy] = useState<string | null>(null);
  const [studyData, setStudyData] = useState<OptunaStudyData | null>(null);
  const [selectedTrialNumbers, setSelectedTrialNumbers] = useState<number[]>([]);
  const [compareStudy, setCompareStudy] = useState<string | null>(null);
  const [compareStudyData, setCompareStudyData] = useState<OptunaStudyData | null>(null);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [lastReportDir, setLastReportDir] = useState<string | null>(null);

  const recentHpoId = useMemo(
    () => findRecentHpoProcessId(processes),
    [processes]
  );
  const recentHpoProc = recentHpoId ? processes[recentHpoId] : null;
  const recentHpoRunning = recentHpoProc?.status === "running";
  const recentHpoDone =
    recentHpoProc != null && recentHpoProc.status !== "running";
  const processRunLabel = useProcessRunLabelBrush(
    recentHpoId,
    recentHpoProc?.logLines
  );

  const outputRunPath = useMemo(
    () =>
      recentHpoProc ? outputRunPathFromLogLines(recentHpoProc.logLines) : null,
    [recentHpoProc]
  );
  const trainingRunPath = useMemo(
    () =>
      recentHpoProc
        ? trainingRunPathFromLogLines(recentHpoProc.logLines)
        : null,
    [recentHpoProc]
  );
  const processLogPath = useMemo(
    () =>
      recentHpoProc
        ? brushLogPathFromProcessLines(recentHpoProc.logLines, "train")
        : null,
    [recentHpoProc]
  );

  const liveHealthEntries = useMemo(
    () =>
      recentHpoProc
        ? collectTrainingHealthFromLogLines(recentHpoProc.logLines)
        : [],
    [recentHpoProc]
  );

  const liveAttentionEntries = useMemo(
    () =>
      recentHpoProc
        ? collectAttentionVizFromLogLines(recentHpoProc.logLines)
        : [],
    [recentHpoProc]
  );

  const liveMetrics = useMemo(
    () =>
      recentHpoProc
        ? collectTrainingMetricsFromLogLines(recentHpoProc.logLines)
        : [],
    [recentHpoProc]
  );

  const storageDbPath = useMemo(
    () => sqliteStoragePathFromUrl(storageUrl, projectRoot),
    [storageUrl, projectRoot]
  );
  const reportDirPath = useMemo(
    () =>
      lastReportDir ? resolveLocalProjectPath(lastReportDir, projectRoot) : null,
    [lastReportDir, projectRoot]
  );

  const trialBrushLabel = useCallback(
    (trial: OptunaTrial): string | null => {
      const logDir = trialLogDirFromUserAttrs(trial.user_attrs);
      if (!logDir) return null;
      const resolved = resolveLocalProjectPath(logDir, projectRoot) ?? logDir;
      return runLabelFromPath(resolved);
    },
    [projectRoot]
  );

  const filterRunLabels = useMemo(() => {
    if (processRunLabel) return [processRunLabel];
    if (trainingRunPath) {
      const resolved =
        resolveLocalProjectPath(trainingRunPath, projectRoot) ?? trainingRunPath;
      return [runLabelFromPath(resolved)];
    }
    if (outputRunPath) {
      const resolved =
        resolveLocalProjectPath(outputRunPath, projectRoot) ?? outputRunPath;
      return [runLabelFromPath(resolved)];
    }
    const labels = selectedTrialNumbers
      .map((n) => studyData?.trials.find((t) => t.number === n))
      .filter((t): t is OptunaTrial => t != null)
      .map(trialBrushLabel)
      .filter((l): l is string => l != null);
    return labels.length > 0 ? labels : [];
  }, [
    processRunLabel,
    trainingRunPath,
    outputRunPath,
    projectRoot,
    selectedTrialNumbers,
    studyData,
    trialBrushLabel,
  ]);

  const toggleTrial = useCallback((trialNumber: number) => {
    setSelectedTrialNumbers((prev) =>
      prev.includes(trialNumber)
        ? prev.filter((n) => n !== trialNumber)
        : [...prev, trialNumber]
    );
  }, []);

  const refreshStudies = useCallback(async () => {
    if (!projectRoot) return;
    setLoading(true);
    try {
      const found = await invoke<OptunaStudySummary[]>("list_optuna_studies", {
        storageUrl,
        projectRoot,
        pythonExecutable: pythonPath || null,
      });
      setStudies(found);
      if (found.length > 0 && !selectedStudy) {
        setSelectedStudy(found[0].name);
      }
    } catch (err) {
      toast.error("Failed to list Optuna studies", { description: String(err) });
      setStudies([]);
    } finally {
      setLoading(false);
    }
  }, [projectRoot, pythonPath, storageUrl, selectedStudy]);

  const loadStudy = useCallback(
    async (name: string) => {
      if (!projectRoot) return;
      setLoading(true);
      try {
        const data = await invoke<OptunaStudyData>("load_optuna_study", {
          storageUrl,
          studyName: name,
          projectRoot,
          pythonExecutable: pythonPath || null,
        });
        setStudyData(data);
      } catch (err) {
        toast.error("Failed to load study", { description: String(err) });
        setStudyData(null);
      } finally {
        setLoading(false);
      }
    },
    [projectRoot, pythonPath, storageUrl]
  );

  useEffect(() => {
    if (projectRoot) void refreshStudies();
  }, [projectRoot, storageUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (selectedStudy && projectRoot) void loadStudy(selectedStudy);
  }, [selectedStudy]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setSelectedTrialNumbers([]);
  }, [selectedStudy]);

  useEffect(() => {
    if (!compareStudy || !projectRoot) {
      setCompareStudyData(null);
      return;
    }
    invoke<OptunaStudyData>("load_optuna_study", {
      storageUrl,
      studyName: compareStudy,
      projectRoot,
      pythonExecutable: pythonPath || null,
    })
      .then(setCompareStudyData)
      .catch(() => setCompareStudyData(null));
  }, [compareStudy, projectRoot, pythonPath, storageUrl]);

  const historyOption = useMemo(
    () => (studyData ? buildHistoryOption(studyData.trials, logScale) : null),
    [studyData, logScale]
  );

  const crossStudyOption = useMemo(() => {
    if (!studyData || !compareStudyData) return null;
    return buildCrossStudyHistoryOption(
      [
        { name: studyData.name, trials: studyData.trials },
        { name: compareStudyData.name, trials: compareStudyData.trials },
      ],
      logScale
    );
  }, [studyData, compareStudyData, logScale]);

  const parallelOption = useMemo(
    () => (studyData ? buildParallelOption(studyData, logScale) : null),
    [studyData, logScale]
  );

  const exportPlotlyReports = useCallback(async () => {
    if (!projectRoot || !selectedStudy) return;
    setExporting(true);
    try {
      const result = await invoke<HpoReportExportResult>(
        "export_optuna_reports",
        {
          storageUrl,
          studyName: selectedStudy,
          projectRoot,
          pythonExecutable: pythonPath || null,
          outputDir: null,
        }
      );
      if (result.report_dir) {
        setLastReportDir(result.report_dir);
        toast.success("Plotly HPO reports exported", {
          description: `${result.files.length} files in ${result.report_dir}`,
        });
      } else {
        toast.info(result.message);
      }
    } catch (err) {
      toast.error("Failed to export HPO reports", { description: String(err) });
    } finally {
      setExporting(false);
    }
  }, [projectRoot, pythonPath, selectedStudy, storageUrl]);

  const openReportDir = useCallback(async () => {
    if (!lastReportDir) return;
    try {
      await openUrl(lastReportDir);
    } catch (err) {
      toast.error("Could not open report folder", { description: String(err) });
    }
  }, [lastReportDir]);

  const copyBestParams = useCallback(async () => {
    if (!studyData?.best_params) return;
    const lines = Object.entries(studyData.best_params).map(
      ([k, v]) => `${k}=${v}`
    );
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      toast.success("Copied best trial params as Hydra overrides");
    } catch {
      toast.error("Clipboard write failed");
    }
  }, [studyData]);

  const pickStorage = async () => {
    const path = (await open({
      filters: [{ name: "SQLite DB", extensions: ["db", "sqlite"] }],
    })) as string | null;
    if (path) setStorageUrl(`sqlite:///${path}`);
  };

  if (!projectRoot) {
    return (
      <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
        Set project root to browse Optuna studies.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <GlobalFilterBar showLogScale runLabels={filterRunLabels} />

      {recentHpoId && recentHpoProc && (
        <TrainHpoLivePanel
          cardClassName="border-accent-success/30"
          header={{
            status: recentHpoRunning ? "running" : recentHpoProc.status,
            title: trainHpoLivePanelTitle({
              isRunning: recentHpoRunning,
              status: recentHpoProc.status,
              processId: recentHpoId,
              command: recentHpoProc.command,
              kind: "hpo",
            }),
            runLabel: processRunLabel,
            logPath: processLogPath,
            projectRoot,
            showLiveSuffix: recentHpoRunning,
            metricCount: liveMetrics.length,
            healthCount: liveHealthEntries.length,
            attentionCount: liveAttentionEntries.length,
            navMesh: {
              showHpoLinks: true,
              showOutputBrowser:
                recentHpoDone && recentHpoProc.status === "completed",
              outputRunPath,
              trainingRunPath,
            },
          }}
          progress={
            recentHpoRunning ? { processId: recentHpoId } : undefined
          }
          analytics={{
            metrics: liveMetrics,
            healthEntries: liveHealthEntries,
            attentionEntries: liveAttentionEntries,
            logScale,
            theme: effectiveTheme,
            exportNamePrefix: "hpo-tracker",
            isPostRun: !recentHpoRunning,
            postRunFallback:
              "Post-run shortcuts — open Output Browser or Training Monitor for this sweep",
          }}
          logLines={recentHpoProc.logLines}
          logTailWaiting={recentHpoRunning}
          footer={
            <ProcessIdFooter
              processId={recentHpoId}
              logPath={processLogPath}
              runLabel={processRunLabel}
              projectRoot={projectRoot}
            />
          }
        />
      )}

      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Optuna Storage</h2>
        <div className="flex gap-2">
          <input
            type="text"
            className="input-base font-mono text-xs flex-1"
            value={storageUrl}
            onChange={(e) => setStorageUrl(e.target.value)}
            placeholder="sqlite:///assets/hpo/study.db"
          />
          <button onClick={pickStorage} className="btn-ghost p-2" title="Browse DB file">
            <FolderOpen size={14} />
          </button>
          <button onClick={refreshStudies} disabled={loading} className="btn-primary flex items-center gap-2">
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} /> Refresh
          </button>
        </div>
        {storageDbPath && <OpenPathToolbar path={storageDbPath} chipClassName="max-w-full" handoff={false} />}
        {reportDirPath && (
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-[10px] text-canvas-muted shrink-0">Reports</span>
            <OpenPathToolbar path={reportDirPath} kind="run" labeled chipClassName="flex-1 min-w-0" className="flex-1 min-w-0">
              <button onClick={openReportDir} className="btn-ghost text-xs flex items-center gap-1 shrink-0" title="Open report folder in file manager">
                <ExternalLink size={12} /> Open
              </button>
            </OpenPathToolbar>
          </div>
        )}
      </div>

      {studies.length > 0 && (
        <div className="card space-y-3">
          <div className="flex items-center gap-3">
            <label className="text-xs text-canvas-muted">Study</label>
            <select className="select-base flex-1" value={selectedStudy ?? ""} onChange={(e) => setSelectedStudy(e.target.value)}>
              {studies.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name} — {s.n_complete}/{s.n_trials} trials{s.best_value != null ? ` (best: ${s.best_value.toFixed(4)})` : ""}
                </option>
              ))}
            </select>
            <div className="flex items-center gap-2">
              {studyData && studyData.trials.filter((t) => t.state === "COMPLETE").length >= 2 && (
                <button
                  onClick={exportPlotlyReports}
                  disabled={exporting}
                  className="btn-ghost text-xs flex items-center gap-1"
                  title="Export optuna.visualization Plotly HTML to assets/hpo_reports/"
                >
                  <Download size={12} className={exporting ? "animate-pulse" : ""} /> Export Plotly
                </button>
              )}
              {studyData && Object.keys(studyData.best_params).length > 0 && (
                <button onClick={copyBestParams} className="btn-ghost text-xs flex items-center gap-1">
                  <Copy size={12} /> Copy best params
                </button>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-canvas-muted">Compare with</label>
            <select className="select-base flex-1 text-xs" value={compareStudy ?? ""} onChange={(e) => setCompareStudy(e.target.value || null)}>
              <option value="">— none —</option>
              {studies
                .filter((s) => s.name !== selectedStudy)
                .map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.name}{s.best_value != null ? ` (best: ${s.best_value.toFixed(4)})` : ""}
                  </option>
                ))}
            </select>
          </div>

          {studyData && (
            <div className="grid grid-cols-4 gap-3 text-xs">
              <div className="kpi-card">
                <p className="text-canvas-muted">Trials</p>
                <p className="text-lg font-mono text-gray-100 tabular-nums">{studyData.trials.length}</p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">Completed</p>
                <p className="text-lg font-mono text-gray-100 tabular-nums">
                  {studyData.trials.filter((t) => t.state === "COMPLETE").length}
                </p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">Best Value</p>
                <p className="text-lg font-mono text-accent-success tabular-nums">
                  {studyData.best_value != null ? studyData.best_value.toFixed(6) : "—"}
                </p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">Parameters</p>
                <p className="text-lg font-mono text-gray-100 tabular-nums">
                  {Object.keys(studyData.importances).length || "—"}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {studyData && (
        <HpoStudyVisualsSection
          studyData={studyData} compareStudyData={compareStudyData} historyOption={historyOption}
          crossStudyOption={crossStudyOption} parallelOption={parallelOption} logScale={logScale}
          projectRoot={projectRoot} activeRunLabel={activeRunLabel}
          selectedTrialNumbers={selectedTrialNumbers} toggleTrial={toggleTrial} trialBrushLabel={trialBrushLabel}
        />
      )}

      {studies.length === 0 && !loading && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          No Optuna studies found. Check the storage URL or run an HPO sweep
          first.
        </div>
      )}
    </div>
  );
}
