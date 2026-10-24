/**
 * Process Monitor — tabular view of all spawned processes with inline log viewer (§G.15, Slice V5).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Terminal } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { useProcessRunLabelBrush } from "../../hooks/brush/useProcessRunLabelBrush";
import { PolicyTelemetryPanel } from "../../components/analysis/telemetry/PolicyTelemetryPanel";
import { PolicyTelemetryTrendsPanel } from "../../components/analysis/telemetry/PolicyTelemetryTrendsPanel";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useProcessStore } from "../../store/process";
import {
  collectPolicyVizFromLogLines,
  uniquePolicyVizPolicies,
} from "../../utils/benchmark/policyTelemetry";
import { collectAttentionVizFromLogLines } from "../../utils/graph/attentionViz";
import {
  extractJsonlPathFromLogLines,
  runLabelMapFromProcesses,
} from "../../utils/benchmark/policyTelemetryTrends";
import { collectTrainingHealthFromLogLines } from "../../utils/training/trainingHealth";
import { collectTrainingMetricsFromLogLines } from "../../utils/training/trainingMetrics";
import {
  isHpoProcess,
  isTrainOrHpoProcess,
  trainHpoLivePanelTitle,
} from "../../utils/training/trainingProcess";
import { EvalCheckpointLiveCard } from "../../components/monitor/eval/EvalCheckpointLiveCard";
import { EvalResultCard } from "../../components/monitor/eval/EvalResultCard";
import { LauncherLivePanel } from "../../components/monitor/live/LauncherLivePanel";
import { ProcessIdFooter } from "../../components/monitor/process/ProcessIdFooter";
import { ProcessRow } from "../../components/monitor/process/ProcessRow";
import { TrainHpoLivePanel } from "../../components/monitor/live/TrainHpoLivePanel";
import {
  dataGenLivePanelTitle,
  launcherKindFromProcess,
  simLivePanelTitle,
} from "../../utils/process/launcherProcess";
import {
  checkpointLabelFromEvalProcess,
  checkpointPathFromEvalCommand,
  collectEvalResultFromLogLines,
  evalLivePanelTitle,
  hasEvalMetrics,
  toEvalAnalyticsRows,
} from "../../utils/benchmark/evalResults";
import { dataExplorerPathFromGenData } from "../../utils/runs/genDataPath";
import {
  brushLogPathFromProcessLines,
  brushLogPathMapFromProcesses,
  outputRunPathFromLogLines,
} from "../../utils/runs/outputRunPath";
import { trainingRunPathFromLogLines } from "../../utils/training/trainingRunPath";

function isTrainProcess(command: string, id: string): boolean {
  return isTrainOrHpoProcess(id, command);
}

export function ProcessMonitor() {
  const processes = useProcessStore((s) => s.processes);
  const clearCompleted = useProcessStore((s) => s.clearCompleted);
  const { effectiveTheme: theme, projectRoot, setMode, setPendingEvalResults } = useAppStore();
  const {
    policy: activePolicy,
    runLabel: activeRunLabel,
    logScale,
    setPolicy,
  } = useGlobalFiltersStore();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [telemetryTrendsKey, setTelemetryTrendsKey] = useState(0);

  const ids = Object.keys(processes).sort(
    (a, b) => (processes[b].startTime ?? 0) - (processes[a].startTime ?? 0)
  );

  const running = ids.filter((id) => processes[id].status === "running").length;
  const completed = ids.length - running;

  useEffect(() => {
    if (selectedId && !processes[selectedId]) {
      setSelectedId(null);
    }
  }, [selectedId, processes]);

  const selectedProc = selectedId ? processes[selectedId] : null;
  const selectedLauncherKind = selectedProc
    ? launcherKindFromProcess(selectedProc.id, selectedProc.command)
    : null;
  const selectedIsSim = selectedLauncherKind === "sim";
  const selectedIsEval = selectedLauncherKind === "eval";
  const selectedIsTrain = selectedProc
    ? isTrainProcess(selectedProc.command, selectedProc.id)
    : false;

  const policyVizEntries = useMemo(
    () => (selectedProc ? collectPolicyVizFromLogLines(selectedProc.logLines) : []),
    [selectedProc]
  );

  const vizPolicies = useMemo(
    () => uniquePolicyVizPolicies(policyVizEntries),
    [policyVizEntries]
  );

  const selectedPolicy = activePolicy ?? vizPolicies[0] ?? null;

  const processRunLabel = useProcessRunLabelBrush(
    selectedId,
    selectedProc?.logLines
  );

  const processRunBrushById = useMemo(
    () => runLabelMapFromProcesses(processes, ids, projectRoot),
    [ids, processes, projectRoot]
  );

  const processLogPathById = useMemo(
    () => brushLogPathMapFromProcesses(processes, ids),
    [ids, processes]
  );

  useEffect(() => {
    if (policyVizEntries.length > 0) {
      setTelemetryTrendsKey((k) => k + 1);
    }
  }, [policyVizEntries.length, selectedProc?.logLines.length]);

  const policyVizLive = selectedProc?.status === "running";

  const trainingMetrics = useMemo(
    () => (selectedProc ? collectTrainingMetricsFromLogLines(selectedProc.logLines) : []),
    [selectedProc]
  );
  const latestTrainingMetric = trainingMetrics[trainingMetrics.length - 1];

  const trainingHealthEntries = useMemo(
    () => (selectedProc ? collectTrainingHealthFromLogLines(selectedProc.logLines) : []),
    [selectedProc]
  );

  const attentionEntries = useMemo(
    () => (selectedProc ? collectAttentionVizFromLogLines(selectedProc.logLines) : []),
    [selectedProc]
  );

  const evalCheckpointName = useMemo(() => {
    if (!selectedProc || !selectedIsEval) return "";
    return checkpointLabelFromEvalProcess(selectedProc.id, selectedProc.command);
  }, [selectedProc, selectedIsEval]);

  const evalResult = useMemo(() => {
    if (!selectedProc || !selectedIsEval) return null;
    const result = collectEvalResultFromLogLines(selectedProc.logLines, evalCheckpointName);
    const path = checkpointPathFromEvalCommand(selectedProc.command);
    if (path) result.checkpointPath = path;
    return result;
  }, [selectedProc, selectedIsEval, evalCheckpointName]);

  const evalCheckpointPath = useMemo(() => {
    if (!selectedProc || !selectedIsEval) return null;
    return checkpointPathFromEvalCommand(selectedProc.command);
  }, [selectedProc, selectedIsEval]);

  const launcherOutputRunPath = useMemo(() => {
    if (!selectedProc || !selectedLauncherKind) return null;
    return outputRunPathFromLogLines(selectedProc.logLines);
  }, [selectedProc, selectedLauncherKind]);

  const dataGenCsvPath = useMemo(() => {
    if (!selectedProc || selectedLauncherKind !== "data_gen") return null;
    return dataExplorerPathFromGenData(selectedProc.command, selectedProc.logLines);
  }, [selectedProc, selectedLauncherKind]);

  const trainOutputRunPath = useMemo(() => {
    if (!selectedProc || !selectedIsTrain) return null;
    return outputRunPathFromLogLines(selectedProc.logLines);
  }, [selectedProc, selectedIsTrain]);

  const trainRunPath = useMemo(() => {
    if (!selectedProc || !selectedIsTrain) return null;
    return trainingRunPathFromLogLines(selectedProc.logLines);
  }, [selectedProc, selectedIsTrain]);

  const processLogPath = useMemo(() => {
    if (!selectedProc) return null;
    const kind = selectedIsTrain ? "train" : "sim";
    return brushLogPathFromProcessLines(selectedProc.logLines, kind);
  }, [selectedProc, selectedIsTrain]);

  const simJsonlPath = useMemo(() => {
    if (!selectedProc || !selectedIsSim) return null;
    return extractJsonlPathFromLogLines(selectedProc.logLines);
  }, [selectedProc, selectedIsSim]);

  const openEvalInAnalytics = useCallback(() => {
    if (!evalResult || !hasEvalMetrics(evalResult)) return;
    setPendingEvalResults(toEvalAnalyticsRows([evalResult]));
    setMode("benchmark");
  }, [evalResult, setMode, setPendingEvalResults]);

  if (ids.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-canvas-muted gap-2">
        <Terminal size={28} className="opacity-30" />
        <p className="text-sm">No processes yet.</p>
        <p className="text-xs">
          Launch a simulation, training run, or data generation to see it here.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <GlobalFilterBar
        policies={vizPolicies.length > 0 ? vizPolicies : undefined}
        runLabels={processRunLabel ? [processRunLabel] : []}
        showLogScale
      />

      <div className="flex items-center gap-3">
        <p className="text-sm text-canvas-muted tabular-nums">
          {ids.length} process{ids.length !== 1 ? "es" : ""}
          {running > 0 && (
            <span className="ml-2 text-accent-success">· {running} running</span>
          )}
        </p>
        {completed > 0 && (
          <button
            onClick={clearCompleted}
            className="btn-ghost text-xs text-canvas-muted ml-auto"
          >
            Clear completed ({completed})
          </button>
        )}
      </div>

      {ids.map((id) => (
        <ProcessRow
          key={id}
          id={id}
          selected={selectedId === id}
          runBrushActive={
            Boolean(activeRunLabel) && processRunBrushById[id] === activeRunLabel
          }
          logPath={processLogPathById[id]}
          brushLabel={processRunBrushById[id]}
          projectRoot={projectRoot}
          onSelect={() => setSelectedId((prev) => (prev === id ? null : id))}
        />
      ))}

      {selectedIsSim && selectedProc && (
        <LauncherLivePanel
          variant="embedded"
          header={{
            status: selectedProc.status,
            title: simLivePanelTitle({
              isRunning: selectedProc.status === "running",
              status: selectedProc.status,
            }),
            runLabel: processRunLabel,
            logPath: processLogPath,
            projectRoot,
            navMesh: {
              kind: "sim",
              showPostRun: selectedProc.status === "completed",
              showOutputBrowser: selectedProc.status === "completed",
              outputRunPath: launcherOutputRunPath,
              simLogPath: simJsonlPath,
            },
          }}
          footer={
            <ProcessIdFooter
              processId={selectedProc.id}
              logPath={processLogPath}
              runLabel={processRunLabel}
              projectRoot={projectRoot}
            />
          }
          logLines={selectedProc.logLines}
          logTailWaiting={selectedProc.status === "running"}
        >
          {vizPolicies.length > 1 && (
            <div className="flex flex-wrap gap-1.5">
              {vizPolicies.map((pol) => (
                <button
                  key={pol}
                  onClick={() => setPolicy(activePolicy === pol ? null : pol)}
                  className={`text-[10px] font-mono px-2 py-0.5 rounded border transition-colors ${
                    selectedPolicy === pol
                      ? "border-accent-secondary text-accent-secondary bg-accent-secondary/10"
                      : "border-canvas-border text-gray-400 hover:text-gray-200"
                  }`}
                >
                  {pol}
                </button>
              ))}
            </div>
          )}

          <PolicyTelemetryPanel
            entries={policyVizEntries}
            policy={selectedPolicy}
            sampleId={null}
            day={null}
            theme={theme}
            logScale={logScale}
            live={policyVizLive}
          />

          <PolicyTelemetryTrendsPanel
            theme={theme}
            logScale={logScale}
            refreshKey={telemetryTrendsKey}
            initialPolicy={selectedPolicy}
            initialRunLabel={processRunLabel}
          />
        </LauncherLivePanel>
      )}

      {selectedIsEval && selectedProc && (
        <LauncherLivePanel
          variant="embedded"
          header={{
            status: selectedProc.status,
            title: evalLivePanelTitle({
              isRunning: selectedProc.status === "running",
              status: selectedProc.status,
            }),
            runLabel: processRunLabel,
            logPath: processLogPath,
            projectRoot,
            navMesh: {
              kind: "eval",
              showPostRun: selectedProc.status === "completed",
              showOutputBrowser: selectedProc.status === "completed",
              outputRunPath: launcherOutputRunPath,
              checkpointPath: evalCheckpointPath,
              onOpenAnalytics:
                evalResult && hasEvalMetrics(evalResult) ? openEvalInAnalytics : undefined,
            },
          }}
          footer={
            <ProcessIdFooter
              processId={selectedProc.id}
              logPath={processLogPath}
              runLabel={processRunLabel}
              projectRoot={projectRoot}
            />
          }
          logLines={selectedProc.logLines}
          logTailWaiting={selectedProc.status === "running"}
        >
          {selectedProc.status === "running" ||
          !evalResult ||
          !hasEvalMetrics(evalResult) ? (
            <EvalCheckpointLiveCard
              procId={selectedProc.id}
              checkpointName={evalCheckpointName}
              checkpointPath={evalCheckpointPath}
              projectRoot={projectRoot}
              status={selectedProc.status}
              isRunning={selectedProc.status === "running"}
              result={evalResult ?? undefined}
              logLines={selectedProc.logLines}
              showLogTail={false}
            />
          ) : (
            <EvalResultCard
              result={evalResult}
              projectRoot={projectRoot}
              onOpenAnalytics={openEvalInAnalytics}
            />
          )}
        </LauncherLivePanel>
      )}

      {selectedLauncherKind === "data_gen" && selectedProc && (
        <LauncherLivePanel
          variant="embedded"
          header={{
            status: selectedProc.status,
            title: dataGenLivePanelTitle({
              isRunning: selectedProc.status === "running",
              status: selectedProc.status,
            }),
            runLabel: processRunLabel,
            logPath: processLogPath,
            projectRoot,
            navMesh: {
              kind: "data_gen",
              showPostRun: selectedProc.status === "completed",
              showOutputBrowser: selectedProc.status === "completed",
              outputRunPath: launcherOutputRunPath,
              csvPath: dataGenCsvPath,
            },
          }}
          footer={
            <ProcessIdFooter
              processId={selectedProc.id}
              logPath={processLogPath}
              runLabel={processRunLabel}
              projectRoot={projectRoot}
            />
          }
          logLines={selectedProc.logLines}
          logTailWaiting={selectedProc.status === "running"}
        />
      )}

      {selectedIsTrain && selectedProc && (
        <TrainHpoLivePanel
          variant="embedded"
          header={{
            status: selectedProc.status,
            title: trainHpoLivePanelTitle({
              isRunning: selectedProc.status === "running",
              status: selectedProc.status,
              processId: selectedProc.id,
              command: selectedProc.command,
            }),
            runLabel: processRunLabel,
            logPath: processLogPath,
            projectRoot,
            metricCount: trainingMetrics.length,
            healthCount: trainingHealthEntries.length,
            attentionCount: attentionEntries.length,
            navMesh: {
              showHpoLinks: isHpoProcess(selectedProc.id, selectedProc.command),
              showOutputBrowser: selectedProc.status === "completed",
              outputRunPath: trainOutputRunPath,
              trainingRunPath: trainRunPath,
            },
          }}
          progress={
            selectedProc.status === "running"
              ? {
                  processId: selectedProc.id,
                  fallbackValue: latestTrainingMetric?.epoch,
                }
              : undefined
          }
          analytics={{
            metrics: trainingMetrics,
            healthEntries: trainingHealthEntries,
            attentionEntries: attentionEntries,
            logScale,
            theme,
            exportNamePrefix: "process-monitor",
            isPostRun: selectedProc.status !== "running",
            postRunFallback:
              "Post-run shortcuts — open Training Monitor or Output Browser for this run",
          }}
          logLines={selectedProc.logLines}
          logTailWaiting={selectedProc.status === "running"}
          footer={
            <ProcessIdFooter
              processId={selectedProc.id}
              logPath={processLogPath}
              runLabel={processRunLabel}
              projectRoot={projectRoot}
            />
          }
        />
      )}
    </div>
  );
}
