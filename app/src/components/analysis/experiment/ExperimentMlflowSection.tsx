import { useRef } from "react";
import ReactECharts from "echarts-for-react";
import { Download, ExternalLink, RefreshCw } from "lucide-react";
import { open as openUrl } from "@tauri-apps/plugin-shell";
import { LoadedRunRow } from "../../common/LoadedRunRow";
import { OpenPathToolbar } from "../../common/OpenPathToolbar";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import { downloadCsv } from "../../../utils/charts/tableExport";
import {
  mlflowRunDirFromArtifactUri,
  resolveLocalProjectPath,
} from "../../../utils/runs/outputRunPath";
import type { MlflowRun, OutputDir } from "../../../types";

export type MlflowView = "runs" | "dashboard";

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 ** 2).toFixed(1)} MB`;
}

function formatTime(ms: number | null) {
  if (ms == null) return "—";
  return new Date(ms).toLocaleString();
}

export function ExperimentMlflowSection({
  projectRoot,
  mlflowView,
  setMlflowView,
  mlflowUiUrl,
  setMlflowUiUrl,
  trackingUri,
  setTrackingUri,
  trackingUriPath,
  experimentName,
  setExperimentName,
  refreshRuns,
  loading,
  runs,
  error,
  selectedRunIds,
  toggleRun,
  mlflowRunBrushLabel,
  activeRunLabel,
  metricKeys,
  selectedMetric,
  setSelectedMetric,
  normalizeY,
  setNormalizeY,
  useMetricLogScale,
  comparisonOption,
}: {
  projectRoot: string;
  mlflowView: MlflowView;
  setMlflowView: (v: MlflowView) => void;
  mlflowUiUrl: string;
  setMlflowUiUrl: (url: string) => void;
  trackingUri: string;
  setTrackingUri: (uri: string) => void;
  trackingUriPath: string | null;
  experimentName: string;
  setExperimentName: (name: string) => void;
  refreshRuns: () => void;
  loading: boolean;
  runs: MlflowRun[];
  error: string | null;
  selectedRunIds: string[];
  toggleRun: (runId: string) => void;
  mlflowRunBrushLabel: (run: MlflowRun) => string;
  activeRunLabel: string | null;
  metricKeys: string[];
  selectedMetric: string;
  setSelectedMetric: (m: string) => void;
  normalizeY: boolean;
  setNormalizeY: (v: boolean) => void;
  useMetricLogScale: boolean;
  comparisonOption: object;
}) {
  const chartRef = useRef<ReactECharts>(null);

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h2 className="text-sm font-semibold text-gray-200">MLflow Tracking</h2>
        <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
          {(["runs", "dashboard"] as const).map((v) => (
            <button
              key={v}
              onClick={() => setMlflowView(v)}
              className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                mlflowView === v
                  ? "bg-accent-primary text-white"
                  : "text-canvas-muted hover:text-gray-200"
              }`}
            >
              {v === "runs" ? "Runs" : "Dashboard"}
            </button>
          ))}
        </div>
      </div>

      {mlflowView === "dashboard" && (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2 items-center">
            <input
              className="input-base font-mono text-xs flex-1 min-w-[200px]"
              value={mlflowUiUrl}
              onChange={(e) => setMlflowUiUrl(e.target.value)}
              placeholder="http://localhost:5000"
            />
            <button
              onClick={() => openUrl(mlflowUiUrl).catch(() => {})}
              className="btn-ghost text-xs flex items-center gap-1"
            >
              <ExternalLink size={12} /> Open in browser
            </button>
          </div>
          <iframe
            title="MLflow Dashboard"
            src={mlflowUiUrl}
            className="w-full h-[480px] rounded-lg border border-canvas-border bg-white"
            sandbox="allow-scripts allow-same-origin allow-forms"
          />
          <p className="text-[10px] text-canvas-muted">
            Embedded MLflow UI fallback (§G.18). Start the MLflow server locally
            to use this view.
          </p>
        </div>
      )}

      {mlflowView === "runs" && (
        <>
          <div className="flex flex-wrap gap-3 items-end">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-canvas-muted">Tracking URI</label>
              <input
                className="input-base font-mono text-xs w-48"
                value={trackingUri}
                onChange={(e) => setTrackingUri(e.target.value)}
                placeholder="mlruns"
              />
              {trackingUriPath ? (
                <OpenPathToolbar
                  path={trackingUriPath}
                  chipClassName="max-w-full"
                  handoff={false}
                />
              ) : null}
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-canvas-muted">Experiment</label>
              <input
                className="input-base font-mono text-xs w-40"
                value={experimentName}
                onChange={(e) => setExperimentName(e.target.value)}
                placeholder="wsmart-route"
              />
            </div>
            <button
              onClick={refreshRuns}
              disabled={loading}
              className="btn-primary flex items-center gap-2"
            >
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
              Refresh
            </button>
            <span className="text-xs text-canvas-muted">{runs.length} run(s)</span>
            {runs.length > 0 && (
              <button
                onClick={() =>
                  downloadCsv(
                    "mlflow-runs.csv",
                    ["run_id", "run_name", "status", "start_time"],
                    runs.map((r) => [
                      r.run_id,
                      r.run_name,
                      r.status,
                      r.start_time != null
                        ? new Date(r.start_time).toISOString()
                        : "",
                    ])
                  )
                }
                className="btn-ghost text-xs flex items-center gap-1"
              >
                <Download size={12} /> Export CSV
              </button>
            )}
          </div>
          {error && <p className="text-xs text-accent-danger">{error}</p>}

          <div className="card overflow-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-canvas-border">
                  <th className="py-2 px-2 text-canvas-muted font-medium w-8" />
                  <th className="text-left py-2 px-2 text-canvas-muted font-medium">
                    Run
                  </th>
                  <th className="text-left py-2 px-2 text-canvas-muted font-medium">
                    Status
                  </th>
                  <th className="text-left py-2 px-2 text-canvas-muted font-medium">
                    Started
                  </th>
                  <th className="text-left py-2 px-2 text-canvas-muted font-medium">
                    Metrics
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-canvas-border">
                {runs.map((r) => {
                  const rawRunDir = mlflowRunDirFromArtifactUri(r.artifact_uri);
                  const runDir = rawRunDir
                    ? resolveLocalProjectPath(rawRunDir, projectRoot) ?? rawRunDir
                    : null;
                  const brushLabel = mlflowRunBrushLabel(r);
                  const brushActive = Boolean(
                    activeRunLabel && activeRunLabel === brushLabel
                  );
                  return (
                    <tr
                      key={r.run_id}
                      className={`hover:bg-canvas-hover cursor-pointer ${
                        selectedRunIds.includes(r.run_id) || brushActive
                          ? "bg-canvas-hover/60"
                          : ""
                      } ${
                        brushActive
                          ? "ring-1 ring-inset ring-accent-secondary/30"
                          : ""
                      }`}
                      onClick={() => toggleRun(r.run_id)}
                    >
                      <td className="py-2 px-2">
                        <input
                          type="checkbox"
                          className="accent-accent-primary"
                          checked={selectedRunIds.includes(r.run_id)}
                          onChange={() => toggleRun(r.run_id)}
                          onClick={(e) => e.stopPropagation()}
                        />
                      </td>
                      <td className="py-2 px-2">
                        {runDir ? (
                          <OpenPathToolbar
                            path={runDir}
                            kind="training"
                            label={brushLabel}
                            brushLabel={brushLabel}
                            storedLabel={brushLabel}
                            chipClassName="flex-1 min-w-0 max-w-none"
                            className="flex-1 min-w-0"
                          >
                            <span className="text-canvas-muted font-mono text-[10px] shrink-0 truncate max-w-[6rem]">
                              {r.run_id.slice(0, 12)}…
                            </span>
                          </OpenPathToolbar>
                        ) : (
                          <>
                            <div className="font-mono text-gray-300">
                              {brushLabel}
                            </div>
                            <div className="text-canvas-muted font-mono text-[10px]">
                              {r.run_id.slice(0, 12)}…
                            </div>
                          </>
                        )}
                      </td>
                      <td className="py-2 px-2 text-canvas-muted">{r.status}</td>
                      <td className="py-2 px-2 text-canvas-muted">
                        {formatTime(r.start_time)}
                      </td>
                      <td className="py-2 px-2 text-canvas-muted font-mono tabular-nums">
                        {Object.entries(r.metrics)
                          .slice(0, 3)
                          .map(([k, v]) => `${k}=${v.toFixed(3)}`)
                          .join(", ") || "—"}
                      </td>
                    </tr>
                  );
                })}
                {runs.length === 0 && !loading && (
                  <tr>
                    <td colSpan={5} className="py-8 text-center text-canvas-muted">
                      No MLflow runs found. Enable MLflow tracking and run
                      training to populate this view.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {selectedRunIds.length > 0 && metricKeys.length > 0 && (
            <div className="card space-y-3">
              <div className="flex flex-wrap items-center gap-3">
                <h3 className="text-sm font-semibold text-gray-200">
                  Metric Comparison ({selectedRunIds.length} run
                  {selectedRunIds.length > 1 ? "s" : ""})
                </h3>
                <select
                  className="select-base text-xs w-48"
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                >
                  {metricKeys.map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
                <label className="flex items-center gap-1.5 text-xs text-canvas-muted cursor-pointer">
                  <input
                    type="checkbox"
                    className="accent-accent-primary"
                    checked={normalizeY}
                    onChange={(e) => setNormalizeY(e.target.checked)}
                  />
                  Normalize Y-axis
                </label>
                <ChartExportButtons
                  chartRef={chartRef}
                  filenameStem={`mlflow-${selectedMetric}`}
                  className="ml-auto"
                />
              </div>
              <p className="text-[10px] text-canvas-muted">
                {useMetricLogScale
                  ? "Log-scale y-axis (disabled when Normalize is on)"
                  : "Linear y-axis"}
              </p>
              <ReactECharts
                ref={chartRef}
                option={comparisonOption}
                style={{ height: 320 }}
              />
            </div>
          )}

          {selectedRunIds.length === 1 && (
            <div className="card space-y-2">
              <h3 className="text-sm font-semibold text-gray-200">Parameters</h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                {Object.entries(
                  runs.find((r) => r.run_id === selectedRunIds[0])?.params ?? {}
                ).map(([k, v]) => (
                  <div key={k} className="flex gap-2">
                    <span className="text-canvas-muted shrink-0">{k}</span>
                    <span className="font-mono text-gray-300 truncate">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function ExperimentOutputDirsSection({
  outputDirs,
  projectRoot,
  activeRunLabel,
}: {
  outputDirs: OutputDir[];
  projectRoot: string;
  activeRunLabel: string | null;
}) {
  return (
    <div className="card space-y-2">
      <h3 className="text-sm font-semibold text-gray-200">Output Directories</h3>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-canvas-border">
            <th className="text-left py-2 px-3 text-canvas-muted font-medium">
              Run Name
            </th>
            <th className="text-left py-2 px-3 text-canvas-muted font-medium">
              Created
            </th>
            <th className="text-right py-2 px-3 text-canvas-muted font-medium">
              Size
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-canvas-border">
          {outputDirs.slice(0, 10).map((d) => (
            <tr key={d.path} className="hover:bg-canvas-hover">
              <td className="py-2 px-3">
                <LoadedRunRow
                  path={d.path}
                  projectRoot={projectRoot}
                  label={d.name}
                  activeRunLabel={activeRunLabel}
                  pathHandoffs
                  handoffKind="run"
                />
              </td>
              <td className="py-2 px-3 text-canvas-muted">
                {new Date(d.created_at).toLocaleString()}
              </td>
              <td className="py-2 px-3 text-right text-canvas-muted tabular-nums">
                {formatBytes(d.size_bytes)}
              </td>
            </tr>
          ))}
          {outputDirs.length === 0 && (
            <tr>
              <td colSpan={3} className="py-4 text-center text-canvas-muted">
                No output directories found.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
