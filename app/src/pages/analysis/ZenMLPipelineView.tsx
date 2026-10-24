/**
 * ZenML pipeline run browser with step-duration Gantt chart (§G.18).
 * Ports Streamlit `experiment_tracker_zenml.py`.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { Download, RefreshCw } from "lucide-react";
import { useAppStore } from "../../store/app";
import {
  chartMetricDisplay,
  chartMetricYAxisType,
} from "../../utils/charts/chartLogScale";
import { ChartExportButtons } from "../../components/common/ChartExportButtons";
import { downloadCsv } from "../../utils/charts/tableExport";
import {
  OrionConstellationGraph,
  type ConstellationNode,
  type ConstellationLink,
} from "../../components/ui/orion";
import type { ZenmlPipelineRun, ZenmlPipelineStep } from "../../types";

const STEP_COLORS: Record<string, string> = {
  completed: "#34d399",
  failed: "#f87171",
  running: "#6366f1",
  cached: "#fbbf24",
};

function stepColor(status: string): string {
  const key = status.toLowerCase();
  for (const [k, c] of Object.entries(STEP_COLORS)) {
    if (key.includes(k)) return c;
  }
  return "#818cf8";
}

export function ZenMLPipelineView({ logScale = false }: { logScale?: boolean }) {
  const { projectRoot, pythonPath } = useAppStore();
  const [runs, setRuns] = useState<ZenmlPipelineRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [steps, setSteps] = useState<ZenmlPipelineStep[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chartRef = useRef<ReactECharts>(null);

  const refreshRuns = useCallback(async () => {
    if (!projectRoot) return;
    setLoading(true);
    setError(null);
    try {
      const found = await invoke<ZenmlPipelineRun[]>("list_zenml_pipeline_runs", {
        projectRoot,
        pythonExecutable: pythonPath || null,
      });
      setRuns(found);
      if (found.length > 0 && !selectedRunId) {
        setSelectedRunId(found[0].id);
      }
    } catch (e) {
      setError(String(e));
      setRuns([]);
    } finally {
      setLoading(false);
    }
  }, [projectRoot, pythonPath, selectedRunId]);

  useEffect(() => {
    if (projectRoot) refreshRuns();
  }, [projectRoot, refreshRuns]);

  useEffect(() => {
    if (!projectRoot || !selectedRunId) {
      setSteps([]);
      return;
    }
    invoke<ZenmlPipelineStep[]>("load_zenml_run_steps", {
      runId: selectedRunId,
      projectRoot,
      pythonExecutable: pythonPath || null,
    })
      .then(setSteps)
      .catch(() => setSteps([]));
  }, [projectRoot, pythonPath, selectedRunId]);

  const { dagNodes, dagLinks } = useMemo(() => {
    if (steps.length === 0) return { dagNodes: [], dagLinks: [] };
    const nodes: ConstellationNode[] = [
      {
        id: "pipeline-root",
        label: "DAG Root",
        tier: 0,
        group: "Pipeline Execution",
        status: "nominal",
        tone: "cyan",
        details: `Active ZenML pipeline run: ${selectedRunId?.slice(0, 8)}`,
      },
    ];
    const links: ConstellationLink[] = [];

    steps.forEach((s, idx) => {
      const isComplete = s.status.toLowerCase().includes("complete");
      const isFailed = s.status.toLowerCase().includes("fail");
      const tone = isComplete ? "ok" : isFailed ? "crit" : "amber";

      nodes.push({
        id: s.name,
        label: s.name,
        tier: 1,
        group: "Step",
        status: isComplete ? "nominal" : "warn",
        tone,
        details: `Step duration: ${s.duration_seconds != null ? `${s.duration_seconds.toFixed(2)}s` : "—"}`,
        metrics: {
          Status: s.status,
          Duration: s.duration_seconds != null ? `${s.duration_seconds.toFixed(1)}s` : "—",
        },
      });

      if (idx === 0) {
        links.push({
          source: "pipeline-root",
          target: s.name,
          active: true,
          tone: "cyan",
        });
      } else {
        links.push({
          source: steps[idx - 1].name,
          target: s.name,
          active: true,
          tone: "indigo",
        });
      }
    });

    return { dagNodes: nodes, dagLinks: links };
  }, [steps, selectedRunId]);

  const ganttOption = useMemo(() => {
    const withDuration = steps.filter((s) => s.duration_seconds != null && s.duration_seconds > 0);
    if (withDuration.length === 0) return null;

    const names = withDuration.map((s) => s.name);
    const durations = withDuration.map((s) => s.duration_seconds as number);
    const colors = withDuration.map((s) => stepColor(s.status));

    const durationKey = "duration_seconds";
    return {
      backgroundColor: "transparent",
      grid: { left: 120, right: 20, top: 10, bottom: 30 },
      xAxis: {
        type: chartMetricYAxisType(durationKey, logScale),
        logBase: 10,
        name: logScale ? "Duration (s, log)" : "Duration (s)",
        axisLabel: { color: "#9090b0", fontSize: 10 },
        minorSplitLine: { show: false },
      },
      yAxis: {
        type: "category",
        data: names,
        axisLabel: { color: "#9090b0", fontSize: 10 },
      },
      series: [
        {
          type: "bar",
          data: durations.map((d, i) => ({
            value: chartMetricDisplay(d, durationKey, logScale) ?? d,
            rawSeconds: d,
            itemStyle: { color: colors[i] },
          })),
          barMaxWidth: 18,
        },
      ],
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: (params: unknown) => {
          const p = (params as Array<{ name: string; data: { rawSeconds?: number }; value: number }>)[0];
          const secs = p?.data?.rawSeconds ?? p?.value;
          return p ? `${p.name}: ${secs.toFixed(2)}s` : "";
        },
      },
    };
  }, [steps, logScale]);

  const exportRunsCsv = () => {
    downloadCsv(
      "zenml-pipeline-runs.csv",
      ["id", "pipeline", "status", "stack", "created", "updated"],
      runs.map((r) => [r.id, r.pipeline, r.status, r.stack, r.created, r.updated])
    );
  };

  if (!projectRoot) return null;

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h2 className="text-sm font-semibold text-gray-200">ZenML Pipeline Runs</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={refreshRuns}
            disabled={loading}
            className="btn-primary flex items-center gap-2 text-xs"
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
          {runs.length > 0 && (
            <button onClick={exportRunsCsv} className="btn-ghost text-xs flex items-center gap-1">
              <Download size={12} />
              Export CSV
            </button>
          )}
        </div>
      </div>

      {error && <p className="text-xs text-accent-danger">{error}</p>}

      {runs.length === 0 && !loading && !error && (
        <p className="text-xs text-canvas-muted">
          No ZenML pipeline runs found. Enable ZenML tracking and run a pipeline to populate this view.
        </p>
      )}

      {runs.length > 0 && (
        <>
          <div className="overflow-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-canvas-border">
                  <th className="text-left py-2 px-2 text-canvas-muted font-medium">ID</th>
                  <th className="text-left py-2 px-2 text-canvas-muted font-medium">Pipeline</th>
                  <th className="text-left py-2 px-2 text-canvas-muted font-medium">Status</th>
                  <th className="text-left py-2 px-2 text-canvas-muted font-medium">Stack</th>
                  <th className="text-left py-2 px-2 text-canvas-muted font-medium">Created</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-canvas-border">
                {runs.map((r) => (
                  <tr
                    key={r.id}
                    className={`hover:bg-canvas-hover cursor-pointer ${
                      selectedRunId === r.id ? "bg-canvas-hover/60" : ""
                    }`}
                    onClick={() => setSelectedRunId(r.id)}
                  >
                    <td className="py-2 px-2 font-mono text-gray-300">{r.id.slice(0, 8)}</td>
                    <td className="py-2 px-2 text-gray-300">{r.pipeline}</td>
                    <td className="py-2 px-2 text-canvas-muted">{r.status}</td>
                    <td className="py-2 px-2 text-canvas-muted">{r.stack}</td>
                    <td className="py-2 px-2 text-canvas-muted">{r.created}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {selectedRunId && dagNodes.length > 0 && (
            <OrionConstellationGraph
              nodes={dagNodes}
              links={dagLinks}
              title={`Pipeline DAG Topology: ${runs.find((r) => r.id === selectedRunId)?.pipeline ?? selectedRunId.slice(0, 8)}`}
              subtitle="Step execution status & DAG dependency flow"
              height={320}
            />
          )}

          {selectedRunId && steps.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-xs text-canvas-muted">
                  Step durations — run {selectedRunId.slice(0, 8)}
                </p>
                {ganttOption && (
                  <ChartExportButtons chartRef={chartRef} filenameStem="zenml-steps" />
                )}
              </div>
              {ganttOption ? (
                <ReactECharts ref={chartRef} option={ganttOption} style={{ height: Math.max(160, steps.length * 28) }} />
              ) : (
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-canvas-border">
                      <th className="text-left py-1 px-2 text-canvas-muted font-medium">Step</th>
                      <th className="text-left py-1 px-2 text-canvas-muted font-medium">Status</th>
                      <th className="text-right py-1 px-2 text-canvas-muted font-medium">Duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {steps.map((s) => (
                      <tr key={s.name} className="border-b border-canvas-border/30">
                        <td className="py-1 px-2 font-mono text-gray-300">{s.name}</td>
                        <td className="py-1 px-2 text-canvas-muted">{s.status}</td>
                        <td className="py-1 px-2 text-right font-mono text-canvas-muted">
                          {s.duration_seconds != null ? `${s.duration_seconds.toFixed(2)}s` : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
