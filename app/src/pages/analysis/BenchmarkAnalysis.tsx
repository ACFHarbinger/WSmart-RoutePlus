/**
 * Benchmark Analysis — multi-run, multi-policy comparison.
 * Ports Streamlit `benchmark` mode.
 *
 * Also consumes `pendingEvalResults` from EvaluationRunner (§G.12) to display
 * checkpoint comparison charts without requiring simulation log files.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, X, Download } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { usePortfolioRunBrush } from "../../hooks/brush/usePortfolioRunBrush";
import { useAppStore } from "../../store/app";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { useGlobalFiltersStore } from "../../store/filters";
import { filterEntries } from "../../store/sim";
import { PortfolioEfficiencyRanking } from "../../components/analysis/benchmark/PortfolioEfficiencyRanking";
import { barOpacity } from "../../utils/charts/chartHighlight";
import { errorBarBounds, groupedBarWhiskerX } from "../../utils/charts/chartLogScale";
import { ChartExportButtons } from "../../components/common/ChartExportButtons";
import { symlog } from "../../utils/charts/symlog";
import { PARETO_PANELS } from "../../utils/benchmark/paretoPanels";
import { buildParetoByPanel } from "../../utils/benchmark/paretoPortfolio";
import { BenchmarkParetoPanel } from "../../components/analysis/benchmark/BenchmarkParetoPanel";
import { BenchmarkGraphHeatmap } from "../../components/analysis/benchmark/BenchmarkGraphHeatmap";
import type { HeatmapMode } from "../../utils/benchmark/heatmapMetrics";
import { BenchmarkPortfolioParallel } from "../../components/analysis/benchmark/BenchmarkPortfolioParallel";
import {
  buildCityComparisonSeries,
  cityComparisonChartOption,
  groupRunsByCity,
} from "../../utils/benchmark/cityComparison";
import {
  loadPortfolioLogs,
  PORTFOLIO_SCAN_DEFAULT,
  scanOutputPortfolio,
} from "../../utils/runs/outputRunLogs";
import { downloadCsv } from "../../utils/charts/tableExport";
import {
  formatPipelineTimingBadge,
  portfolioRunLabel,
  runPortfolioSimulationArrowPipeline,
} from "../../utils/duckdb/arrowPipeline";
import { PolicyTelemetryTrendsPanel } from "../../components/analysis/telemetry/PolicyTelemetryTrendsPanel";
import { SqlQueryPanel } from "../../components/analysis/explorer/SqlQueryPanel";
import { LoadedRunRow } from "../../components/common/LoadedRunRow";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { parentRunBrushLabelFromCheckpointPath } from "../../utils/training/checkpoints";
import { useDuckDbStore } from "../../store/duckdb";
import { toast } from "sonner";
import type { DayLogEntry, EvalAnalyticsRow } from "../../types";
import { simOutputRoots } from "../../utils/runs/outputRoots";

const BENCHMARK_SIM_TABLE = "benchmark_sim";

interface RunFile {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function std(arr: number[]) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}

function fmt(n: number, d = 2) {
  return Number.isFinite(n) ? n.toFixed(d) : "—";
}

const SIM_METRICS = [
  { key: "profit", label: "Profit (€)", lowerIsBetter: false },
  { key: "km", label: "Distance (km)", lowerIsBetter: true },
  { key: "overflows", label: "Overflows", lowerIsBetter: true },
  { key: "kg/km", label: "Efficiency (kg/km)", lowerIsBetter: false },
  { key: "cost", label: "Cost (€)", lowerIsBetter: true },
];

const EVAL_METRICS = [
  { key: "cost", label: "Tour Cost", lowerIsBetter: true },
  { key: "gap", label: "Optimality Gap (%)", lowerIsBetter: true },
  { key: "time", label: "Time (s)", lowerIsBetter: true },
];

const COLORS = ["#6366f1", "#34d399", "#fbbf24", "#f87171", "#818cf8", "#a3e635"];

function EvalResultsPanel({
  rows,
  logScale,
  projectRoot,
  onDismiss,
}: {
  rows: EvalAnalyticsRow[];
  logScale: boolean;
  projectRoot?: string | null;
  onDismiss: () => void;
}) {
  const chartRefs = useRef<Record<string, EChartsReact | null>>({});
  const checkpoints = rows.map((r) => r.checkpoint);

  const makeBarOption = (metricKey: string, metricLabel: string) => ({
    backgroundColor: "transparent",
    grid: { left: 50, right: 10, top: 20, bottom: 55 },
    xAxis: {
      type: "category",
      data: checkpoints,
      axisLabel: { color: "#9090b0", fontSize: 9, rotate: 25 },
    },
    yAxis: {
      type: logScale ? "log" : "value",
      logBase: 10,
      name: metricLabel,
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 10 },
      minorSplitLine: { show: false },
    },
    series: [
      {
        type: "bar",
        data: rows.map((r) => {
          const v = (r[metricKey] as number | undefined) ?? 0;
          return logScale ? Math.max(v, 0.001) : v;
        }),
        itemStyle: {
          color: (params: { dataIndex: number }) => COLORS[params.dataIndex % COLORS.length],
        },
      },
    ],
    tooltip: { trigger: "axis" },
  });

  return (
    <div className="space-y-4">
      <div className="card flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-gray-200">Eval Results — Checkpoint Comparison</p>
          <p className="text-xs text-canvas-muted mt-0.5">
            {rows.length} checkpoint(s) loaded from Evaluation Runner
            {logScale ? " · log-scale bars" : " · linear bars"}
          </p>
        </div>
        <button onClick={onDismiss} className="btn-ghost text-xs flex items-center gap-1">
          <X size={12} />
          Dismiss
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {EVAL_METRICS.map(({ key, label }) => (
          <div key={key} className="card">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-canvas-muted">{label}</p>
              <ChartExportButtons
                chartRef={{ current: chartRefs.current[key] }}
                filenameStem={`eval-${key}`}
              />
            </div>
            <ReactECharts
              ref={(el) => {
                chartRefs.current[key] = el;
              }}
              option={makeBarOption(key, label)}
              style={{ height: 220 }}
            />
          </div>
        ))}
      </div>

      <div className="card overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-canvas-border">
              <th className="text-left py-2 px-3 text-canvas-muted font-medium">Checkpoint</th>
              {EVAL_METRICS.map(({ key, label }) => (
                <th key={key} className="text-right py-2 px-3 text-canvas-muted font-medium">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-canvas-border/30">
            {rows.map((r) => (
              <tr key={r.checkpoint} className="hover:bg-canvas-hover/40">
                <td className="py-1.5 px-3 max-w-[240px]">
                  {r.checkpointPath ? (
                    <OpenPathToolbar
                      path={r.checkpointPath}
                      projectRoot={projectRoot}
                      kind="checkpoint"
                      label={r.checkpoint}
                      brushLabel={parentRunBrushLabelFromCheckpointPath(
                        r.checkpointPath,
                        projectRoot
                      )}
                      chipClassName="max-w-full"
                      className="max-w-full min-w-0"
                    />
                  ) : (
                    <span className="font-mono text-gray-300">{r.checkpoint}</span>
                  )}
                </td>
                {EVAL_METRICS.map(({ key }) => (
                  <td key={key} className="py-1.5 px-3 text-right font-mono text-gray-400">
                    {typeof r[key] === "number" ? (r[key] as number).toFixed(4) : "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function BenchmarkAnalysis() {
  const chartRefs = useRef<Record<string, EChartsReact | null>>({});
  const {
    pendingEvalResults, setPendingEvalResults,
    pendingBenchmarkLogs, setPendingBenchmarkLogs,
    effectiveTheme: theme,
  } = useAppStore();
  const { projectRoot, handoff } = useRecentHandoff();
  const {
    ready: duckdbReady,
    loading: duckdbLoading,
    lastPipeline,
    setLastPipeline,
    setLoading: setDuckdbLoading,
  } = useDuckDbStore();
  const [runs, setRuns] = useState<RunFile[]>([]);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const [evalRows, setEvalRows] = useState<EvalAnalyticsRow[] | null>(null);
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const [showErrorBars, setShowErrorBars] = useState(false);
  const [heatmapMode, setHeatmapMode] = useState<HeatmapMode>("all");
  const { policy, sampleId, setPolicy } = useGlobalFiltersStore();
  const brushedPolicies = useMemo(() => (policy ? [policy] : null), [policy]);

  const normalizedRuns = useMemo(
    () =>
      runs.map((r) => ({
        ...r,
        label: portfolioRunLabel(r.path, r.label, projectRoot),
      })),
    [runs, projectRoot]
  );

  const portfolioDuckDbLogs = useMemo(
    () => normalizedRuns.map((r) => ({ path: r.path, label: r.label })),
    [normalizedRuns]
  );

  const filteredRuns = useMemo(
    () =>
      normalizedRuns.map((r) => ({
        ...r,
        entries: filterEntries(r.entries, policy, sampleId),
      })),
    [normalizedRuns, policy, sampleId]
  );

  // Consume pending eval results on mount
  useEffect(() => {
    if (pendingEvalResults && pendingEvalResults.length > 0) {
      setEvalRows(pendingEvalResults);
      setPendingEvalResults(null);
    }
  }, [pendingEvalResults, setPendingEvalResults]);

  useEffect(() => {
    if (!duckdbReady || portfolioDuckDbLogs.length === 0) return;
    setDuckdbLoading(true);
    runPortfolioSimulationArrowPipeline(portfolioDuckDbLogs, BENCHMARK_SIM_TABLE, projectRoot)
      .then(setLastPipeline)
      .catch((err) => console.warn("Benchmark Arrow pipeline:", err))
      .finally(() => setDuckdbLoading(false));
  }, [portfolioDuckDbLogs, duckdbReady, projectRoot, setLastPipeline, setDuckdbLoading]);

  // Consume pending benchmark logs from Output Browser compare action
  useEffect(() => {
    if (!pendingBenchmarkLogs || pendingBenchmarkLogs.length === 0) return;
    const load = async () => {
      const loaded: RunFile[] = [];
      for (const ref of pendingBenchmarkLogs) {
        const entries = await invoke<DayLogEntry[]>("load_simulation_log", { path: ref.path });
        const label = portfolioRunLabel(ref.path, ref.label, projectRoot);
        handoff(ref.path, "log", { storedLabel: label, navigate: false });
        loaded.push({
          path: ref.path,
          label,
          entries,
        });
      }
      setRuns(loaded);
      setPendingBenchmarkLogs(null);
    };
    load().catch(console.error);
  }, [pendingBenchmarkLogs, projectRoot, handoff, setPendingBenchmarkLogs]);

  const addRun = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "Logs", extensions: ["jsonl", "log", "txt"] }],
    })) as string | null;
    if (!path) return;
    const entries = await invoke<DayLogEntry[]>("load_simulation_log", { path });
    const label = portfolioRunLabel(path, undefined, projectRoot);
    handoff(path, "log", { storedLabel: label, navigate: false });
    setRuns((r) => [...r, { path, label, entries }]);
  }, [projectRoot, handoff]);

  const removeRun = (path: string) => setRuns((r) => r.filter((x) => x.path !== path));

  const loadOutputPortfolio = useCallback(async () => {
    if (!projectRoot) {
      toast.error("Set project root in Settings to scan output portfolio");
      return;
    }
    setPortfolioLoading(true);
    try {
      const refs = await scanOutputPortfolio(
        simOutputRoots(projectRoot),
        PORTFOLIO_SCAN_DEFAULT
      );
      if (!refs.length) {
        toast.error("No simulation logs found under outputs/simulations or assets/output");
        return;
      }
      const progressId = toast.loading(`Scanning portfolio… 0 / ${refs.length}`);
      const loaded = await loadPortfolioLogs(refs, {
        batchSize: 24,
        onProgress: (n, total) => {
          toast.loading(`Loading portfolio… ${n} / ${total}`, { id: progressId });
        },
      });
      const normalized = loaded.map((r) => ({
        ...r,
        label: portfolioRunLabel(r.path, r.label, projectRoot),
      }));
      for (const r of normalized) {
        handoff(r.path, "log", { storedLabel: r.label, navigate: false });
      }
      setRuns(normalized);
      toast.success(`Loaded ${loaded.length} simulation log(s) from output portfolio`, {
        id: progressId,
      });
    } catch (err) {
      toast.error("Portfolio load failed", { description: String(err) });
    } finally {
      setPortfolioLoading(false);
    }
  }, [projectRoot, handoff]);

  const exportComparisonCsv = useCallback(() => {
    const policies = [...new Set(filteredRuns.flatMap((r) => r.entries.map((e) => e.policy)))];
    downloadCsv(
      "benchmark-comparison.csv",
      ["run", "policy", ...SIM_METRICS.map((m) => m.key)],
      filteredRuns.flatMap((r) =>
        policies.map((p) => {
          const vals = r.entries
            .filter((e) => e.policy === p)
            .map((e) => e.data as Record<string, number>);
          const row: Array<string | number> = [r.label, p];
          for (const { key } of SIM_METRICS) {
            const v = vals.map((d) => d[key]).filter((x): x is number => x != null);
            row.push(v.length ? mean(v).toFixed(4) : "");
          }
          return row;
        })
      )
    );
  }, [filteredRuns]);

  const efficiencyRanking = useMemo(() => {
    const policies = [...new Set(filteredRuns.flatMap((r) => r.entries.map((e) => e.policy)))];
    return policies
      .map((p) => {
        const vals = filteredRuns.flatMap((r) =>
          r.entries
            .filter((e) => e.policy === p)
            .map((e) => (e.data as Record<string, number>)["kg/km"])
            .filter((v): v is number => v != null)
        );
        return { policy: p, mean: mean(vals), std: std(vals) };
      })
      .filter((r) => r.mean > 0)
      .sort((a, b) => b.mean - a.mean);
  }, [filteredRuns]);

  const paretoByPanel = useMemo(() => buildParetoByPanel(filteredRuns), [filteredRuns]);

  const cityGroups = useMemo(() => groupRunsByCity(filteredRuns), [filteredRuns]);

  const {
    runLabels: portfolioRunLabels,
    runLabel: activeRunLabel,
    brushedRunLabels,
    handleCityClick,
  } = usePortfolioRunBrush(filteredRuns);

  const cityComparisonOption = useMemo(
    () =>
      cityComparisonChartOption(buildCityComparisonSeries(cityGroups), {
        logScale,
        showErrorBars,
      }),
    [cityGroups, logScale, showErrorBars]
  );

  const onCityChartClick = useCallback(
    (params: { name?: string }) => {
      if (params.name) handleCityClick(params.name);
    },
    [handleCityClick]
  );

  const handlePolicyClick = useCallback(
    (name: string) => {
      setPolicy(policy === name ? null : name);
    },
    [policy, setPolicy]
  );

  const onChartClick = useCallback(
    (params: { name?: string; seriesName?: string }) => {
      const name = params.seriesName ?? params.name;
      if (name) handlePolicyClick(name);
    },
    [handlePolicyClick]
  );

  const efficiencyRankOption = useMemo(
    () => ({
      backgroundColor: "transparent",
      grid: { left: 110, right: 24, top: 12, bottom: 12 },
      xAxis: {
        type: (logScale ? "log" : "value") as "log" | "value",
        logBase: 10,
        name: "kg/km",
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLabel: { color: "#9090b0", fontSize: 9 },
        minorSplitLine: { show: false },
      },
      yAxis: {
        type: "category",
        data: efficiencyRanking.map((r) => r.policy),
        inverse: true,
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      tooltip: {
        trigger: "axis",
        formatter: (params: unknown[]) => {
          const p = (params as Array<{ dataIndex: number; name: string }>)[0];
          const r = efficiencyRanking[p.dataIndex];
          return `${p.name}<br/>${fmt(r.mean)} ± ${fmt(r.std)} kg/km`;
        },
      },
      series: [
        {
          type: "bar",
          data: efficiencyRanking.map((r, i) => ({
            value: logScale ? Math.max(r.mean, 0.001) : r.mean,
            itemStyle: {
              color: COLORS[i % COLORS.length],
              opacity: barOpacity(r.policy, brushedPolicies),
            },
          })),
        },
        ...(showErrorBars
          ? [
              {
                type: "custom" as const,
                renderItem: (
                  params: { dataIndex: number },
                  api: {
                    coord: (v: [number, number]) => [number, number];
                    style: (s: object) => object;
                  }
                ) => {
                  const i = params.dataIndex;
                  const r = efficiencyRanking[i];
                  const bounds = errorBarBounds(r.mean, r.std, "kg/km", logScale);
                  const y = api.coord([bounds.center, i])[1];
                  const xLeft = api.coord([bounds.low, i])[0];
                  const xRight = api.coord([bounds.high, i])[0];
                  const cap = 4;
                  return {
                    type: "group",
                    children: [
                      {
                        type: "line",
                        shape: { x1: xLeft, y1: y, x2: xRight, y2: y },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                      {
                        type: "line",
                        shape: { x1: xLeft, y1: y - cap, x2: xLeft, y2: y + cap },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                      {
                        type: "line",
                        shape: { x1: xRight, y1: y - cap, x2: xRight, y2: y + cap },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                    ],
                  };
                },
                data: efficiencyRanking.map((_, i) => i),
                z: 10,
              },
            ]
          : []),
      ],
    }),
    [efficiencyRanking, brushedPolicies, logScale, showErrorBars]
  );

  const makeBarOption = (metricKey: string, metricLabel: string) => {
    const symlogOverflows = logScale && metricKey === "overflows";
    const runLabels = filteredRuns.map((r) => r.label);
    const policies = [...new Set(filteredRuns.flatMap((r) => r.entries.map((e) => e.policy)))];

    const statsGrid = policies.map((p) =>
      filteredRuns.map((r) => {
        const vals = r.entries
          .filter((e) => e.policy === p)
          .map((e) => (e.data as Record<string, number>)[metricKey] ?? null)
          .filter((v): v is number => v !== null);
        return { mean: mean(vals), std: std(vals) };
      })
    );

    const barSeries = policies.map((p, i) => ({
      name: p,
      type: "bar" as const,
      data: statsGrid[i].map((s) => {
        if (!logScale) return s.mean;
        return symlogOverflows ? symlog(s.mean) : Math.max(s.mean, 0.001);
      }),
      itemStyle: {
        color: COLORS[i % COLORS.length],
        opacity: barOpacity(p, brushedPolicies),
      },
    }));

    const errorBarPoints = policies.flatMap((_, polIdx) =>
      filteredRuns.map((_, runIdx) => ({
        runIdx,
        polIdx,
        ...statsGrid[polIdx][runIdx],
      }))
    );

    const errorBarSeries = showErrorBars
      ? [
          {
            type: "custom" as const,
            renderItem: (
              params: { dataIndex: number },
              api: {
                coord: (v: [number, number]) => [number, number];
                size: (v: [number, number]) => [number, number];
                style: (s: object) => object;
              }
            ) => {
              const point = errorBarPoints[params.dataIndex];
              const bounds = errorBarBounds(
                point.mean,
                point.std,
                metricKey,
                logScale,
                symlogOverflows
              );
              const x = groupedBarWhiskerX(
                api,
                point.runIdx,
                point.polIdx,
                policies.length,
                bounds.center
              );
              const yTop = api.coord([point.runIdx, bounds.high])[1];
              const yBot = api.coord([point.runIdx, bounds.low])[1];
              const cap = 4;
              return {
                type: "group",
                children: [
                  {
                    type: "line",
                    shape: { x1: x, y1: yTop, x2: x, y2: yBot },
                    style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                  },
                  {
                    type: "line",
                    shape: { x1: x - cap, y1: yTop, x2: x + cap, y2: yTop },
                    style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                  },
                  {
                    type: "line",
                    shape: { x1: x - cap, y1: yBot, x2: x + cap, y2: yBot },
                    style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                  },
                ],
              };
            },
            data: errorBarPoints.map((_, i) => i),
            z: 10,
          },
        ]
      : [];

    return {
      backgroundColor: "transparent",
      legend: { textStyle: { color: "#9090b0", fontSize: 11 } },
      grid: { left: 50, right: 10, top: 35, bottom: 55 },
      xAxis: {
        type: "category",
        data: runLabels,
        axisLabel: { color: "#9090b0", fontSize: 9, rotate: 20 },
      },
      yAxis: {
        type: (logScale && !symlogOverflows ? "log" : "value") as "log" | "value",
        logBase: 10,
        name: metricLabel,
        nameTextStyle: { color: "#9090b0" },
        axisLabel: { color: "#9090b0", fontSize: 10 },
        minorSplitLine: { show: false },
      },
      series: [...barSeries, ...errorBarSeries],
      tooltip: {
        trigger: "axis",
        formatter: (params: unknown[]) => {
          const items = params as Array<{ seriesName: string; dataIndex: number }>;
          const run = runLabels[items[0]?.dataIndex ?? 0];
          const lines = items
            .filter((p) => p.seriesName)
            .map((p) => {
              const polIdx = policies.indexOf(p.seriesName);
              const s = statsGrid[polIdx]?.[p.dataIndex];
              return s
                ? `${p.seriesName}: ${fmt(s.mean)} ± ${fmt(s.std)}`
                : `${p.seriesName}: —`;
            });
          return `${run}<br/>${lines.join("<br/>")}`;
        },
      },
    };
  };

  return (
    <div className="space-y-4">
      <GlobalFilterBar
        runLabels={portfolioRunLabels.length > 0 ? portfolioRunLabels : []}
        cities={filteredRuns.length > 1 ? cityGroups.map(([city]) => city) : []}
        showLogScale
      />

      {evalRows && evalRows.length > 0 && (
        <EvalResultsPanel
          rows={evalRows}
          logScale={logScale}
          projectRoot={projectRoot}
          onDismiss={() => setEvalRows(null)}
        />
      )}

      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={addRun} className="btn-primary flex items-center gap-2">
          <FolderOpen size={14} />
          Add Simulation Run
        </button>
        {projectRoot && (
          <button
            onClick={() => void loadOutputPortfolio()}
            disabled={portfolioLoading}
            className="btn-ghost flex items-center gap-2 text-xs"
          >
            <FolderOpen size={14} />
            {portfolioLoading ? "Scanning output…" : "Load output portfolio (§G.1.4)"}
          </button>
        )}
        <span className="text-xs text-canvas-muted">
          {runs.length} simulation run(s) loaded
          {duckdbLoading && " · DuckDB ingesting…"}
          {!duckdbLoading && lastPipeline?.tableName === BENCHMARK_SIM_TABLE && (
            <> · {formatPipelineTimingBadge(lastPipeline)}</>
          )}
        </span>
        {filteredRuns.length > 0 && (
          <button onClick={exportComparisonCsv} className="btn-ghost text-xs flex items-center gap-1">
            <Download size={12} />
            Export CSV
          </button>
        )}
        {filteredRuns.length > 0 && (
          <button
            onClick={() => setShowErrorBars((v) => !v)}
            className={`btn-ghost text-xs ${showErrorBars ? "text-accent-secondary" : ""}`}
          >
            {showErrorBars ? "Error bars (on)" : "Error bars (off)"}
          </button>
        )}
      </div>

      {runs.length > 0 && (
        <div className="card">
          <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider mb-2">
            Loaded Runs
          </p>
          <div className="space-y-1">
            {normalizedRuns.map((r) => (
              <LoadedRunRow
                key={r.path}
                path={r.path}
                projectRoot={projectRoot}
                label={r.label}
                activeRunLabel={activeRunLabel}
                onRemove={() => removeRun(r.path)}
                pathHandoffs
                trailing={
                  <span className="ml-auto text-canvas-muted shrink-0">{r.entries.length} days</span>
                }
              />
            ))}
          </div>
        </div>
      )}

      {runs.length === 0 && !evalRows && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Add simulation run logs to compare, or use &quot;Open in Analytics →&quot; from the Evaluation Runner.
        </div>
      )}

      {runs.length >= 2 && (
        <BenchmarkPortfolioParallel runs={filteredRuns} logScale={logScale} />
      )}

      {runs.length >= 1 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-gray-300">Pareto Panels (§G.1.2)</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {PARETO_PANELS.map((panel) => (
              <BenchmarkParetoPanel
                key={panel.id}
                label={panel.label}
                points={paretoByPanel[panel.id] ?? []}
                logScale={logScale}
              />
            ))}
          </div>
        </div>
      )}

      {runs.length >= 2 && cityGroups.length > 1 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <p className="text-xs font-semibold text-gray-300">Heatmaps by Graph (§G.1.3)</p>
            <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
              {(
                [
                  { key: "all", label: "All metrics" },
                  { key: "overflows", label: "Overflows" },
                  { key: "kg/km", label: "kg/km" },
                ] as const
              ).map((o) => (
                <button
                  key={o.key}
                  onClick={() => setHeatmapMode(o.key)}
                  className={`text-xs px-2.5 py-1 rounded-md ${
                    heatmapMode === o.key
                      ? "bg-accent-primary text-white"
                      : "text-canvas-muted hover:text-gray-200"
                  }`}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {cityGroups.map(([graph, graphRuns]) => (
              <BenchmarkGraphHeatmap
                key={graph}
                graphLabel={graph}
                runs={graphRuns}
                heatmapMode={heatmapMode}
                logScale={logScale}
              />
            ))}
          </div>
        </div>
      )}

      {runs.length >= 2 && cityGroups.length >= 1 && (
        <div className="card space-y-2">
          <p className="text-xs font-semibold text-gray-300">City Comparison (§G.1.6)</p>
          <p className="text-[10px] text-canvas-muted">
            {logScale
              ? `Log-scale bars — profit · symlog-overflows · kg/km by graph scale${showErrorBars ? " · error bars on" : ""}`
              : `Linear bars — profit · overflows · kg/km by graph scale${showErrorBars ? " · error bars on" : ""}`}
          </p>
          <ReactECharts
            ref={(el) => {
              chartRefs.current["city-compare"] = el;
            }}
            option={cityComparisonOption}
            style={{ height: 240 }}
            onEvents={{ click: onCityChartClick }}
          />
        </div>
      )}

      {runs.length >= 2 && efficiencyRanking.length > 0 && (
        <PortfolioEfficiencyRanking
          runs={filteredRuns}
          showErrorBars={showErrorBars}
          logScale={logScale}
          brushed={brushedPolicies}
          onPolicyClick={handlePolicyClick}
        />
      )}

      {runs.length === 1 && efficiencyRanking.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-canvas-muted">Efficiency Ranking (kg/km)</p>
            <ChartExportButtons
              chartRef={{ current: chartRefs.current["efficiency-rank"] }}
              filenameStem="benchmark-efficiency-rank"
            />
          </div>
          <ReactECharts
            ref={(el) => {
              chartRefs.current["efficiency-rank"] = el;
            }}
            option={efficiencyRankOption}
            style={{ height: Math.max(180, efficiencyRanking.length * 28) }}
            onEvents={{ click: onChartClick }}
          />
        </div>
      )}

      {runs.length >= 1 && (
        <PolicyTelemetryTrendsPanel
          theme={theme}
          logScale={logScale}
          initialPolicy={brushedPolicies?.length === 1 ? brushedPolicies[0]! : null}
          initialRunLabel={
            brushedRunLabels?.length === 1 ? brushedRunLabels[0]! : null
          }
        />
      )}

      {runs.length >= 1 && duckdbReady && (
        <SqlQueryPanel
          tableName={BENCHMARK_SIM_TABLE}
          theme={theme}
          highlightPolicies={brushedPolicies}
          highlightRunLabels={brushedRunLabels}
          brushSqlSync
          autoRunOnBrushSync
          portfolioMode={runs.length > 1}
          portfolioRunLabels={runs.length > 1 ? portfolioRunLabels : []}
        />
      )}

      {runs.length >= 1 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {SIM_METRICS.map(({ key, label }) => (
            <div key={key} className="card">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-canvas-muted">{label}</p>
                <ChartExportButtons
                  chartRef={{ current: chartRefs.current[key] }}
                  filenameStem={`benchmark-${key}`}
                />
              </div>
              <ReactECharts
                ref={(el) => {
                  chartRefs.current[key] = el;
                }}
                option={makeBarOption(key, label)}
                style={{ height: 220 }}
                onEvents={{ click: onChartClick, legendselectchanged: onChartClick }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
