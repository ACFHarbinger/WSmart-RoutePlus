/**
 * Benchmark Analysis — multi-run, multi-policy comparison.
 * Ports Streamlit `benchmark` mode.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, Download } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { usePortfolioRunBrush } from "../../hooks/brush/usePortfolioRunBrush";
import { useAppStore } from "../../store/app";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { useGlobalFiltersStore } from "../../store/filters";
import { filterEntries } from "../../store/sim";
import { buildParetoByPanel } from "../../utils/benchmark/paretoPortfolio";
import type { HeatmapMode } from "../../utils/benchmark/heatmapMetrics";
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
import { LoadedRunRow } from "../../components/common/LoadedRunRow";
import { useDuckDbStore } from "../../store/duckdb";
import { toast } from "sonner";
import type { DayLogEntry, EvalAnalyticsRow } from "../../types";
import { simOutputRoots } from "../../utils/runs/outputRoots";
import {
  buildEfficiencyRankOption,
  EvalResultsPanel,
  mean,
  SIM_METRICS,
  std,
  type RunFile,
} from "../../components/analysis/benchmark/BenchmarkEvalAndBars";
import { BenchmarkPortfolioCharts } from "../../components/analysis/benchmark/BenchmarkPortfolioCharts";

const BENCHMARK_SIM_TABLE = "benchmark_sim";

export function BenchmarkAnalysis() {
  const {
    pendingEvalResults,
    setPendingEvalResults,
    pendingBenchmarkLogs,
    setPendingBenchmarkLogs,
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

  useEffect(() => {
    if (pendingEvalResults && pendingEvalResults.length > 0) {
      setEvalRows(pendingEvalResults);
      setPendingEvalResults(null);
    }
  }, [pendingEvalResults, setPendingEvalResults]);

  useEffect(() => {
    if (!duckdbReady || portfolioDuckDbLogs.length === 0) return;
    setDuckdbLoading(true);
    runPortfolioSimulationArrowPipeline(
      portfolioDuckDbLogs,
      BENCHMARK_SIM_TABLE,
      projectRoot
    )
      .then(setLastPipeline)
      .catch((err) => console.warn("Benchmark Arrow pipeline:", err))
      .finally(() => setDuckdbLoading(false));
  }, [
    portfolioDuckDbLogs,
    duckdbReady,
    projectRoot,
    setLastPipeline,
    setDuckdbLoading,
  ]);

  useEffect(() => {
    if (!pendingBenchmarkLogs || pendingBenchmarkLogs.length === 0) return;
    const load = async () => {
      const loaded: RunFile[] = [];
      for (const ref of pendingBenchmarkLogs) {
        const entries = await invoke<DayLogEntry[]>("load_simulation_log", {
          path: ref.path,
        });
        const label = portfolioRunLabel(ref.path, ref.label, projectRoot);
        handoff(ref.path, "log", { storedLabel: label, navigate: false });
        loaded.push({ path: ref.path, label, entries });
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

  const removeRun = (path: string) =>
    setRuns((r) => r.filter((x) => x.path !== path));

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
        toast.error(
          "No simulation logs found under outputs/simulations or assets/output"
        );
        return;
      }
      const progressId = toast.loading(`Scanning portfolio… 0 / ${refs.length}`);
      const loaded = await loadPortfolioLogs(refs, {
        batchSize: 24,
        onProgress: (n, total) => {
          toast.loading(`Loading portfolio… ${n} / ${total}`, {
            id: progressId,
          });
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
      toast.success(
        `Loaded ${loaded.length} simulation log(s) from output portfolio`,
        { id: progressId }
      );
    } catch (err) {
      toast.error("Portfolio load failed", { description: String(err) });
    } finally {
      setPortfolioLoading(false);
    }
  }, [projectRoot, handoff]);

  const exportComparisonCsv = useCallback(() => {
    const policies = [
      ...new Set(filteredRuns.flatMap((r) => r.entries.map((e) => e.policy))),
    ];
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
            const v = vals
              .map((d) => d[key])
              .filter((x): x is number => x != null);
            row.push(v.length ? mean(v).toFixed(4) : "");
          }
          return row;
        })
      )
    );
  }, [filteredRuns]);

  const efficiencyRanking = useMemo(() => {
    const policies = [
      ...new Set(filteredRuns.flatMap((r) => r.entries.map((e) => e.policy))),
    ];
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

  const paretoByPanel = useMemo(
    () => buildParetoByPanel(filteredRuns),
    [filteredRuns]
  );

  const cityGroups = useMemo(
    () => groupRunsByCity(filteredRuns),
    [filteredRuns]
  );

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
    () =>
      buildEfficiencyRankOption(
        efficiencyRanking,
        brushedPolicies,
        logScale,
        showErrorBars
      ),
    [efficiencyRanking, brushedPolicies, logScale, showErrorBars]
  );

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
        <button
          onClick={addRun}
          className="btn-primary flex items-center gap-2"
        >
          <FolderOpen size={14} /> Add Simulation Run
        </button>
        {projectRoot && (
          <button
            onClick={() => void loadOutputPortfolio()}
            disabled={portfolioLoading}
            className="btn-ghost flex items-center gap-2 text-xs"
          >
            <FolderOpen size={14} />
            {portfolioLoading
              ? "Scanning output…"
              : "Load output portfolio (§G.1.4)"}
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
          <button
            onClick={exportComparisonCsv}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <Download size={12} /> Export CSV
          </button>
        )}
        {filteredRuns.length > 0 && (
          <button
            onClick={() => setShowErrorBars((v) => !v)}
            className={`btn-ghost text-xs ${
              showErrorBars ? "text-accent-secondary" : ""
            }`}
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
                  <span className="ml-auto text-canvas-muted shrink-0">
                    {r.entries.length} days
                  </span>
                }
              />
            ))}
          </div>
        </div>
      )}

      {runs.length === 0 && !evalRows && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Add simulation run logs to compare, or use &quot;Open in Analytics
          →&quot; from the Evaluation Runner.
        </div>
      )}

      <BenchmarkPortfolioCharts
        runsCount={runs.length}
        filteredRuns={filteredRuns}
        paretoByPanel={paretoByPanel}
        cityGroups={cityGroups}
        heatmapMode={heatmapMode}
        setHeatmapMode={setHeatmapMode}
        logScale={logScale}
        showErrorBars={showErrorBars}
        cityComparisonOption={cityComparisonOption}
        onCityChartClick={onCityChartClick}
        efficiencyRanking={efficiencyRanking}
        efficiencyRankOption={efficiencyRankOption}
        brushedPolicies={brushedPolicies}
        brushedRunLabels={brushedRunLabels}
        handlePolicyClick={handlePolicyClick}
        onChartClick={onChartClick}
        theme={theme}
        duckdbReady={duckdbReady}
        benchmarkSimTable={BENCHMARK_SIM_TABLE}
        portfolioRunLabels={portfolioRunLabels}
      />
    </div>
  );
}
