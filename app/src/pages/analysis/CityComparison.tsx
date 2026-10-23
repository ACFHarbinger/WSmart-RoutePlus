/**
 * Dedicated City Comparison view (§G.1.6).
 *
 * Log-scale bar charts grouped by city/graph scale across a multi-run portfolio.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen } from "lucide-react";
import { toast } from "sonner";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { usePortfolioRunBrush } from "../../hooks/brush/usePortfolioRunBrush";
import { useAppStore } from "../../store/app";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { useGlobalFiltersStore } from "../../store/filters";
import { filterEntries } from "../../store/sim";
import { ChartExportButtons } from "../../components/common/ChartExportButtons";
import {
  buildCityComparisonSeries,
  cityComparisonChartOption,
  groupRunsByCity,
  type CityRunSlice,
} from "../../utils/benchmark/cityComparison";
import {
  loadPortfolioLogs,
  PORTFOLIO_SCAN_DEFAULT,
  scanOutputPortfolio,
} from "../../utils/runs/outputRunLogs";
import { LoadedRunRow } from "../../components/common/LoadedRunRow";
import {
  formatPipelineTimingBadge,
  portfolioRunLabel,
  runPortfolioSimulationArrowPipeline,
} from "../../utils/duckdb/arrowPipeline";
import { PolicyTelemetryTrendsPanel } from "../../components/analysis/telemetry/PolicyTelemetryTrendsPanel";
import { SqlQueryPanel } from "../../components/analysis/explorer/SqlQueryPanel";
import { useDuckDbStore } from "../../store/duckdb";
import type { DayLogEntry } from "../../types";
import { simOutputRoots } from "../../utils/runs/outputRoots";
const CITY_SIM_TABLE = "city_sim";

interface RunFile extends CityRunSlice {}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

export function CityComparison() {
  const { effectiveTheme: theme } = useAppStore();
  const { projectRoot, handoff } = useRecentHandoff();
  const { policy: filterPolicy, sampleId: filterSample, logScale } = useGlobalFiltersStore();
  const brushedPolicies = useMemo(() => (filterPolicy ? [filterPolicy] : null), [filterPolicy]);
  const {
    ready: duckdbReady,
    loading: duckdbLoading,
    lastPipeline,
    setLastPipeline,
    setLoading: setDuckdbLoading,
  } = useDuckDbStore();
  const chartRef = useRef<EChartsReact | null>(null);
  const [runs, setRuns] = useState<RunFile[]>([]);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const [showErrorBars, setShowErrorBars] = useState(false);

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
        entries: filterEntries(r.entries, filterPolicy, filterSample),
      })),
    [normalizedRuns, filterPolicy, filterSample]
  );

  const cityGroups = useMemo(() => groupRunsByCity(filteredRuns), [filteredRuns]);
  const {
    runLabels: portfolioRunLabels,
    runLabel: activeRunLabel,
    brushedCity,
    brushedRunLabels,
    handleCityClick,
  } = usePortfolioRunBrush(filteredRuns);

  const series = useMemo(() => buildCityComparisonSeries(cityGroups), [cityGroups]);
  const chartOption = useMemo(
    () => cityComparisonChartOption(series, { logScale, showErrorBars }),
    [series, logScale, showErrorBars]
  );

  const addRun = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "Simulation Log", extensions: ["jsonl"] }],
    })) as string | null;
    if (!path) return;
    try {
      const entries = await invoke<DayLogEntry[]>("load_simulation_log", { path });
      const label = portfolioRunLabel(path, undefined, projectRoot);
      handoff(path, "log", { storedLabel: label, navigate: false });
      setRuns((prev) => [...prev.filter((r) => r.path !== path), { path, label, entries }]);
    } catch (err) {
      toast.error("Failed to load log", { description: String(err) });
    }
  }, [projectRoot, handoff]);

  const loadOutputPortfolio = useCallback(async () => {
    if (!projectRoot) return;
    const outputPath = simOutputRoots(projectRoot);
    setPortfolioLoading(true);
    try {
      const refs = await scanOutputPortfolio(outputPath, PORTFOLIO_SCAN_DEFAULT);
      if (!refs.length) {
        toast.info("No simulation logs found under outputs/simulations or assets/output");
        return;
      }
      toast.info(`Loading ${refs.length} logs…`);
      const loaded = await loadPortfolioLogs(refs, {
        onProgress: (n, total) => {
          if (n % 48 === 0 || n === total) {
            toast.info(`Portfolio: ${n}/${total} logs loaded`);
          }
        },
      });
      const normalized = loaded.map((r) => ({
        path: r.path,
        label: portfolioRunLabel(r.path, r.label, projectRoot),
        entries: r.entries,
      }));
      for (const r of normalized) {
        handoff(r.path, "log", { storedLabel: r.label, navigate: false });
      }
      setRuns(normalized);
      toast.success(`Loaded ${loaded.length} simulation logs`);
    } catch (err) {
      toast.error("Portfolio load failed", { description: String(err) });
    } finally {
      setPortfolioLoading(false);
    }
  }, [projectRoot, handoff]);

  const { pendingBenchmarkLogs, setPendingBenchmarkLogs } = useAppStore();

  useEffect(() => {
    if (!duckdbReady || portfolioDuckDbLogs.length === 0) return;
    setDuckdbLoading(true);
    runPortfolioSimulationArrowPipeline(portfolioDuckDbLogs, CITY_SIM_TABLE, projectRoot)
      .then(setLastPipeline)
      .catch((err) => console.warn("City comparison Arrow pipeline:", err))
      .finally(() => setDuckdbLoading(false));
  }, [portfolioDuckDbLogs, duckdbReady, projectRoot, setLastPipeline, setDuckdbLoading]);

  useEffect(() => {
    if (!pendingBenchmarkLogs?.length) return;
    void (async () => {
      const loaded: RunFile[] = [];
      for (const ref of pendingBenchmarkLogs) {
        try {
          const entries = await invoke<DayLogEntry[]>("load_simulation_log", { path: ref.path });
          const label = portfolioRunLabel(ref.path, ref.label, projectRoot);
          handoff(ref.path, "log", { storedLabel: label, navigate: false });
          loaded.push({
            path: ref.path,
            label,
            entries,
          });
        } catch {
          /* skip */
        }
      }
      if (loaded.length) setRuns(loaded);
      setPendingBenchmarkLogs(null);
    })();
  }, [pendingBenchmarkLogs, projectRoot, handoff, setPendingBenchmarkLogs]);

  const onChartClick = useCallback(
    (params: { name?: string }) => {
      if (params.name) handleCityClick(params.name);
    },
    [handleCityClick]
  );

  const citySummaryRows = useMemo(() => {
    return cityGroups.map(([city, cityRuns]) => {
      const profits = cityRuns.flatMap((r) =>
        r.entries.map((e) => e.data.profit).filter((v): v is number => v != null)
      );
      const overflows = cityRuns.flatMap((r) =>
        r.entries.map((e) => e.data.overflows).filter((v): v is number => v != null)
      );
      const kgkm = cityRuns.flatMap((r) =>
        r.entries.map((e) => e.data["kg/km"]).filter((v): v is number => v != null)
      );
      return {
        city,
        runs: cityRuns.length,
        profit: mean(profits),
        overflows: mean(overflows),
        kgkm: mean(kgkm),
      };
    });
  }, [cityGroups]);

  const removeRun = (path: string) => setRuns((prev) => prev.filter((r) => r.path !== path));

  return (
    <div className="space-y-4">
      <GlobalFilterBar
        runLabels={portfolioRunLabels.length > 0 ? portfolioRunLabels : []}
        cities={runs.length > 1 ? cityGroups.map(([city]) => city) : []}
        showLogScale
      />

      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={() => void addRun()} className="btn-primary flex items-center gap-2 text-xs">
          <FolderOpen size={14} />
          Add simulation run
        </button>
        {projectRoot && (
          <button
            onClick={() => void loadOutputPortfolio()}
            disabled={portfolioLoading}
            className="btn-ghost flex items-center gap-2 text-xs"
          >
            <FolderOpen size={14} />
            {portfolioLoading ? "Loading portfolio…" : "Load output portfolio"}
          </button>
        )}
        <span className="text-xs text-canvas-muted">
          {runs.length} run(s) loaded
          {duckdbLoading && " · DuckDB ingesting…"}
          {!duckdbLoading && lastPipeline?.tableName === CITY_SIM_TABLE && (
            <> · {formatPipelineTimingBadge(lastPipeline)}</>
          )}
        </span>
        {cityGroups.length >= 1 && (
          <button
            onClick={() => setShowErrorBars((v) => !v)}
            className={`btn-ghost text-xs ${showErrorBars ? "text-accent-secondary" : ""}`}
          >
            {showErrorBars ? "Error bars (on)" : "Error bars (off)"}
          </button>
        )}
      </div>

      {runs.length > 0 && (
        <div className="card space-y-1">
          <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider mb-2">
            Loaded runs
          </p>
          {normalizedRuns.map((r) => (
            <LoadedRunRow
              key={r.path}
              path={r.path}
              projectRoot={projectRoot}
              label={r.label}
              activeRunLabel={activeRunLabel}
              onRemove={() => removeRun(r.path)}
              pathHandoffs
            />
          ))}
        </div>
      )}

      {runs.length === 0 && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Add simulation runs or load the output portfolio to compare cities.
        </div>
      )}

      {cityGroups.length >= 1 && (
        <div className="card space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <p className="text-xs font-semibold text-gray-300">City Comparison (§G.1.6)</p>
              <p className="text-[10px] text-canvas-muted">
                {logScale
                  ? `Log-scale bars — profit · symlog-overflows · kg/km by graph scale${showErrorBars ? " · error bars on" : ""}`
                  : `Linear bars — profit · overflows · kg/km by graph scale${showErrorBars ? " · error bars on" : ""}`}
              </p>
            </div>
            <ChartExportButtons chartRef={chartRef} filenameStem="city-comparison" />
          </div>
          <ReactECharts
            ref={chartRef}
            option={chartOption}
            style={{ height: 280 }}
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
            activeRunLabel ??
            (brushedRunLabels?.length === 1 ? brushedRunLabels[0]! : null)
          }
        />
      )}

      {runs.length >= 1 && duckdbReady && (
        <SqlQueryPanel
          tableName={CITY_SIM_TABLE}
          theme={theme}
          highlightPolicies={brushedPolicies}
          highlightRunLabels={brushedRunLabels}
          brushSqlSync
          autoRunOnBrushSync
          portfolioMode={runs.length > 1}
          portfolioRunLabels={runs.length > 1 ? portfolioRunLabels : []}
        />
      )}

      {citySummaryRows.length > 0 && (
        <div className="card overflow-auto">
          <p className="text-xs font-semibold text-gray-300 mb-2">Summary table</p>
          <table className="w-full text-xs">
            <thead className="text-canvas-muted">
              <tr>
                <th className="text-left py-1 pr-3">City / scale</th>
                <th className="text-right py-1 px-2">Runs</th>
                <th className="text-right py-1 px-2">Profit (€)</th>
                <th className="text-right py-1 px-2">Overflows</th>
                <th className="text-right py-1 pl-2">kg/km</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-canvas-border font-mono text-gray-300">
              {citySummaryRows.map((row) => (
                <tr
                  key={row.city}
                  className={`cursor-pointer hover:bg-canvas-hover ${
                    brushedCity === row.city ? "bg-accent-primary/15" : ""
                  }`}
                  onClick={() => handleCityClick(row.city)}
                >
                  <td className="py-1.5 pr-3">{row.city}</td>
                  <td className="text-right py-1.5 px-2">{row.runs}</td>
                  <td className="text-right py-1.5 px-2">{row.profit.toFixed(1)}</td>
                  <td className="text-right py-1.5 px-2">{row.overflows.toFixed(2)}</td>
                  <td className="text-right py-1.5 pl-2">{row.kgkm.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
