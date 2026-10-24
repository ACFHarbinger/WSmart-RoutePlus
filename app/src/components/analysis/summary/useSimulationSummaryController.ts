import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { toast } from "sonner";
import { useAppStore } from "../../../store/app";
import { useLoadedRunStore } from "../../../store/loadedRun";
import { useRecentHandoff } from "../../../hooks/files/useRecentHandoff";
import { useLogPathRunLabelBrush } from "../../../hooks/brush/useLogPathRunLabelBrush";
import { usePortfolioRunBrush } from "../../../hooks/brush/usePortfolioRunBrush";
import { useGlobalFiltersStore } from "../../../store/filters";
import { filterEntries } from "../../../store/sim";
import { toggleBrush } from "../../../utils/charts/chartHighlight";
import { parseLogPath, parsePolicyLabel, type PolicyMeta } from "../../../utils/sim/simMetadata";
import type { PortfolioHierarchyRun } from "../../../utils/benchmark/policyHierarchy";
import type { HeatmapMode } from "../../../utils/benchmark/heatmapMetrics";
import { buildParetoByPanel, type PortfolioRunSlice } from "../../../utils/benchmark/paretoPortfolio";
import { groupRunsByDistribution } from "../../../utils/benchmark/portfolioDistribution";
import { groupRunsByCity, buildCityComparisonSeries, cityComparisonChartOption } from "../../../utils/benchmark/cityComparison";
import { loadPortfolioLogs, PORTFOLIO_SCAN_DEFAULT, scanOutputPortfolio } from "../../../utils/runs/outputRunLogs";
import { downloadCsv, downloadParquetTable } from "../../../utils/charts/tableExport";
import { portfolioRunLabel, runPortfolioSimulationArrowPipeline } from "../../../utils/duckdb/arrowPipeline";
import { useDuckDbStore } from "../../../store/duckdb";
import { simOutputRoots } from "../../../utils/runs/outputRoots";
import type { DayLogEntry } from "../../../types";
import { aggregateByPolicy, buildGroupedStats, mean, std, type MetricKey } from "./SummaryCoreWidgets";

export const SUMMARY_SIM_TABLE = "summary_sim";

export type SummarySubtab = "all" | "table" | "trajectories" | "map" | "compare";

export interface ComparisonRun extends PortfolioRunSlice {}

export function useSimulationSummaryController() {
  const { pendingLogPath, setPendingLogPath, effectiveTheme: theme } = useAppStore();
  const {
    path: loadedRunPath,
    entries: loadedRunEntries,
    setLoadedRun,
  } = useLoadedRunStore();
  const { projectRoot, handoff } = useRecentHandoff();
  const {
    ready: duckdbReady,
    loading: duckdbLoading,
    lastPipeline,
    setLastPipeline,
    setLoading: setDuckdbLoading,
  } = useDuckDbStore();
  const [subtab, setSubtab] = useState<SummarySubtab>("all");
  const [parquetExporting, setParquetExporting] = useState(false);
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const [showErrorBars, setShowErrorBars] = useState(false);
  const [brushed, setBrushed] = useState<string[] | null>(null);
  const [heatmapMode, setHeatmapMode] = useState<HeatmapMode>("all");
  const [overflowMax, setOverflowMax] = useState<number | null>(null);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const { policy, sampleId } = useGlobalFiltersStore();
  const [entries, setEntries] = useState<DayLogEntry[]>(loadedRunEntries);
  const [logPath, setLogPath] = useState<string | null>(loadedRunPath);
  const derivedRunLabel = useLogPathRunLabelBrush(logPath);
  const [comparisonRuns, setComparisonRuns] = useState<ComparisonRun[]>([]);
  const [routeVizDay, setRouteVizDay] = useState(1);
  const [showFailureOverlay, setShowFailureOverlay] = useState(true);
  const [showRouteDiff, setShowRouteDiff] = useState(false);

  const loadLog = useCallback(
    async (path: string) => {
      const loaded = await invoke<DayLogEntry[]>("load_simulation_log", { path });
      setEntries(loaded);
      setLogPath(path);
      setLoadedRun({
        path,
        label: portfolioRunLabel(path, undefined, projectRoot) ?? "simulation_log",
        entries: loaded,
        source: "viewing",
      });
      handoff(path, "log", { navigate: false });
    },
    [handoff, projectRoot, setLoadedRun]
  );

  const allDuckDbLogs = useMemo(() => {
    const logs: { path: string; label: string }[] = [];
    if (logPath) {
      logs.push({
        path: logPath,
        label: portfolioRunLabel(logPath, undefined, projectRoot),
      });
    }
    for (const r of comparisonRuns) {
      if (!logs.some((l) => l.path === r.path)) {
        logs.push({
          path: r.path,
          label: portfolioRunLabel(r.path, r.label, projectRoot),
        });
      }
    }
    return logs;
  }, [logPath, comparisonRuns, projectRoot]);

  useEffect(() => {
    if (!duckdbReady || allDuckDbLogs.length === 0) return;
    setDuckdbLoading(true);
    runPortfolioSimulationArrowPipeline(allDuckDbLogs, SUMMARY_SIM_TABLE, projectRoot)
      .then(setLastPipeline)
      .catch((err) => console.warn("Summary Arrow pipeline:", err))
      .finally(() => setDuckdbLoading(false));
  }, [allDuckDbLogs, duckdbReady, projectRoot, setLastPipeline, setDuckdbLoading]);

  useEffect(() => {
    if (pendingLogPath) {
      void loadLog(pendingLogPath);
      setPendingLogPath(null);
    }
  }, [pendingLogPath, setPendingLogPath, loadLog]);

  const openLog = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "Logs", extensions: ["jsonl", "log", "txt"] }],
    })) as string | null;
    if (!path) return;
    await loadLog(path);
  }, [loadLog]);

  const addComparisonRun = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "Logs", extensions: ["jsonl", "log", "txt"] }],
    })) as string | null;
    if (!path) return;
    try {
      const loaded = await invoke<DayLogEntry[]>("load_simulation_log", { path });
      const label = portfolioRunLabel(path, undefined, projectRoot);
      handoff(path, "log", { storedLabel: label, navigate: false });
      setComparisonRuns((prev) => [
        ...prev.filter((r) => r.path !== path),
        { path, label, entries: loaded },
      ]);
    } catch (err) {
      toast.error("Failed to load comparison log", { description: String(err) });
    }
  }, [projectRoot, handoff]);

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
        toast.info("No simulation logs found under outputs/simulations or assets/output");
        return;
      }
      const progressId = toast.loading(`Loading portfolio… 0 / ${refs.length}`);
      const loaded = await loadPortfolioLogs(refs, {
        batchSize: 24,
        onProgress: (n, total) => {
          toast.loading(`Loading portfolio… ${n} / ${total}`, { id: progressId });
        },
      });
      if (loaded.length > 0) {
        const [primary, ...rest] = loaded;
        for (const r of loaded) {
          handoff(r.path, "log", { storedLabel: r.label, navigate: false });
        }
        if (!logPath) {
          await loadLog(primary.path);
        }
        setComparisonRuns((prev) => {
          const seen = new Set(prev.map((r) => r.path));
          if (logPath) seen.add(logPath);
          const normalize = (r: ComparisonRun): ComparisonRun => ({
            ...r,
            label: portfolioRunLabel(r.path, r.label, projectRoot),
          });
          return [
            ...prev,
            ...rest.filter((r) => !seen.has(r.path)).map(normalize),
            ...(logPath && primary.path !== logPath && !seen.has(primary.path)
              ? [normalize(primary)]
              : []),
          ];
        });
      }
      toast.success(`Loaded ${loaded.length} simulation log(s) from output portfolio`, {
        id: progressId,
      });
    } catch (err) {
      toast.error("Portfolio load failed", { description: String(err) });
    } finally {
      setPortfolioLoading(false);
    }
  }, [projectRoot, logPath, loadLog, handoff]);

  const removeComparisonRun = useCallback((path: string) => {
    setComparisonRuns((prev) => prev.filter((r) => r.path !== path));
  }, []);

  const filteredEntries = useMemo(
    () => filterEntries(entries, policy, sampleId),
    [entries, policy, sampleId]
  );

  const routeVizDays = useMemo(
    () => [...new Set(filteredEntries.map((e) => e.day))].sort((a, b) => a - b),
    [filteredEntries]
  );

  useEffect(() => {
    if (routeVizDays.length === 0) return;
    setRouteVizDay((d) => {
      if (routeVizDays.includes(d)) return d;
      return routeVizDays[routeVizDays.length - 1];
    });
  }, [routeVizDays]);

  const stats = useMemo(() => aggregateByPolicy(filteredEntries), [filteredEntries]);
  const policies = useMemo(() => Object.keys(stats), [stats]);

  const logMeta = useMemo(() => parseLogPath(logPath), [logPath]);
  const policyMeta = useMemo(() => {
    const map: Record<string, PolicyMeta> = {};
    for (const p of policies) map[p] = parsePolicyLabel(p);
    return map;
  }, [policies]);

  const handlePolicyClick = useCallback((p: string) => {
    setBrushed((prev) => toggleBrush(prev, p));
  }, []);

  const handleBrushPolicies = useCallback(
    (ps: string[] | null) => {
      if (!ps?.length) {
        setBrushed(null);
        return;
      }
      setBrushed(ps.length >= policies.length ? null : ps);
    },
    [policies.length]
  );

  const effectiveBrushed = useMemo(() => {
    if (overflowMax == null) return brushed;
    const corridor = policies.filter((p) => mean(stats[p].overflows) <= overflowMax);
    if (!brushed || brushed.length === 0) return corridor;
    return corridor.filter((p) => brushed.includes(p));
  }, [brushed, overflowMax, policies, stats]);

  const routeVizPolicies = useMemo(() => {
    if (effectiveBrushed?.length) return effectiveBrushed;
    return policies;
  }, [effectiveBrushed, policies]);

  const routeVizEntries = useMemo(
    () =>
      routeVizPolicies
        .map((p) => filteredEntries.find((e) => e.policy === p && e.day === routeVizDay))
        .filter((e): e is DayLogEntry => e != null && (e.data.all_bin_coords?.length ?? 0) > 0),
    [filteredEntries, routeVizPolicies, routeVizDay]
  );

  const routeVizHasFailureData = useMemo(
    () =>
      routeVizEntries.some(
        (e) =>
          e.data.failure_analysis?.has_failure &&
          ((e.data.failure_analysis.overflow_bins?.length ?? 0) > 0 ||
            (e.data.failure_analysis.skipped_high_fill_bins?.length ?? 0) > 0)
      ),
    [routeVizEntries]
  );

  const binMatrixData = useMemo(() => {
    const primaryPol = routeVizPolicies[0] ?? policies[0];
    if (!primaryPol) return null;
    const polEntries = filteredEntries
      .filter((e) => e.policy === primaryPol && e.data.bin_state_c?.length)
      .sort((a, b) => a.day - b.day);
    if (polEntries.length === 0) return null;
    const numBins = polEntries[0]?.data.bin_state_c?.length ?? 0;
    if (numBins === 0) return null;
    return {
      policy: primaryPol,
      dayLabels: polEntries.map((e) => `D${e.day}`),
      binLabels: Array.from({ length: numBins }, (_, i) => `B${i}`),
      matrix: polEntries.map((e) => e.data.bin_state_c ?? []),
    };
  }, [filteredEntries, policies, routeVizPolicies]);

  const maxOverflow = useMemo(
    () => Math.max(...policies.map((p) => mean(stats[p].overflows)), 0),
    [policies, stats]
  );

  const overflowGroups = useMemo(
    () =>
      buildGroupedStats(
        policies,
        stats,
        (p) => policyMeta[p]?.selectionStrategy ?? "—",
        "overflows"
      ),
    [policies, stats, policyMeta]
  );

  const kgkmGroups = useMemo(
    () =>
      buildGroupedStats(
        policies,
        stats,
        (p) => policyMeta[p]?.constructor ?? "—",
        "kg/km"
      ),
    [policies, stats, policyMeta]
  );

  const allRuns = useMemo((): ComparisonRun[] => {
    const runs: ComparisonRun[] = [];
    if (logPath && entries.length > 0) {
      runs.push({
        path: logPath,
        label: portfolioRunLabel(logPath, undefined, projectRoot),
        entries: filteredEntries,
      });
    }
    for (const r of comparisonRuns) {
      if (r.path === logPath) continue;
      runs.push({
        ...r,
        label: portfolioRunLabel(r.path, r.label, projectRoot),
        entries: filterEntries(r.entries, policy, sampleId),
      });
    }
    return runs;
  }, [logPath, entries, filteredEntries, comparisonRuns, policy, sampleId, projectRoot]);

  const portfolioMode = allRuns.length >= 2;

  const portfolioHierarchyRuns = useMemo((): PortfolioHierarchyRun[] | undefined => {
    if (!portfolioMode) return undefined;
    return allRuns.map((run) => {
      const runStats = aggregateByPolicy(run.entries);
      const runPolicies = Object.keys(runStats);
      const runMeta: Record<string, PolicyMeta> = {};
      for (const p of runPolicies) runMeta[p] = parsePolicyLabel(p);
      return {
        path: run.path,
        policies: runPolicies,
        stats: runStats,
        policyMeta: runMeta,
      };
    });
  }, [allRuns, portfolioMode]);

  const paretoByPanel = useMemo(() => buildParetoByPanel(allRuns), [allRuns]);
  const cityGroups = useMemo(() => groupRunsByCity(allRuns), [allRuns]);

  const {
    runLabels: portfolioRunLabels,
    runLabel: activeRunLabel,
    brushedRunLabels,
    handleCityClick,
    handleRunLabelClick,
  } = usePortfolioRunBrush(allRuns);

  const cityComparisonOption = useMemo(
    () =>
      cityComparisonChartOption(buildCityComparisonSeries(cityGroups), {
        logScale,
        showErrorBars,
      }),
    [cityGroups, logScale, showErrorBars]
  );

  const handlePortfolioConfigClick = useCallback(
    (policyName: string, label: string) => {
      handlePolicyClick(policyName);
      handleRunLabelClick(label);
    },
    [handlePolicyClick, handleRunLabelClick]
  );

  const onCityChartClick = useCallback(
    (params: { name?: string }) => {
      if (params.name) handleCityClick(params.name);
    },
    [handleCityClick]
  );

  const cityScaleKgkmGroups = useMemo(() => {
    if (!portfolioMode) return null;
    return cityGroups.map(([label, runs]) => {
      const vals = runs.flatMap((r) =>
        r.entries
          .map((e) => e.data["kg/km"])
          .filter((v): v is number => v != null)
      );
      return { label, policies: [] as string[], mean: mean(vals), std: std(vals) };
    });
  }, [cityGroups, portfolioMode]);

  const cityScaleOverflowGroups = useMemo(() => {
    if (!portfolioMode) return null;
    return cityGroups.map(([label, runs]) => {
      const vals = runs.flatMap((r) =>
        r.entries
          .map((e) => e.data.overflows)
          .filter((v): v is number => v != null)
      );
      return { label, policies: [] as string[], mean: mean(vals), std: std(vals) };
    });
  }, [cityGroups, portfolioMode]);

  const distributionGroups = useMemo(() => groupRunsByDistribution(allRuns), [allRuns]);

  const rankingExportData = useCallback(() => {
    const cols: MetricKey[] = ["profit", "km", "overflows", "kg"];
    const headers = [
      "policy",
      ...cols.map((c) => `mean_${c}`),
      ...cols.map((c) => `std_${c}`),
      "days",
    ];
    const rows = policies.map((p) => [
      p,
      ...cols.map((c) => mean(stats[p][c]).toFixed(4)),
      ...cols.map((c) => std(stats[p][c]).toFixed(4)),
      stats[p].days,
    ]);
    return { headers, rows };
  }, [policies, stats]);

  const exportRankingCsv = useCallback(() => {
    const { headers, rows } = rankingExportData();
    downloadCsv("simulation-ranking.csv", headers, rows);
  }, [rankingExportData]);

  const exportRankingParquet = useCallback(async () => {
    if (!projectRoot) {
      toast.error("Set project root in Settings to export Parquet");
      return;
    }
    const { headers, rows } = rankingExportData();
    setParquetExporting(true);
    try {
      const out = await downloadParquetTable(
        projectRoot,
        "simulation-ranking.parquet",
        headers,
        rows
      );
      if (out) toast.success("Parquet export complete", { description: out.split("/").pop() });
    } catch (err) {
      toast.error("Parquet export failed", { description: String(err) });
    } finally {
      setParquetExporting(false);
    }
  }, [projectRoot, rankingExportData]);

  return {
    theme, projectRoot, duckdbReady, duckdbLoading, lastPipeline, subtab, setSubtab,
    parquetExporting, logScale, showErrorBars, setShowErrorBars, brushed, setBrushed,
    heatmapMode, setHeatmapMode, overflowMax, setOverflowMax, portfolioLoading, entries,
    logPath, derivedRunLabel, comparisonRuns, routeVizDay, setRouteVizDay,
    showFailureOverlay, setShowFailureOverlay, showRouteDiff, setShowRouteDiff,
    allDuckDbLogs, openLog, addComparisonRun, loadOutputPortfolio, removeComparisonRun,
    filteredEntries, routeVizDays, stats, policies, logMeta, policyMeta,
    handlePolicyClick, handleBrushPolicies, effectiveBrushed, routeVizEntries,
    routeVizHasFailureData, binMatrixData, maxOverflow, overflowGroups, kgkmGroups,
    allRuns, portfolioMode, portfolioHierarchyRuns, paretoByPanel, cityGroups,
    portfolioRunLabels, activeRunLabel, brushedRunLabels, cityComparisonOption,
    handlePortfolioConfigClick, onCityChartClick, cityScaleKgkmGroups,
    cityScaleOverflowGroups, distributionGroups, exportRankingCsv, exportRankingParquet,
  };
}
