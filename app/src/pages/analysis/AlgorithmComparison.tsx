/**
 * Algorithm Comparison — side-by-side policy metric comparison.
 * Ports Streamlit `algorithms` mode.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { Map } from "lucide-react";
import { SubtabBar } from "../../components/ui/telemetry/SubtabBar";
import { OrionRadarChart } from "../../components/ui/orion/OrionRadarChart";
import { OrionOrbitalBubbleChart } from "../../components/ui/orion/OrionOrbitalBubbleChart";
import {
  orionPolicySubtitle,
  policyProfitBubbles,
  radarCohortFromLog,
} from "../../utils/analysis/algorithmOrion";
import { PolicyTelemetryTrendsPanel } from "../../components/analysis/telemetry/PolicyTelemetryTrendsPanel";
import { SqlQueryPanel } from "../../components/analysis/explorer/SqlQueryPanel";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { useLogPathRunLabelBrush } from "../../hooks/brush/useLogPathRunLabelBrush";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { useAppStore } from "../../store/app";
import { useDuckDbStore } from "../../store/duckdb";
import { useGlobalFiltersStore } from "../../store/filters";
import { useSimStore, filterEntries } from "../../store/sim";
import {
  formatPipelineTimingBadge,
  portfolioRunLabel,
  runSimulationArrowPipeline,
} from "../../utils/duckdb/arrowPipeline";
import { barOpacity } from "../../utils/charts/chartHighlight";
import { ChartExportButtons } from "../../components/common/ChartExportButtons";
import { errorBarBounds, radarAxisValue } from "../../utils/charts/chartLogScale";
import { symlog } from "../../utils/charts/symlog";

const ALGORITHM_SIM_TABLE = "algorithm_sim";

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

const METRICS = [
  { key: "profit", label: "Profit (€)" },
  { key: "km", label: "Distance (km)" },
  { key: "overflows", label: "Overflows" },
  { key: "kg/km", label: "Efficiency (kg/km)" },
];

const COLORS = ["#6366f1", "#34d399", "#fbbf24", "#f87171", "#818cf8", "#a3e635"];

export function AlgorithmComparison() {
  const { entries, watchPath } = useSimStore();
  const { projectRoot, setMode, setPendingMapCompare, effectiveTheme: theme } = useAppStore();
  const { handoff } = useRecentHandoff();
  const { policy, sampleId, runLabel, setPolicy } = useGlobalFiltersStore();
  useLogPathRunLabelBrush(watchPath);
  const sourceRunLabel = useMemo(
    () => (watchPath ? portfolioRunLabel(watchPath, undefined, projectRoot) : null),
    [watchPath, projectRoot]
  );
  const brushedPolicies = useMemo(() => (policy ? [policy] : null), [policy]);
  const {
    ready: duckdbReady,
    loading: duckdbLoading,
    lastPipeline,
    setLastPipeline,
    setLoading: setDuckdbLoading,
  } = useDuckDbStore();
  const radarRef = useRef<ReactECharts>(null);
  const barRefs = useRef<Record<string, EChartsReact | null>>({});
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const [showErrorBars, setShowErrorBars] = useState(false);
  const [subtab, setSubtab] = useState<"charts" | "orion">("charts");

  useEffect(() => {
    if (!duckdbReady || !watchPath) return;
    setDuckdbLoading(true);
    runSimulationArrowPipeline(watchPath, ALGORITHM_SIM_TABLE, projectRoot)
      .then(setLastPipeline)
      .catch((err) => console.warn("Algorithm Arrow pipeline:", err))
      .finally(() => setDuckdbLoading(false));
  }, [watchPath, duckdbReady, projectRoot, setLastPipeline, setDuckdbLoading]);

  const filtered = useMemo(
    () => filterEntries(entries, policy, sampleId),
    [entries, policy, sampleId]
  );
  const policies = useMemo(() => [...new Set(filtered.map((e) => e.policy))], [filtered]);
  const orionCohort = useMemo(() => radarCohortFromLog(filtered), [filtered]);
  const orionBubbles = useMemo(() => policyProfitBubbles(filtered), [filtered]);

  const radarOption = useMemo(() => {
    const metricMeans: Record<string, Record<string, number>> = {};
    for (const p of policies) {
      metricMeans[p] = {};
      for (const { key } of METRICS) {
        const vals = filtered
          .filter((e) => e.policy === p)
          .map((e) => (e.data as Record<string, number>)[key] ?? 0);
        metricMeans[p][key] = mean(vals);
      }
    }

    const displayValue = (key: string, raw: number) => radarAxisValue(raw, key, logScale);
    const maxes = METRICS.map(({ key }) =>
      Math.max(...policies.map((p) => displayValue(key, metricMeans[p][key] ?? 0)), 1)
    );

    return {
      backgroundColor: "transparent",
      legend: { data: policies, textStyle: { color: "#9090b0" } },
      radar: {
        indicator: METRICS.map(({ label }, i) => ({ name: label, max: maxes[i] * 1.1 })),
        axisLine: { lineStyle: { color: "#2d2d50" } },
        splitLine: { lineStyle: { color: "#2d2d50" } },
        name: { textStyle: { color: "#9090b0" } },
      },
      series: [
        {
          type: "radar",
          data: policies.map((p, i) => ({
            name: p,
            value: METRICS.map(({ key }) => displayValue(key, metricMeans[p][key] ?? 0)),
            lineStyle: {
              color: COLORS[i % COLORS.length],
              opacity: barOpacity(p, brushedPolicies),
            },
            areaStyle: { color: `${COLORS[i % COLORS.length]}20` },
          })),
        },
      ],
      tooltip: {},
    };
  }, [filtered, policies, brushedPolicies, logScale]);

  const openOnMap = useCallback(() => {
    setPendingMapCompare({
      policies,
      layout: policies.length === 2 ? "split" : "overlay",
      mapMode: "deckgl",
    });
    // Prefer log handoff so Digital Twin reloads + recents stay in sync (§G.16 / §D.7).
    if (watchPath) {
      handoff(watchPath, "log", { mode: "simulation" });
    } else {
      setMode("simulation");
    }
  }, [policies, watchPath, handoff, setMode, setPendingMapCompare]);

  const handlePolicyClick = useCallback(
    (name: string) => {
      setPolicy(policy === name ? null : name);
    },
    [policy, setPolicy]
  );

  const onChartClick = useCallback(
    (params: { name?: string; seriesName?: string }) => {
      const name = params.name ?? params.seriesName;
      if (name) handlePolicyClick(name);
    },
    [handlePolicyClick]
  );

  if (entries.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-canvas-muted text-sm">
        Load a simulation log in the Simulation Monitor to compare algorithms here.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <GlobalFilterBar
        runLabels={sourceRunLabel ? [sourceRunLabel] : []}
        showLogScale
      />

      {watchPath && (
        <div className="flex items-center gap-3 flex-wrap">
          <OpenPathToolbar
            path={watchPath}
            projectRoot={projectRoot}
            kind="log"
            labeled
            labeledTargets={["summary"]}
            labeledIconSize={12}
            order="chip-first"
            trailing={
              !duckdbLoading && lastPipeline?.tableName === ALGORITHM_SIM_TABLE ? (
                <span className="shrink-0">· {formatPipelineTimingBadge(lastPipeline)}</span>
              ) : null
            }
          />
        </div>
      )}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <SubtabBar
          tabs={[
            { id: "charts", label: "Standard Metrics" },
            { id: "orion", label: "Orion Radar & Orbital View" },
          ]}
          activeId={subtab}
          onChange={setSubtab}
        />
        <div className="flex items-center gap-2">
          <button onClick={openOnMap} className="btn-ghost text-xs flex items-center gap-1.5">
            <Map size={12} />
            Compare on Map
          </button>
          {subtab === "charts" && (
            <button
              onClick={() => setShowErrorBars((v) => !v)}
              className={`btn-ghost text-xs ${showErrorBars ? "text-accent-secondary" : ""}`}
            >
              {showErrorBars ? "Error bars (on)" : "Error bars (off)"}
            </button>
          )}
        </div>
      </div>

      {subtab === "orion" ? (
        <div className="space-y-4">
          {orionCohort.length === 0 ? (
            <p className="text-sm text-canvas-muted">
              This log has no policy metrics for the radar.
            </p>
          ) : (
            <OrionRadarChart
              cohort={orionCohort}
              title="Direction-Aware Multi-Objective Policy Radar"
              subtitle={orionPolicySubtitle(orionCohort.map((entry) => entry.policy))}
              height={340}
            />
          )}
          {orionBubbles.length === 0 ? (
            <p className="text-sm text-canvas-muted">
              This log has no profit values for the orbital chart.
            </p>
          ) : (
            <OrionOrbitalBubbleChart
              bubbles={orionBubbles}
              title="Policy profit"
              subtitle="Bubble size is profit. A policy with no profit is omitted."
              height={340}
            />
          )}
        </div>
      ) : (
        <>
          <div className="card">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs text-canvas-muted">
            Radar — {logScale ? "log-normalised" : "normalised"} average metrics per policy
          </p>
          <ChartExportButtons chartRef={radarRef} filenameStem="algorithm-radar" />
        </div>
        <ReactECharts
          ref={radarRef}
          option={radarOption}
          style={{ height: 340 }}
          onEvents={{ click: onChartClick, legendselectchanged: onChartClick }}
        />
      </div>

      <p className="text-[10px] text-canvas-muted">
        {logScale
          ? "Log-scale bars — profit · km · symlog-overflows · kg/km per policy"
          : "Linear bars — profit · km · overflows · kg/km per policy"}
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {METRICS.map(({ key, label }) => {
          const symlogOverflows = logScale && key === "overflows";
          const stats = policies.map((p) => {
            const vals = filtered
              .filter((e) => e.policy === p)
              .map((e) => (e.data as Record<string, number>)[key] ?? 0);
            return { mean: mean(vals), std: std(vals) };
          });
          const option = {
            backgroundColor: "transparent",
            grid: { left: 50, right: 10, top: 10, bottom: 40 },
            xAxis: {
              type: "category",
              data: policies,
              axisLabel: { color: "#9090b0", fontSize: 9, rotate: 20 },
            },
            yAxis: {
              type: (logScale && !symlogOverflows ? "log" : "value") as "log" | "value",
              logBase: 10,
              axisLabel: { color: "#9090b0", fontSize: 10 },
              minorSplitLine: { show: false },
            },
            tooltip: {
              trigger: "axis",
              formatter: (params: unknown[]) => {
                const p = (params as Array<{ dataIndex: number; name: string }>)[0];
                const s = stats[p.dataIndex];
                return `${p.name}<br/>${fmt(s.mean)} ± ${fmt(s.std)}`;
              },
            },
            series: [
              {
                type: "bar",
                data: policies.map((p, i) => {
                  const raw = stats[i].mean;
                  const value = !logScale
                    ? raw
                    : symlogOverflows
                      ? symlog(raw)
                      : Math.max(raw, 0.001);
                  return {
                    value,
                    itemStyle: {
                      color: COLORS[i % COLORS.length],
                      opacity: barOpacity(p, brushedPolicies),
                    },
                  };
                }),
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
                        const s = stats[i];
                        const bounds = errorBarBounds(
                          s.mean,
                          s.std,
                          key,
                          logScale,
                          symlogOverflows
                        );
                        const x = api.coord([i, bounds.center])[0];
                        const yTop = api.coord([i, bounds.high])[1];
                        const yBot = api.coord([i, bounds.low])[1];
                        const cap = 5;
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
                      data: policies.map((_, i) => i),
                      z: 10,
                    },
                  ]
                : []),
            ],
          };
          return (
            <div key={key} className="card">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-canvas-muted">{label}</p>
                <ChartExportButtons
                  chartRef={{ current: barRefs.current[key] }}
                  filenameStem={`algorithm-${key}`}
                />
              </div>
              <ReactECharts
                ref={(el) => {
                  barRefs.current[key] = el;
                }}
                option={option}
                style={{ height: 140 }}
                onEvents={{ click: onChartClick }}
              />
            </div>
          );
        })}
      </div>

        </>
      )}

      {watchPath && (
        <PolicyTelemetryTrendsPanel
          theme={theme}
          logScale={logScale}
          initialPolicy={brushedPolicies?.length === 1 ? brushedPolicies[0]! : null}
          initialRunLabel={runLabel ?? sourceRunLabel}
        />
      )}

      {watchPath && duckdbReady && (
        <SqlQueryPanel
          tableName={ALGORITHM_SIM_TABLE}
          theme={theme}
          highlightPolicies={brushedPolicies}
          highlightRunLabels={sourceRunLabel ? [sourceRunLabel] : null}
          brushSqlSync
          autoRunOnBrushSync
          portfolioMode
          portfolioRunLabels={sourceRunLabel ? [sourceRunLabel] : []}
          algorithmMode
        />
      )}
    </div>
  );
}
