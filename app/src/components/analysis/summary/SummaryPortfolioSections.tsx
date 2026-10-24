import { useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import { BenchmarkParetoPanel } from "../benchmark/BenchmarkParetoPanel";
import { BenchmarkGraphHeatmap } from "../benchmark/BenchmarkGraphHeatmap";
import { BenchmarkDistributionHeatmap } from "../benchmark/BenchmarkDistributionHeatmap";
import { BenchmarkPortfolioHeatmap } from "../benchmark/BenchmarkPortfolioHeatmap";
import { BenchmarkPortfolioParallel } from "../benchmark/BenchmarkPortfolioParallel";
import { PortfolioEfficiencyRanking } from "../benchmark/PortfolioEfficiencyRanking";
import { PARETO_PANELS } from "../../../utils/benchmark/paretoPanels";
import type { LogPathMeta, PolicyMeta } from "../../../utils/sim/simMetadata";
import type { HeatmapMode } from "../../../utils/benchmark/heatmapMetrics";
import type {
  ParetoPoint,
  PortfolioRunSlice,
} from "../../../utils/benchmark/paretoPortfolio";
import type { PortfolioHierarchyRun } from "../../../utils/benchmark/policyHierarchy";
import {
  fmt,
  GroupedMetricBarChart,
  mean,
  std,
  type GroupRow,
  type MetricKey,
  type PolicyStats,
} from "./SummaryCoreWidgets";
import { PolicyRadarChart } from "./SummaryComparisonCharts";
import {
  DistributionFacetHeatmaps,
  EfficiencyRankingChart,
  MetricBarChart,
  PolicyHierarchyPanel,
  PolicyParallelChart,
  PolicyParetoChart,
} from "./SummaryHierarchyAndBars";

export { SummaryRouteSection } from "./SummaryRouteSection";

export function SummaryPortfolioAndCompareSection({
  stats,
  policies,
  policyMeta,
  logMeta,
  portfolioRuns,
  allRuns,
  portfolioMode,
  effectiveBrushed,
  handleBrushPolicies,
  handlePolicyClick,
  handlePortfolioConfigClick,
  showErrorBars,
  setShowErrorBars,
  logScale,
  heatmapMode,
  setHeatmapMode,
  paretoByPanel,
  cityScaleOverflowGroups,
  overflowGroups,
  cityScaleKgkmGroups,
  kgkmGroups,
  distributionGroups,
  cityGroups,
  cityComparisonOption,
  onCityChartClick,
  maxOverflow,
  overflowMax,
  setOverflowMax,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  policyMeta: Record<string, PolicyMeta>;
  logMeta: LogPathMeta;
  portfolioRuns?: PortfolioHierarchyRun[];
  allRuns: PortfolioRunSlice[];
  portfolioMode: boolean;
  effectiveBrushed: string[] | null;
  handleBrushPolicies: (ps: string[] | null) => void;
  handlePolicyClick: (policy: string) => void;
  handlePortfolioConfigClick: (policyName: string, label: string) => void;
  showErrorBars: boolean;
  setShowErrorBars: (fn: (v: boolean) => boolean) => void;
  logScale: boolean;
  heatmapMode: HeatmapMode;
  setHeatmapMode: (mode: HeatmapMode) => void;
  paretoByPanel: Record<string, ParetoPoint[]>;
  cityScaleOverflowGroups: GroupRow[] | null;
  overflowGroups: GroupRow[];
  cityScaleKgkmGroups: GroupRow[] | null;
  kgkmGroups: GroupRow[];
  distributionGroups: Array<[string, PortfolioRunSlice[]]>;
  cityGroups: Array<[string, PortfolioRunSlice[]]>;
  cityComparisonOption: object;
  onCityChartClick: (params: { name?: string }) => void;
  maxOverflow: number;
  overflowMax: number | null;
  setOverflowMax: (val: number | null) => void;
}) {
  const cityCompareChartRef = useRef<EChartsReact | null>(null);
  const metricValues = (key: MetricKey) =>
    policies.map((p) => ({
      mean: mean(stats[p][key]),
      std: std(stats[p][key]),
    }));

  return (
    <div className="space-y-4">
      <PolicyHierarchyPanel
        stats={stats}
        policies={policies}
        policyMeta={policyMeta}
        logMeta={logMeta}
        portfolioRuns={portfolioRuns}
        brushed={effectiveBrushed}
        onBrushPolicies={handleBrushPolicies}
        showErrorBars={showErrorBars}
        logScale={logScale}
      />

      {portfolioMode && allRuns.length >= 2 && (
        <BenchmarkPortfolioParallel runs={allRuns} logScale={logScale} />
      )}

      {portfolioMode && (
        <BenchmarkPortfolioHeatmap
          runs={allRuns}
          heatmapMode={heatmapMode}
          onModeChange={setHeatmapMode}
          brushed={effectiveBrushed}
          logScale={logScale}
        />
      )}

      {allRuns.length >= 1 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-gray-300">
            Pareto Panels (§G.1.2)
          </p>
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

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {cityScaleOverflowGroups ? (
          <GroupedMetricBarChart
            title={
              logScale
                ? "Overflows by City / Scale (symlog)"
                : "Overflows by City / Scale"
            }
            subtitle="Mean ± std across portfolio runs (§G.1.1 multi-city)"
            groups={cityScaleOverflowGroups}
            color="#f87171"
            showErrorBars={showErrorBars}
            logScale={logScale}
            useSymlog
            exportName="summary-overflows-city"
          />
        ) : (
          <GroupedMetricBarChart
            title={
              logScale
                ? "Overflows by Selection Strategy (symlog)"
                : "Overflows by Selection Strategy"
            }
            subtitle="Mean ± std across constructor variants (§G.1.1)"
            groups={overflowGroups}
            color="#f87171"
            showErrorBars={showErrorBars}
            logScale={logScale}
            useSymlog
            exportName="summary-overflows-grouped"
          />
        )}
        {cityScaleKgkmGroups ? (
          <GroupedMetricBarChart
            title="kg/km by City / Scale"
            subtitle="Mean ± std across portfolio runs (§G.1.1 multi-city)"
            groups={cityScaleKgkmGroups}
            color="#34d399"
            showErrorBars={showErrorBars}
            logScale={logScale}
            metricKey="kg/km"
            exportName="summary-kgkm-city"
          />
        ) : (
          <GroupedMetricBarChart
            title="kg/km by Constructor"
            subtitle="Mean ± std across selection strategies (§G.1.1)"
            groups={kgkmGroups}
            color="#34d399"
            showErrorBars={showErrorBars}
            logScale={logScale}
            metricKey="kg/km"
            exportName="summary-kgkm-grouped"
          />
        )}
      </div>

      {portfolioMode && distributionGroups.length > 1 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-gray-300">
            Heatmaps by Distribution (§G.1.3)
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {distributionGroups.map(([dist, distRuns]) => (
              <BenchmarkDistributionHeatmap
                key={dist}
                distributionLabel={dist}
                runs={distRuns}
                heatmapMode={heatmapMode}
                logScale={logScale}
              />
            ))}
          </div>
        </div>
      )}

      {portfolioMode && cityGroups.length > 1 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-gray-300">
            Heatmaps by Graph (§G.1.3)
          </p>
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

      {portfolioMode && cityGroups.length >= 1 && (
        <div className="card space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-gray-300">
                City Comparison (§G.1.6)
              </p>
              <p className="text-[10px] text-canvas-muted">
                {logScale
                  ? `Log-scale bars — profit · symlog-overflows · kg/km${
                      showErrorBars ? " · error bars on" : ""
                    }`
                  : `Linear bars — profit · overflows · kg/km${
                      showErrorBars ? " · error bars on" : ""
                    }`}
              </p>
            </div>
            <ChartExportButtons
              chartRef={{ current: cityCompareChartRef.current }}
              filenameStem="summary-city-compare"
            />
          </div>
          <ReactECharts
            ref={cityCompareChartRef}
            option={cityComparisonOption}
            style={{ height: 240 }}
            onEvents={{ click: onCityChartClick }}
          />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PolicyRadarChart
          stats={stats}
          policies={policies}
          brushed={effectiveBrushed}
          policyMeta={policyMeta}
          logScale={logScale}
        />
        <div className="space-y-2">
          <div className="card space-y-2">
            <p className="text-xs font-semibold text-gray-300">
              Zero-overflow corridor (§G.1.4)
            </p>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={0}
                max={Math.max(maxOverflow, 1)}
                step={0.5}
                value={overflowMax ?? maxOverflow}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setOverflowMax(v >= maxOverflow ? null : v);
                }}
                className="flex-1 accent-accent-primary"
              />
              <span className="text-xs font-mono text-canvas-muted w-24 text-right tabular-nums">
                ≤ {overflowMax == null ? "off" : fmt(overflowMax, 1)}
              </span>
              {overflowMax != null && (
                <button
                  onClick={() => setOverflowMax(null)}
                  className="btn-ghost text-xs"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
          <PolicyParallelChart
            stats={stats}
            policies={policies}
            policyMeta={policyMeta}
            logMeta={logMeta}
            brushed={effectiveBrushed}
            logScale={logScale}
            onPolicyClick={handlePolicyClick}
            onAxisBrush={handleBrushPolicies}
            onOverflowCorridorBrush={(max) => setOverflowMax(max)}
          />
        </div>
      </div>

      <DistributionFacetHeatmaps
        stats={stats}
        policies={policies}
        policyMeta={policyMeta}
        heatmapMode={heatmapMode}
        onModeChange={setHeatmapMode}
        brushed={effectiveBrushed}
        logScale={logScale}
      />

      {portfolioMode && (
        <PortfolioEfficiencyRanking
          runs={allRuns}
          showErrorBars={showErrorBars}
          logScale={logScale}
          brushed={effectiveBrushed}
          onConfigClick={handlePortfolioConfigClick}
        />
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {!portfolioMode && (
          <EfficiencyRankingChart
            stats={stats}
            policies={policies}
            showErrorBars={showErrorBars}
            logScale={logScale}
            brushed={effectiveBrushed}
            onPolicyClick={handlePolicyClick}
            policyMeta={policyMeta}
          />
        )}
        <div className={portfolioMode ? "lg:col-span-2" : ""}>
          <PolicyParetoChart
            stats={stats}
            policies={policies}
            logScale={logScale}
            brushed={effectiveBrushed}
            onPolicyClick={handlePolicyClick}
            policyMeta={policyMeta}
            logMeta={logMeta}
          />
        </div>
      </div>

      <div className="flex items-center justify-end gap-2">
        <button
          onClick={() => setShowErrorBars((v) => !v)}
          className={`btn-ghost text-xs ${
            showErrorBars ? "text-accent-secondary" : ""
          }`}
        >
          {showErrorBars ? "Error bars (on)" : "Error bars (off)"}
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {(
          [
            { key: "profit" as const, title: "Avg Profit by Policy (€) — hover for ±std", color: "#6366f1", symlog: logScale, exp: "summary-profit" },
            { key: "km" as const, title: "Avg Distance by Policy (km)", color: "#38bdf8", symlog: logScale, exp: "summary-km" },
            { key: "overflows" as const, title: "Avg Overflows by Policy", color: "#f87171", symlog: true, exp: "summary-overflows" },
            { key: "kg" as const, title: "Avg Waste Collected by Policy (kg)", color: "#34d399", symlog: logScale, exp: "summary-kg" },
          ]
        ).map((m) => (
          <MetricBarChart
            key={m.key}
            title={m.title}
            policies={policies}
            values={metricValues(m.key)}
            color={m.color}
            logScale={logScale}
            useSymlog={m.symlog}
            showErrorBars={showErrorBars}
            exportName={m.exp}
            brushed={effectiveBrushed}
            onPolicyClick={handlePolicyClick}
            policyMeta={policyMeta}
            stats={stats}
          />
        ))}
      </div>

      {!logScale && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-gray-300">Log-scale views (§G.1.6)</p>
          <div className="grid grid-cols-2 gap-4">
            {(
              [
                { key: "profit" as const, title: "Avg Profit by Policy (€) — symlog scale", color: "#6366f1", exp: "summary-profit-log" },
                { key: "km" as const, title: "Avg Distance by Policy (km) — symlog scale", color: "#38bdf8", exp: "summary-km-log" },
                { key: "overflows" as const, title: "Avg Overflows by Policy — symlog scale", color: "#f87171", exp: "summary-overflows-log" },
                { key: "kg" as const, title: "Avg Waste Collected by Policy (kg) — symlog scale", color: "#34d399", exp: "summary-kg-log" },
              ]
            ).map((m) => (
              <MetricBarChart
                key={m.key}
                title={m.title}
                policies={policies}
                values={metricValues(m.key)}
                color={m.color}
                logScale
                useSymlog
                exportName={m.exp}
                brushed={effectiveBrushed}
                onPolicyClick={handlePolicyClick}
                policyMeta={policyMeta}
                stats={stats}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
