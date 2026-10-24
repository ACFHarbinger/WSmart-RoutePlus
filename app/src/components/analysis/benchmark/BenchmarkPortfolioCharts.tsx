import { useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import { PARETO_PANELS } from "../../../utils/benchmark/paretoPanels";
import type { ParetoPoint } from "../../../utils/benchmark/paretoPortfolio";
import { BenchmarkParetoPanel } from "./BenchmarkParetoPanel";
import { BenchmarkGraphHeatmap } from "./BenchmarkGraphHeatmap";
import type { HeatmapMode } from "../../../utils/benchmark/heatmapMetrics";
import { BenchmarkPortfolioParallel } from "./BenchmarkPortfolioParallel";
import { PortfolioEfficiencyRanking } from "./PortfolioEfficiencyRanking";
import { PolicyTelemetryTrendsPanel } from "../telemetry/PolicyTelemetryTrendsPanel";
import { SqlQueryPanel } from "../explorer/SqlQueryPanel";
import {
  buildBenchmarkMetricBarOption,
  SIM_METRICS,
  type RunFile,
} from "./BenchmarkEvalAndBars";

interface BenchmarkPortfolioChartsProps {
  runsCount: number;
  filteredRuns: RunFile[];
  paretoByPanel: Record<string, ParetoPoint[]>;
  cityGroups: Array<[string, RunFile[]]>;
  heatmapMode: HeatmapMode;
  setHeatmapMode: (mode: HeatmapMode) => void;
  logScale: boolean;
  showErrorBars: boolean;
  cityComparisonOption: Record<string, unknown>;
  onCityChartClick: (params: { name?: string }) => void;
  efficiencyRanking: Array<{ policy: string; mean: number; std: number }>;
  efficiencyRankOption: Record<string, unknown>;
  brushedPolicies: string[] | null;
  brushedRunLabels: string[] | null;
  handlePolicyClick: (name: string) => void;
  onChartClick: (params: { name?: string; seriesName?: string }) => void;
  theme: "dark" | "light";
  duckdbReady: boolean;
  benchmarkSimTable: string;
  portfolioRunLabels: string[];
}

export function BenchmarkPortfolioCharts({
  runsCount,
  filteredRuns,
  paretoByPanel,
  cityGroups,
  heatmapMode,
  setHeatmapMode,
  logScale,
  showErrorBars,
  cityComparisonOption,
  onCityChartClick,
  efficiencyRanking,
  efficiencyRankOption,
  brushedPolicies,
  brushedRunLabels,
  handlePolicyClick,
  onChartClick,
  theme,
  duckdbReady,
  benchmarkSimTable,
  portfolioRunLabels,
}: BenchmarkPortfolioChartsProps) {
  const chartRefs = useRef<Record<string, EChartsReact | null>>({});

  return (
    <>
      {runsCount >= 2 && (
        <BenchmarkPortfolioParallel runs={filteredRuns} logScale={logScale} />
      )}

      {runsCount >= 1 && (
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

      {runsCount >= 2 && cityGroups.length > 1 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <p className="text-xs font-semibold text-gray-300">
              Heatmaps by Graph (§G.1.3)
            </p>
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

      {runsCount >= 2 && cityGroups.length >= 1 && (
        <div className="card space-y-2">
          <p className="text-xs font-semibold text-gray-300">
            City Comparison (§G.1.6)
          </p>
          <p className="text-[10px] text-canvas-muted">
            {logScale
              ? `Log-scale bars — profit · symlog-overflows · kg/km by graph scale${
                  showErrorBars ? " · error bars on" : ""
                }`
              : `Linear bars — profit · overflows · kg/km by graph scale${
                  showErrorBars ? " · error bars on" : ""
                }`}
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

      {runsCount >= 2 && efficiencyRanking.length > 0 && (
        <PortfolioEfficiencyRanking
          runs={filteredRuns}
          showErrorBars={showErrorBars}
          logScale={logScale}
          brushed={brushedPolicies}
          onPolicyClick={handlePolicyClick}
        />
      )}

      {runsCount === 1 && efficiencyRanking.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-canvas-muted">
              Efficiency Ranking (kg/km)
            </p>
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

      {runsCount >= 1 && (
        <PolicyTelemetryTrendsPanel
          theme={theme}
          logScale={logScale}
          initialPolicy={
            brushedPolicies?.length === 1 ? brushedPolicies[0]! : null
          }
          initialRunLabel={
            brushedRunLabels?.length === 1 ? brushedRunLabels[0]! : null
          }
        />
      )}

      {runsCount >= 1 && duckdbReady && (
        <SqlQueryPanel
          tableName={benchmarkSimTable}
          theme={theme}
          highlightPolicies={brushedPolicies}
          highlightRunLabels={brushedRunLabels}
          brushSqlSync
          autoRunOnBrushSync
          portfolioMode={runsCount > 1}
          portfolioRunLabels={runsCount > 1 ? portfolioRunLabels : []}
        />
      )}

      {runsCount >= 1 && (
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
                option={buildBenchmarkMetricBarOption(
                  filteredRuns,
                  key,
                  label,
                  brushedPolicies,
                  logScale,
                  showErrorBars
                )}
                style={{ height: 220 }}
                onEvents={{
                  click: onChartClick,
                  legendselectchanged: onChartClick,
                }}
              />
            </div>
          ))}
        </div>
      )}
    </>
  );
}
