/**
 * Simulation Summary — post-run aggregate analytics (§G.1 / §G.16, Slice V6).
 *
 * Includes `SubtabBar` (All / Table / Trajectories / Map / Compare),
 * `useLoadedRunStore` synchronization, `OrionDotMatrixHeatmap`, and
 * direction-aware radar scaling (`1.0 = best in cohort`).
 */
import { FolderOpen } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { LoadedRunRow } from "../../components/common/LoadedRunRow";
import { SubtabBar } from "../../components/ui/telemetry/SubtabBar";
import {
  OrionDotMatrixHeatmap,
  OrionHoneycombMap,
  getSampleRoutingOrionData,
} from "../../components/ui/orion";
import {
  formatPipelineTimingBadge,
  portfolioRunLabel,
} from "../../utils/duckdb/arrowPipeline";
import { PolicyTelemetryTrendsPanel } from "../../components/analysis/telemetry/PolicyTelemetryTrendsPanel";
import { SqlQueryPanel } from "../../components/analysis/explorer/SqlQueryPanel";
import {
  ConfigMetaBanner,
  PolicyBrushBar,
  RankingTable,
} from "../../components/analysis/summary/SummaryCoreWidgets";
import { TrajectoryChart } from "../../components/analysis/summary/SummaryComparisonCharts";
import {
  SummaryPortfolioAndCompareSection,
  SummaryRouteSection,
} from "../../components/analysis/summary/SummaryPortfolioSections";
import {
  SUMMARY_SIM_TABLE,
  useSimulationSummaryController,
} from "../../components/analysis/summary/useSimulationSummaryController";

export function SimulationSummary() {
  const c = useSimulationSummaryController();

  const showTable = c.subtab === "all" || c.subtab === "table";
  const showTrajectories = c.subtab === "all" || c.subtab === "trajectories";
  const showMap = c.subtab === "all" || c.subtab === "map";
  const showCompare = c.subtab === "all" || c.subtab === "compare";

  return (
    <div className="space-y-4">
      <GlobalFilterBar
        runLabels={
          c.portfolioMode
            ? c.portfolioRunLabels
            : c.derivedRunLabel
            ? [c.derivedRunLabel]
            : []
        }
        cities={c.portfolioMode ? c.cityGroups.map(([city]) => city) : []}
        showLogScale
      />

      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={c.openLog} className="btn-primary flex items-center gap-2">
          <FolderOpen size={14} />
          Open Log File
        </button>
        {c.logPath && (
          <>
            <button
              onClick={() => void c.addComparisonRun()}
              className="btn-ghost flex items-center gap-2 text-xs"
            >
              <FolderOpen size={14} />
              Add comparison log
            </button>
            {c.projectRoot && (
              <button
                onClick={() => void c.loadOutputPortfolio()}
                disabled={c.portfolioLoading}
                className="btn-ghost flex items-center gap-2 text-xs"
              >
                <FolderOpen size={14} />
                {c.portfolioLoading ? "Scanning output…" : "Load output portfolio"}
              </button>
            )}
            <OpenPathToolbar
              path={c.logPath}
              projectRoot={c.projectRoot}
              kind="log"
              labeled
              labeledTargets={["monitor"]}
              trailing={
                <>
                  {c.duckdbLoading && <span className="shrink-0">· DuckDB ingesting…</span>}
                  {!c.duckdbLoading && c.lastPipeline?.tableName === SUMMARY_SIM_TABLE && (
                    <span className="shrink-0">· {formatPipelineTimingBadge(c.lastPipeline)}</span>
                  )}
                </>
              }
            />
          </>
        )}
        {c.allRuns.length > 1 && (
          <span className="text-xs text-canvas-muted">{c.allRuns.length} runs loaded</span>
        )}
      </div>

      {c.comparisonRuns.length > 0 && (
        <div className="card">
          <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider mb-2">
            Comparison Runs
          </p>
          <div className="space-y-1">
            {c.comparisonRuns.map((r) => (
              <LoadedRunRow
                key={r.path}
                path={r.path}
                projectRoot={c.projectRoot}
                label={portfolioRunLabel(r.path, r.label, c.projectRoot)}
                activeRunLabel={c.activeRunLabel}
                onRemove={() => c.removeComparisonRun(r.path)}
                pathHandoffs
                trailing={
                  <span className="ml-auto text-canvas-muted shrink-0">{r.entries.length} days</span>
                }
              />
            ))}
          </div>
        </div>
      )}

      {c.entries.length === 0 && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Load a completed simulation log to view summary analytics.
        </div>
      )}

      {c.policies.length > 0 && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <SubtabBar
              tabs={[
                { id: "all", label: "All Panels" },
                { id: "table", label: "Table", count: c.policies.length },
                { id: "trajectories", label: "Trajectories" },
                { id: "map", label: "Map" },
                { id: "compare", label: "Compare" },
              ]}
              activeId={c.subtab}
              onChange={c.setSubtab}
            />
          </div>

          {c.logPath && (
            <ConfigMetaBanner
              logPath={c.logPath}
              logMeta={c.logMeta}
              projectRoot={c.projectRoot}
            />
          )}

          <PolicyBrushBar
            policies={c.policies}
            brushed={c.brushed}
            onToggle={c.handlePolicyClick}
            onClear={() => c.setBrushed(null)}
          />

          {showTable && (
            <RankingTable
              stats={c.stats}
              policies={c.policies}
              onExport={c.exportRankingCsv}
              onExportParquet={c.projectRoot ? c.exportRankingParquet : undefined}
              parquetExporting={c.parquetExporting}
            />
          )}

          {showTrajectories && (
            <>
              <TrajectoryChart
                entries={c.filteredEntries}
                policies={c.policies}
                brushed={c.effectiveBrushed}
                logScale={c.logScale}
              />
              {c.binMatrixData && (
                <OrionDotMatrixHeatmap
                  title={`Bin Fill-Level & Overflow Matrix — ${c.binMatrixData.policy}`}
                  dayLabels={c.binMatrixData.dayLabels}
                  binLabels={c.binMatrixData.binLabels}
                  matrix={c.binMatrixData.matrix}
                />
              )}
            </>
          )}

          {showMap && (
            <div className="space-y-4">
              <SummaryRouteSection
                routeVizDays={c.routeVizDays}
                routeVizDay={c.routeVizDay}
                setRouteVizDay={c.setRouteVizDay}
                routeVizHasFailureData={c.routeVizHasFailureData}
                showFailureOverlay={c.showFailureOverlay}
                setShowFailureOverlay={c.setShowFailureOverlay}
                routeVizEntries={c.routeVizEntries}
                showRouteDiff={c.showRouteDiff}
                setShowRouteDiff={c.setShowRouteDiff}
              />
              <OrionHoneycombMap
                zones={getSampleRoutingOrionData().honeycombZones}
                trajectories={getSampleRoutingOrionData().zoneTrajectories}
                title="Regional Waste Zone Honeycomb Matrix & Tour Corridors"
                subtitle="Spatial bin fill density & vehicle tour trajectories across collection sectors"
                height={340}
              />
            </div>
          )}

          {showCompare && (
            <SummaryPortfolioAndCompareSection
              stats={c.stats}
              policies={c.policies}
              policyMeta={c.policyMeta}
              logMeta={c.logMeta}
              portfolioRuns={c.portfolioHierarchyRuns}
              allRuns={c.allRuns}
              portfolioMode={c.portfolioMode}
              effectiveBrushed={c.effectiveBrushed}
              handleBrushPolicies={c.handleBrushPolicies}
              handlePolicyClick={c.handlePolicyClick}
              handlePortfolioConfigClick={c.handlePortfolioConfigClick}
              showErrorBars={c.showErrorBars}
              setShowErrorBars={c.setShowErrorBars}
              logScale={c.logScale}
              heatmapMode={c.heatmapMode}
              setHeatmapMode={c.setHeatmapMode}
              paretoByPanel={c.paretoByPanel}
              cityScaleOverflowGroups={c.cityScaleOverflowGroups}
              overflowGroups={c.overflowGroups}
              cityScaleKgkmGroups={c.cityScaleKgkmGroups}
              kgkmGroups={c.kgkmGroups}
              distributionGroups={c.distributionGroups}
              cityGroups={c.cityGroups}
              cityComparisonOption={c.cityComparisonOption}
              onCityChartClick={c.onCityChartClick}
              maxOverflow={c.maxOverflow}
              overflowMax={c.overflowMax}
              setOverflowMax={c.setOverflowMax}
            />
          )}

          {c.logPath && (
            <PolicyTelemetryTrendsPanel
              theme={c.theme}
              logScale={c.logScale}
              initialPolicy={c.effectiveBrushed?.length === 1 ? c.effectiveBrushed[0]! : null}
              initialRunLabel={
                c.derivedRunLabel ??
                c.activeRunLabel ??
                (c.brushedRunLabels?.length === 1 ? c.brushedRunLabels[0]! : null)
              }
            />
          )}

          {c.logPath && c.duckdbReady && (
            <SqlQueryPanel
              tableName={SUMMARY_SIM_TABLE}
              theme={c.theme}
              highlightPolicies={c.effectiveBrushed}
              highlightRunLabels={c.brushedRunLabels}
              brushSqlSync
              autoRunOnBrushSync
              portfolioMode={c.allDuckDbLogs.length > 1}
              portfolioRunLabels={c.portfolioMode ? c.portfolioRunLabels : []}
            />
          )}
        </>
      )}
    </div>
  );
}
