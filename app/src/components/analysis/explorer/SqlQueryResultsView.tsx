import { useCallback, useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { PivotTablePanel } from "./PivotTablePanel";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import {
  heatmapCellLabels,
  type AutoChartSpec,
  type AutoChartType,
} from "../../../utils/charts/queryAutoChart";

interface SqlQueryResultsViewProps {
  chartSpec: AutoChartSpec | null;
  chartOption: Record<string, unknown> | null;
  chartAlternatives: AutoChartSpec[];
  setChartTypeOverride: (type: AutoChartType | null) => void;
  sortedRows: Record<string, unknown>[];
  filteredRows: Record<string, unknown>[];
  rows: Record<string, unknown>[];
  columns: string[];
  logScale: boolean;
  handlePivotCrossFilter: (rowKey: string, label: string) => void;
  highlightPolicies: string[] | null;
  highlightRunLabels: string[] | null;
  highlightCityScale: string[] | null;
  filterText: string;
  setFilterText: (text: string) => void;
  sortCol: string | null;
  sortDir: "asc" | "desc";
  toggleSort: (col: string) => void;
  rowMatchesHighlight: (row: Record<string, unknown>) => boolean;
  policyCol: string | null;
  runLabelCol: string | null;
  cityScaleCol: string | null;
  activePolicy: string | null;
  activeRunLabel: string | null;
  brushedCity: string | null;
  applyCrossFilter: (col: string, value: string) => void;
  onDaySelect?: (day: number) => void;
}

export function SqlQueryResultsView({
  chartSpec,
  chartOption,
  chartAlternatives,
  setChartTypeOverride,
  sortedRows,
  filteredRows,
  rows,
  columns,
  logScale,
  handlePivotCrossFilter,
  highlightPolicies,
  highlightRunLabels,
  highlightCityScale,
  filterText,
  setFilterText,
  sortCol,
  sortDir,
  toggleSort,
  rowMatchesHighlight,
  policyCol,
  runLabelCol,
  cityScaleCol,
  activePolicy,
  activeRunLabel,
  brushedCity,
  applyCrossFilter,
  onDaySelect,
}: SqlQueryResultsViewProps) {
  const autoChartRef = useRef<ReactECharts>(null);
  const dayCol = columns.find((c) => /^day$/i.test(c));

  const onAutoChartClick = useCallback(
    (params: {
      componentType?: string;
      name?: string;
      seriesName?: string;
      value?: unknown;
    }) => {
      if (!chartSpec || params.componentType !== "series") return;
      if (chartSpec.type === "bar") {
        applyCrossFilter(chartSpec.xKey, String(params.name ?? ""));
        return;
      }
      if (chartSpec.type === "grouped-bar" && chartSpec.seriesKey) {
        if (params.seriesName) {
          applyCrossFilter(chartSpec.seriesKey, String(params.seriesName));
        }
        if (params.name) {
          applyCrossFilter(chartSpec.xKey, String(params.name));
        }
        return;
      }
      if (chartSpec.type === "heatmap" && chartSpec.seriesKey) {
        const val = params.value as [number, number, number] | undefined;
        if (!val) return;
        const { colLabel, rowLabel } = heatmapCellLabels(
          chartSpec,
          rows,
          val[0],
          val[1]
        );
        if (colLabel) applyCrossFilter(chartSpec.xKey, colLabel);
        if (rowLabel) applyCrossFilter(chartSpec.seriesKey, rowLabel);
        return;
      }
      if (chartSpec.type === "scatter" && chartSpec.labelKey) {
        if (params.seriesName === "Pareto front") return;
        const label =
          params.name ??
          (typeof params.value === "object" &&
          params.value != null &&
          "name" in (params.value as object)
            ? String((params.value as { name?: string }).name ?? "")
            : "");
        if (label) applyCrossFilter(chartSpec.labelKey, label);
        return;
      }
      if (chartSpec.type === "line") {
        const raw =
          typeof params.value === "object" &&
          params.value != null &&
          Array.isArray(params.value)
            ? params.value[0]
            : params.name;
        const value = String(raw ?? "");
        if (/^(policy|run_label|city_scale)$/i.test(chartSpec.xKey)) {
          applyCrossFilter(chartSpec.xKey, value);
          return;
        }
        if (/^day$/i.test(chartSpec.xKey) && onDaySelect != null) {
          const day = Number(value);
          if (Number.isFinite(day)) onDaySelect(day);
        }
      }
    },
    [chartSpec, rows, applyCrossFilter, onDaySelect]
  );

  const autoChartCrossFilterHint = useMemo(() => {
    if (!chartSpec) return false;
    if (chartSpec.type === "bar") {
      return /^(policy|run_label|city_scale)$/i.test(chartSpec.xKey);
    }
    if (chartSpec.type === "grouped-bar" || chartSpec.type === "heatmap") {
      return Boolean(chartSpec.seriesKey);
    }
    if (chartSpec.type === "scatter" && chartSpec.labelKey) {
      return /^(policy|run_label|city_scale)$/i.test(chartSpec.labelKey);
    }
    if (chartSpec.type === "line") {
      return (
        /^(policy|run_label|city_scale|day)$/i.test(chartSpec.xKey) &&
        (onDaySelect != null || !/^day$/i.test(chartSpec.xKey))
      );
    }
    return false;
  }, [chartSpec, onDaySelect]);

  return (
    <>
      {chartSpec && chartOption && (
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-xs text-canvas-muted">
              Auto-chart (§G.6):{" "}
              <span className="text-gray-300">{chartSpec.label}</span>
            </p>
            {chartAlternatives.length > 1 && (
              <div className="flex flex-wrap gap-1">
                {chartAlternatives.map((alt) => (
                  <button
                    key={alt.type}
                    type="button"
                    onClick={() => setChartTypeOverride(alt.type)}
                    className={`text-[10px] px-2 py-0.5 rounded border transition-colors ${
                      chartSpec.type === alt.type
                        ? "border-accent-primary bg-accent-primary/15 text-accent-secondary"
                        : "border-canvas-border text-canvas-muted hover:text-gray-200"
                    }`}
                  >
                    {alt.type}
                  </button>
                ))}
              </div>
            )}
            <ChartExportButtons
              chartRef={autoChartRef}
              filenameStem="auto-chart"
              size={11}
              className="py-0.5"
            />
          </div>
          <ReactECharts
            ref={autoChartRef}
            option={chartOption}
            style={{ height: 200 }}
            onEvents={
              autoChartCrossFilterHint ? { click: onAutoChartClick } : undefined
            }
          />
          {autoChartCrossFilterHint && (
            <p className="text-[10px] text-canvas-muted">
              Click a bar, line point, scatter point, or heatmap cell to
              cross-filter linked panels.
            </p>
          )}
        </div>
      )}

      {sortedRows.length > 0 && (
        <PivotTablePanel
          columns={columns}
          rows={rows}
          logScale={logScale}
          onRowClick={handlePivotCrossFilter}
          highlightPolicyLabels={highlightPolicies}
          highlightRunLabels={highlightRunLabels}
          highlightCityScaleLabels={highlightCityScale}
        />
      )}

      {sortedRows.length > 0 && (
        <input
          type="search"
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          placeholder="Filter rows…"
          className="input-base w-full max-w-xs text-xs"
        />
      )}

      {filteredRows.length > 0 && (
        <div className="overflow-auto max-h-64 rounded-lg border border-canvas-border">
          <table className="w-full text-xs">
            <thead className="bg-canvas-elevated sticky top-0">
              <tr>
                {columns.map((c) => (
                  <th
                    key={c}
                    onClick={() => toggleSort(c)}
                    className="px-3 py-2 text-left text-canvas-muted font-medium whitespace-nowrap cursor-pointer hover:text-gray-200"
                  >
                    <span className="inline-flex items-center gap-1">
                      {c}
                      {sortCol === c &&
                        (sortDir === "asc" ? (
                          <ChevronUp size={10} />
                        ) : (
                          <ChevronDown size={10} />
                        ))}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-canvas-border">
              {filteredRows.slice(0, 200).map((row, i) => {
                const highlighted = rowMatchesHighlight(row);
                const isActive =
                  (policyCol &&
                    activePolicy &&
                    String(row[policyCol] ?? "") === activePolicy) ||
                  (runLabelCol &&
                    activeRunLabel &&
                    String(row[runLabelCol] ?? "") === activeRunLabel) ||
                  (cityScaleCol &&
                    brushedCity &&
                    String(row[cityScaleCol] ?? "") === brushedCity);
                return (
                  <tr
                    key={i}
                    className={`hover:bg-canvas-hover ${
                      isActive
                        ? "bg-accent-primary/15 ring-1 ring-inset ring-accent-primary/40"
                        : highlighted
                          ? ""
                          : "opacity-35"
                    }`}
                  >
                    {columns.map((c) => {
                      const isBrushCol =
                        c === policyCol ||
                        c === runLabelCol ||
                        c === cityScaleCol;
                      const isDayCol = c === dayCol && onDaySelect != null;
                      return (
                        <td
                          key={c}
                          onClick={() => {
                            if (isBrushCol) {
                              applyCrossFilter(c, String(row[c] ?? ""));
                              return;
                            }
                            if (isDayCol && row[c] != null) {
                              const day = Number(row[c]);
                              if (Number.isFinite(day)) onDaySelect(day);
                            }
                          }}
                          className={`px-3 py-1.5 text-gray-300 whitespace-nowrap font-mono ${
                            isBrushCol || isDayCol
                              ? "cursor-pointer hover:text-accent-secondary"
                              : ""
                          }`}
                        >
                          {String(row[c] ?? "—")}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
