import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { ECHARTS_FILL_VISUAL_MAP } from "../../../utils/charts/palettes";
import { nocPanelStyle } from "../telemetry/panel";
import { NOC_COLORS } from "../telemetry/tokens";
import { ProvenanceBadge, type ProvenanceTier } from "../telemetry/gauges";

export interface BinFillMatrixCell {
  dayIdx: number; // 0..D-1
  binIdx: number; // 0..B-1
  fillRatio: number; // 0.0 .. 1.5+ (1.0 = 100% capacity overflow threshold)
}

export interface OrionDotMatrixHeatmapProps {
  title?: string;
  subtitle?: string;
  dayLabels: readonly string[];
  binLabels: readonly string[];
  /** Either flat cells or a 2D [dayIdx][binIdx] matrix of fill ratios (0..1.5+). */
  cells?: readonly BinFillMatrixCell[];
  matrix?: readonly (readonly number[])[];
  height?: number;
  provenance?: ProvenanceTier;
  onSelectCell?: (dayIdx: number, binIdx: number, fillRatio: number) => void;
}

/**
 * High-density 2D Bin Fill-Level & Daily Overflow Matrix (Slice V3, Owner Ruling #2).
 *
 * Uses Apache ECharts / HTML5 Canvas (`progressive: 5000`) to render up to
 * 350 bins × 90 days (31,500 cells) with zero SVG DOM bloat, plus a synchronized
 * daily overflow count summary bar at the top.
 */
export const OrionDotMatrixHeatmap: React.FC<OrionDotMatrixHeatmapProps> = ({
  title = "Bin Fill-Level & Overflow Matrix",
  subtitle = "Canvas-accelerated spatio-temporal fill ratio (≥100% = overflow)",
  dayLabels,
  binLabels,
  cells,
  matrix,
  height = 360,
  provenance = "simulated",
  onSelectCell,
}) => {
  const { echartsData, dailyOverflows, totalOverflows, maxFillPct } = useMemo(() => {
    const out: [number, number, number][] = [];
    const overflowByDay = new Array<number>(dayLabels.length).fill(0);
    let totalOv = 0;
    let maxRatio = 0;

    if (matrix && matrix.length > 0) {
      for (let d = 0; d < matrix.length; d++) {
        const row = matrix[d] ?? [];
        for (let b = 0; b < row.length; b++) {
          const r = row[b] ?? 0;
          out.push([d, b, Number(r.toFixed(3))]);
          if (r >= 1.0) {
            overflowByDay[d] = (overflowByDay[d] ?? 0) + 1;
            totalOv++;
          }
          if (r > maxRatio) maxRatio = r;
        }
      }
    } else if (cells && cells.length > 0) {
      for (const c of cells) {
        out.push([c.dayIdx, c.binIdx, Number(c.fillRatio.toFixed(3))]);
        if (c.fillRatio >= 1.0) {
          if (c.dayIdx >= 0 && c.dayIdx < overflowByDay.length) {
            overflowByDay[c.dayIdx] = (overflowByDay[c.dayIdx] ?? 0) + 1;
          }
          totalOv++;
        }
        if (c.fillRatio > maxRatio) maxRatio = c.fillRatio;
      }
    }

    return {
      echartsData: out,
      dailyOverflows: overflowByDay,
      totalOverflows: totalOv,
      maxFillPct: Math.round(maxRatio * 100),
    };
  }, [cells, matrix, dayLabels.length]);

  const option = useMemo(
    () => ({
      animation: false,
      tooltip: {
        position: "top",
        formatter: (params: { value?: [number, number, number] }) => {
          const val = params.value;
          if (!val) return "";
          const [dIdx, bIdx, ratio] = val;
          const dLabel = dayLabels[dIdx] ?? `Day ${dIdx + 1}`;
          const bLabel = binLabels[bIdx] ?? `Bin #${bIdx}`;
          const pct = (ratio * 100).toFixed(1);
          const status =
            ratio >= 1.0
              ? "OVERFLOW (≥100%)"
              : ratio >= 0.8
              ? "URGENT (80–99%)"
              : ratio >= 0.5
              ? "ELEVATED (50–80%)"
              : "NOMINAL (<50%)";
          return `<div style="font-family:monospace;font-size:11px">
            <strong>${bLabel}</strong> • ${dLabel}<br/>
            Fill: <strong>${pct}%</strong> (${status})
          </div>`;
        },
      },
      grid: {
        top: 18,
        right: 16,
        bottom: 48,
        left: 56,
      },
      xAxis: {
        type: "category",
        data: dayLabels,
        splitArea: { show: false },
        axisLabel: { fontSize: 10 },
      },
      yAxis: {
        type: "category",
        data: binLabels,
        splitArea: { show: false },
        axisLabel: { fontSize: 10 },
      },
      visualMap: {
        ...ECHARTS_FILL_VISUAL_MAP,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        textStyle: { fontSize: 10 },
      },
      series: [
        {
          name: "Bin Fill Ratio",
          type: "heatmap",
          data: echartsData,
          progressive: 5000,
          progressiveThreshold: 3000,
          itemStyle: {
            borderWidth: binLabels.length > 80 ? 0 : 1,
            borderColor: "rgba(9, 14, 26, 0.35)",
          },
        },
      ],
    }),
    [binLabels, dayLabels, echartsData]
  );

  const onEvents = useMemo(
    () =>
      onSelectCell
        ? {
            click: (params: { value?: [number, number, number] }) => {
              if (params.value) {
                onSelectCell(params.value[0], params.value[1], params.value[2]);
              }
            },
          }
        : undefined,
    [onSelectCell]
  );

  return (
    <div
      className="rounded-xl border p-4 space-y-3"
      style={nocPanelStyle}
      data-testid="orion-dot-matrix-heatmap"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-noc-textPrimary">
              {title}
            </h3>
            {provenance && <ProvenanceBadge tier={provenance} />}
          </div>
          <p className="text-[11px] text-noc-textMuted">{subtitle}</p>
        </div>

        <div className="flex items-center gap-4 text-xs tabular-nums">
          <div>
            <span className="text-noc-textMuted">Grid: </span>
            <span className="font-mono font-semibold text-noc-textPrimary">
              {binLabels.length} bins × {dayLabels.length}d
            </span>
          </div>
          <div>
            <span className="text-noc-textMuted">Peak Fill: </span>
            <span
              className="font-mono font-semibold"
              style={{
                color:
                  maxFillPct >= 100
                    ? NOC_COLORS.crit
                    : maxFillPct >= 80
                    ? NOC_COLORS.warn
                    : NOC_COLORS.ok,
              }}
            >
              {maxFillPct}%
            </span>
          </div>
          <div>
            <span className="text-noc-textMuted">Overflows: </span>
            <span
              className="font-mono font-semibold"
              style={{
                color: totalOverflows > 0 ? NOC_COLORS.crit : NOC_COLORS.ok,
              }}
            >
              {totalOverflows}
            </span>
          </div>
        </div>
      </div>

      {/* Synchronized daily overflow summary strip */}
      {dayLabels.length > 0 && (
        <div
          className="flex items-end gap-0.5 h-7 px-1 py-0.5 rounded bg-noc-deep border border-noc-border"
          aria-label="Daily overflow summary strip"
        >
          {dailyOverflows.map((ov, idx) => {
            const maxOv = Math.max(1, ...dailyOverflows);
            const hPct = ov > 0 ? Math.max(20, Math.round((ov / maxOv) * 100)) : 8;
            return (
              <div
                key={idx}
                title={`${dayLabels[idx] ?? `D${idx + 1}`}: ${ov} overflows`}
                className="flex-1 rounded-t transition-all"
                style={{
                  height: `${hPct}%`,
                  backgroundColor: ov > 0 ? NOC_COLORS.crit : "rgba(148,163,184,0.2)",
                }}
              />
            );
          })}
        </div>
      )}

      {echartsData.length === 0 ? (
        <div
          style={{ height: 180 }}
          className="flex items-center justify-center text-xs text-noc-textMuted border border-dashed border-noc-border rounded-lg"
        >
          No bin fill matrix data available for this run
        </div>
      ) : (
        <ReactECharts
          option={option}
          style={{ height, width: "100%" }}
          notMerge
          lazyUpdate
          onEvents={onEvents}
        />
      )}
    </div>
  );
};

export default OrionDotMatrixHeatmap;
