/**
 * Dataset analysis chart library (§H.2) — native ECharts ports of every
 * figure generator in `logic/gen/gen_dataset_analysis.py`.
 */
import type { SeriesOption } from "echarts";
import { DATASET_CFG, type GenTheme } from "../config";
import { axisStyle, type ChartSpec } from "./common";
import type {
  NpzStatRow,
  ExtendedRow,
} from "../data/dataset";
import {
  buildNpzViolin,
  buildNpzBox,
  buildNpzHistKde,
} from "./datasetDistributions";
import {
  buildNpzTdAlignment,
  buildTdStatsComparison,
  buildTdWasteDistributions,
} from "./datasetTd";

export {
  buildNpzViolin,
  buildNpzBox,
  buildNpzHistKde,
  buildNpzTdAlignment,
  buildTdStatsComparison,
  buildTdWasteDistributions,
};

const CITY_LABELS = DATASET_CFG.city_labels;
const DIST_LABELS = DATASET_CFG.dist_labels;
const CITY_COLORS = DATASET_CFG.city_colors;
const DIST_COLORS = DATASET_CFG.dist_colors;

export const REF_CITY = "figueiradafoz"; // single-N reference city (diamonds)
export const LINE_CITY = "riomaior"; // multi-N evolution-line city

export const distColor = (d: string) => DIST_COLORS[d] ?? "#a0a0a0";
export const distLabel = (d: string) => DIST_LABELS[d] ?? d;

// ── grid helper (nrows × ncols with a suptitle band) ─────────────────────────

export interface SubGrid {
  boxes: { left: number; top: number; width: number; height: number }[];
  grids: object[];
}

export function subGrid(nrows: number, ncols: number, opts: { legendPad?: number } = {}): SubGrid {
  const legendPad = opts.legendPad ?? 0;
  const topPad = 7;
  const availH = 100 - topPad - 6 - legendPad;
  const boxes = [];
  for (let i = 0; i < nrows * ncols; i++) {
    const r = Math.floor(i / ncols);
    const c = i % ncols;
    boxes.push({
      left: 6 + c * (92 / ncols),
      top: topPad + r * (availH / nrows) + 4,
      width: 92 / ncols - 7,
      height: availH / nrows - 10,
    });
  }
  return {
    boxes,
    grids: boxes.map((b) => ({
      left: `${b.left}%`,
      top: `${b.top}%`,
      width: `${b.width}%`,
      height: `${b.height}%`,
    })),
  };
}

export function supTitle(text: string, theme: GenTheme) {
  return {
    text,
    left: "center",
    top: 6,
    textStyle: { color: theme.fg, fontSize: 15, fontWeight: "bold" as const },
  };
}

export function subTitle(text: string, box: { left: number; width: number; top: number }, theme: GenTheme) {
  return {
    text,
    left: `${box.left + box.width / 2}%`,
    top: `${box.top - 4}%`,
    textAlign: "center" as const,
    textStyle: { color: theme.fg, fontSize: 12, fontWeight: "bold" as const },
  };
}

/** Line + reference-diamond series pair (ports _plot_line_plus_ref). */
export function linePlusRef(
  gridIndex: number,
  lineData: [number, number][],
  refData: [number, number][],
  dist: string,
  opts: { label?: string; dashed?: boolean; alpha?: number } = {}
): SeriesOption[] {
  const color = distColor(dist);
  const series: SeriesOption[] = [];
  if (lineData.length) {
    series.push({
      type: "line",
      name: opts.label ?? distLabel(dist),
      xAxisIndex: gridIndex,
      yAxisIndex: gridIndex,
      data: lineData,
      symbol: "circle",
      symbolSize: 9,
      lineStyle: { color, width: 2, type: opts.dashed ? "dashed" : "solid", opacity: opts.alpha ?? 1 },
      itemStyle: { color, opacity: opts.alpha ?? 1 },
    } as SeriesOption);
  }
  if (refData.length) {
    series.push({
      type: "scatter",
      name: `${opts.label ?? distLabel(dist)} (FFZ-350)`,
      xAxisIndex: gridIndex,
      yAxisIndex: gridIndex,
      data: refData,
      symbol: "diamond",
      symbolSize: 14,
      itemStyle: {
        color,
        borderColor: CITY_COLORS[REF_CITY] ?? "#e08030",
        borderWidth: 1.8,
        opacity: opts.alpha ?? 1,
      },
      z: 4,
    } as SeriesOption);
  }
  return series;
}

export function valueAxes(
  gridIndex: number,
  theme: GenTheme,
  xName: string,
  yName: string
): { x: object; y: object } {
  return {
    x: { type: "value", gridIndex, scale: true, ...axisStyle(theme, { name: xName }), nameGap: 26 },
    y: { type: "value", gridIndex, scale: true, ...axisStyle(theme, { name: yName }), nameGap: 42 },
  };
}

// ── NPZ stats bar (ports gen_npz_stats_bar) ──────────────────────────────────

export function buildNpzStatsBar(
  npz: NpzStatRow[],
  ext: ExtendedRow[],
  theme: GenTheme
): ChartSpec {
  const extKey = new Map(ext.map((e) => [`${e.city}|${e.N}|${e.dist}`, e]));
  const sub30 = npz.filter((r) => r.horizon === 30);
  const cities = [...new Set(sub30.map((r) => r.city))].sort();
  const dists = [...new Set(sub30.map((r) => r.dist))].sort();
  const metrics: [string, string, (r: NpzStatRow) => number | undefined][] = [
    ["mean_kg", "Mean Waste (kg/bin/day)", (r) => r.mean_kg],
    ["median", "Median Waste (kg/bin/day)", (r) => extKey.get(`${r.city}|${r.N}|${r.dist}`)?.median],
    ["std_kg", "Std Waste (kg/bin/day)", (r) => r.std_kg],
    ["max_kg", "Max Waste (kg)", (r) => r.max_kg],
  ];
  const { boxes, grids } = subGrid(2, 2);
  const titles: object[] = [supTitle("NPZ Dataset Statistics by City and Distribution", theme)];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const series: SeriesOption[] = [];

  metrics.forEach(([metric, ylabel, getter], pi) => {
    titles.push(subTitle(metric, boxes[pi], theme));
    xAxes.push({
      type: "category",
      gridIndex: pi,
      data: cities.map((c) => CITY_LABELS[c] ?? c),
      axisLabel: { color: theme.axisLabelColor, fontSize: 11 },
    });
    yAxes.push({
      type: "value",
      gridIndex: pi,
      ...axisStyle(theme, { name: ylabel }),
      nameGap: 40,
    });
    for (const dist of dists) {
      series.push({
        type: "bar",
        name: distLabel(dist),
        xAxisIndex: pi,
        yAxisIndex: pi,
        data: cities.map((city) => {
          const vals = sub30
            .filter((r) => r.dist === dist && r.city === city)
            .map(getter)
            .filter((v): v is number => v !== undefined && Number.isFinite(v));
          return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
        }),
        itemStyle: { color: distColor(dist), opacity: 0.85 },
      } as SeriesOption);
    }
  });

  return {
    width: 1500,
    height: 1100,
    option: {
      backgroundColor: theme.bg,
      title: titles as never,
      grid: grids as never,
      xAxis: xAxes as never,
      yAxis: yAxes as never,
      legend: {
        top: 30,
        left: "center",
        data: dists.map(distLabel),
        textStyle: { color: theme.fg, fontSize: 11 },
      },
      series,
    },
  };
}

// ── Size scaling / horizon comparison / extended stats line grids ────────────

export function lineGridChart(
  title: string,
  panels: {
    title: string;
    yName: string;
    series: (gridIndex: number) => SeriesOption[];
  }[],
  layout: [number, number],
  size: [number, number],
  theme: GenTheme
): ChartSpec {
  const [nrows, ncols] = layout;
  const { boxes, grids } = subGrid(nrows, ncols);
  const titles: object[] = [supTitle(title, theme)];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const series: SeriesOption[] = [];
  panels.forEach((p, pi) => {
    titles.push(subTitle(p.title, boxes[pi], theme));
    const axes = valueAxes(pi, theme, "Network size N", p.yName);
    xAxes.push(axes.x);
    yAxes.push(axes.y);
    series.push(...p.series(pi));
  });
  return {
    width: size[0],
    height: size[1],
    option: {
      backgroundColor: theme.bg,
      title: titles as never,
      grid: grids as never,
      xAxis: xAxes as never,
      yAxis: yAxes as never,
      legend: {
        type: "scroll",
        top: 28,
        left: "center",
        textStyle: { color: theme.fg, fontSize: 10 },
      },
      tooltip: { trigger: "item" },
      series,
    },
  };
}

export function buildNpzSizeScaling(npz: NpzStatRow[], theme: GenTheme): ChartSpec {
  const sub30 = npz.filter((r) => r.horizon === 30);
  const dists = [...new Set(sub30.map((r) => r.dist))].sort();
  const metrics: [keyof NpzStatRow, string][] = [
    ["mean_kg", "Mean Waste (kg)"],
    ["std_kg", "Std Waste (kg)"],
    ["skewness", "Skewness"],
  ];
  return lineGridChart(
    "NPZ Statistics vs Network Size (lines: Rio Maior · diamonds: FFZ-350)",
    metrics.map(([metric, yName]) => ({
      title: `${String(metric)} vs N`,
      yName,
      series: (gi) =>
        dists.flatMap((dist) =>
          linePlusRef(
            gi,
            sub30
              .filter((r) => r.city === LINE_CITY && r.dist === dist)
              .sort((a, b) => a.N - b.N)
              .map((r) => [r.N, r[metric] as number] as [number, number]),
            sub30
              .filter((r) => r.city === REF_CITY && r.dist === dist)
              .map((r) => [r.N, r[metric] as number] as [number, number]),
            dist
          )
        ),
    })),
    [1, 3],
    [1700, 620],
    theme
  );
}

export function buildNpzHorizonComparison(npz: NpzStatRow[], theme: GenTheme): ChartSpec | null {
  const horizons = [...new Set(npz.map((r) => r.horizon))].sort((a, b) => a - b);
  if (horizons.length < 2) return null;
  const dists = [...new Set(npz.map((r) => r.dist))].sort();
  const metrics: [keyof NpzStatRow, string][] = [
    ["mean_kg", "Mean Waste (kg)"],
    ["std_kg", "Std Waste (kg)"],
    ["skewness", "Skewness"],
  ];
  return lineGridChart(
    `${horizons.map((h) => `${h}-day`).join(" vs ")} Horizon Comparison (lines: Rio Maior · diamonds: FFZ-350)`,
    metrics.map(([metric, yName]) => ({
      title: String(metric),
      yName,
      series: (gi) =>
        dists.flatMap((dist) =>
          horizons.flatMap((horizon, hi) => {
            const subH = npz.filter((r) => r.dist === dist && r.horizon === horizon);
            return linePlusRef(
              gi,
              subH
                .filter((r) => r.city === LINE_CITY)
                .sort((a, b) => a.N - b.N)
                .map((r) => [r.N, r[metric] as number] as [number, number]),
              subH
                .filter((r) => r.city === REF_CITY)
                .map((r) => [r.N, r[metric] as number] as [number, number]),
              dist,
              { label: `${distLabel(dist)} ${horizon}d`, dashed: hi > 0, alpha: hi === 0 ? 1 : 0.7 }
            );
          })
        ),
    })),
    [1, 3],
    [1700, 620],
    theme
  );
}

export function buildNpzExtendedStats(ext: ExtendedRow[], theme: GenTheme): ChartSpec {
  const dists = [...new Set(ext.map((r) => r.dist))].sort();
  const metrics: [keyof ExtendedRow, string][] = [
    ["median", "Median (kg)"],
    ["variance", "Variance (kg²)"],
    ["iqr", "IQR (kg)"],
    ["min", "Minimum (kg)"],
    ["upper_fence", "Upper Outlier Fence (kg)"],
    ["mode", "Mode (kg, binned)"],
  ];
  return lineGridChart(
    "Extended NPZ Statistics vs Network Size (lines: Rio Maior · diamonds: FFZ-350)",
    metrics.map(([metric, yName]) => ({
      title: String(metric),
      yName,
      series: (gi) =>
        dists.flatMap((dist) =>
          linePlusRef(
            gi,
            ext
              .filter((r) => r.city === LINE_CITY && r.dist === dist)
              .sort((a, b) => a.N - b.N)
              .map((r) => [r.N, r[metric] as number] as [number, number]),
            ext
              .filter((r) => r.city === REF_CITY && r.dist === dist)
              .map((r) => [r.N, r[metric] as number] as [number, number]),
            dist
          )
        ),
    })),
    [2, 3],
    [1900, 1050],
    theme
  );
}

// ── City comparison (ports gen_npz_city_comparison) ──────────────────────────

export function buildNpzCityComparison(
  npz: NpzStatRow[],
  ext: ExtendedRow[],
  theme: GenTheme
): ChartSpec {
  const extKey = new Map(ext.map((e) => [`${e.city}|${e.N}|${e.dist}`, e]));
  const sub30 = npz.filter((r) => r.horizon === 30);
  const cities = [...new Set(sub30.map((r) => r.city))].sort();
  const metrics: [string, string, (r: NpzStatRow) => number | undefined][] = [
    ["mean_kg", "Mean Waste (kg)", (r) => r.mean_kg],
    ["median", "Median Waste (kg)", (r) => extKey.get(`${r.city}|${r.N}|${r.dist}`)?.median],
    ["std_kg", "Std (kg)", (r) => r.std_kg],
    ["skewness", "Skewness", (r) => r.skewness],
  ];
  const { boxes, grids } = subGrid(2, 2, { legendPad: 5 });
  const titles: object[] = [supTitle("City Comparison Overview — NPZ Datasets", theme)];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const series: SeriesOption[] = [];

  metrics.forEach(([metric, ylabel, getter], pi) => {
    titles.push(subTitle(metric, boxes[pi], theme));
    const axes = valueAxes(pi, theme, "Network size N", ylabel);
    xAxes.push(axes.x);
    yAxes.push(axes.y);
    for (const city of cities) {
      for (const r of sub30.filter((x) => x.city === city)) {
        const v = getter(r);
        if (v === undefined || !Number.isFinite(v)) continue;
        series.push({
          type: "scatter",
          xAxisIndex: pi,
          yAxisIndex: pi,
          data: [[r.N, v]],
          symbol: city === LINE_CITY ? "circle" : "diamond",
          symbolSize: 16,
          silent: true,
          itemStyle: {
            color: CITY_COLORS[city] ?? "#a0a0a0",
            opacity: 0.85,
            borderColor: r.dist === "gamma3" ? "#ffffff" : "transparent",
            borderWidth: 1.5,
          },
        } as SeriesOption);
      }
    }
  });

  const legendData = [
    ...cities.map((c) => ({
      name: CITY_LABELS[c] ?? c,
      icon: "roundRect",
      itemStyle: { color: CITY_COLORS[c] ?? "gray" },
    })),
    { name: distLabel("emp"), icon: "circle", itemStyle: { color: "gray" } },
    {
      name: distLabel("gamma3"),
      icon: "circle",
      itemStyle: { color: "gray", borderColor: "#ffffff", borderWidth: 1.5 },
    },
  ];

  return {
    width: 1500,
    height: 1150,
    option: {
      backgroundColor: theme.bg,
      title: titles as never,
      grid: grids as never,
      xAxis: xAxes as never,
      yAxis: yAxes as never,
      legend: { bottom: 6, data: legendData as never, textStyle: { color: theme.fg, fontSize: 10 } },
      series,
    },
  };
}
