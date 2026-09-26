/**
 * Dataset analysis chart library (§H.2) — native ECharts ports of every
 * figure generator in `logic/gen/gen_dataset_analysis.py`.
 */
import type { SeriesOption } from "echarts";
import { DATASET_CFG, type GenTheme } from "../config";
import { axisStyle, type ChartSpec } from "./common";
import {
  gaussianKde,
  percentile,
  histogram,
  type NpzStatRow,
  type TdStatRow,
  type ExtendedRow,
  type RawWaste,
} from "../data/dataset";

const CITY_LABELS = DATASET_CFG.city_labels;
const DIST_LABELS = DATASET_CFG.dist_labels;
const CITY_COLORS = DATASET_CFG.city_colors;
const DIST_COLORS = DATASET_CFG.dist_colors;

export const REF_CITY = "figueiradafoz"; // single-N reference city (diamonds)
export const LINE_CITY = "riomaior"; // multi-N evolution-line city

const distColor = (d: string) => DIST_COLORS[d] ?? "#a0a0a0";
const distLabel = (d: string) => DIST_LABELS[d] ?? d;

// ── grid helper (nrows × ncols with a suptitle band) ─────────────────────────

interface SubGrid {
  boxes: { left: number; top: number; width: number; height: number }[];
  grids: object[];
}

function subGrid(nrows: number, ncols: number, opts: { legendPad?: number } = {}): SubGrid {
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

function supTitle(text: string, theme: GenTheme) {
  return {
    text,
    left: "center",
    top: 6,
    textStyle: { color: theme.fg, fontSize: 15, fontWeight: "bold" as const },
  };
}

function subTitle(text: string, box: { left: number; width: number; top: number }, theme: GenTheme) {
  return {
    text,
    left: `${box.left + box.width / 2}%`,
    top: `${box.top - 4}%`,
    textAlign: "center" as const,
    textStyle: { color: theme.fg, fontSize: 12, fontWeight: "bold" as const },
  };
}

/** Line + reference-diamond series pair (ports _plot_line_plus_ref). */
function linePlusRef(
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

function valueAxes(
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

function lineGridChart(
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

// ── TD vs NPZ alignment (ports gen_npz_td_alignment) ─────────────────────────

export function buildNpzTdAlignment(
  npz: NpzStatRow[],
  td: TdStatRow[],
  theme: GenTheme
): ChartSpec {
  const sub30 = npz.filter((r) => r.horizon === 30);
  const dists = [...new Set(sub30.map((r) => r.dist))].sort();
  const series: SeriesOption[] = [];
  for (const dist of dists) {
    series.push(
      ...linePlusRef(
        0,
        sub30
          .filter((r) => r.city === LINE_CITY && r.dist === dist)
          .sort((a, b) => a.N - b.N)
          .map((r) => [r.N, r.mean_kg] as [number, number]),
        sub30
          .filter((r) => r.city === REF_CITY && r.dist === dist)
          .map((r) => [r.N, r.mean_kg] as [number, number]),
        dist,
        { label: `NPZ ${distLabel(dist)} (RM)` }
      )
    );
    const tdSub = td.filter((r) => r.dist === dist).sort((a, b) => a.N - b.N);
    if (tdSub.length) {
      series.push({
        type: "line",
        name: `TD ${distLabel(dist)} (×100)`,
        data: tdSub.map((r) => [r.N, r.waste_mean * 100]),
        symbol: "rect",
        symbolSize: 8,
        lineStyle: { color: distColor(dist), width: 1.5, type: "dashed", opacity: 0.7 },
        itemStyle: { color: distColor(dist), opacity: 0.7 },
      } as SeriesOption);
    }
  }
  return {
    width: 1250,
    height: 720,
    option: {
      backgroundColor: theme.bg,
      title: supTitle("Training (TD) vs Simulator (NPZ) Mean Waste Alignment", theme) as never,
      grid: { left: 90, right: 40, top: 90, bottom: 70 },
      xAxis: { type: "value", scale: true, ...axisStyle(theme, { name: "Network size N" }) },
      yAxis: {
        type: "value",
        scale: true,
        ...axisStyle(theme, { name: "Mean waste (kg/bin/day)" }),
        nameGap: 42,
      },
      legend: { top: 34, left: "center", textStyle: { color: theme.fg, fontSize: 10 } },
      series,
    },
  };
}

// ── TD stats (ports gen_td_stats — two figures) ──────────────────────────────

export function buildTdStatsComparison(td: TdStatRow[], theme: GenTheme): ChartSpec {
  const dists = [...new Set(td.map((r) => r.dist))].sort();
  const metrics: [keyof TdStatRow, string][] = [
    ["waste_mean", "Mean waste fraction"],
    ["waste_std", "Std waste fraction"],
    ["waste_skew", "Skewness"],
  ];
  return lineGridChart(
    "Training Data (TD) Statistics Comparison",
    metrics.map(([metric, yName]) => ({
      title: String(metric),
      yName,
      series: (gi) =>
        dists.map(
          (dist) =>
            ({
              type: "line",
              name: distLabel(dist),
              xAxisIndex: gi,
              yAxisIndex: gi,
              data: td
                .filter((r) => r.dist === dist)
                .sort((a, b) => a.N - b.N)
                .map((r) => [r.N, r[metric] as number]),
              symbol: "circle",
              symbolSize: 9,
              lineStyle: { color: distColor(dist), width: 2 },
              itemStyle: { color: distColor(dist) },
            }) as SeriesOption
        ),
    })),
    [1, 3],
    [1700, 620],
    theme
  );
}

export function buildTdWasteDistributions(td: TdStatRow[], theme: GenTheme): ChartSpec {
  const dists = [...new Set(td.map((r) => r.dist))].sort();
  const ns = [...new Set(td.map((r) => r.N))].sort((a, b) => a - b);
  const metrics: (keyof TdStatRow)[] = ["waste_mean", "waste_std"];
  const { boxes, grids } = subGrid(1, 2);
  const titles: object[] = [supTitle("Training Data Waste Distributions", theme)];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const series: SeriesOption[] = [];
  metrics.forEach((metric, pi) => {
    titles.push(subTitle(String(metric), boxes[pi], theme));
    xAxes.push({
      type: "category",
      gridIndex: pi,
      data: ns.map((n) => `N=${n}`),
      axisLabel: { color: theme.axisLabelColor, fontSize: 11 },
    });
    yAxes.push({
      type: "value",
      gridIndex: pi,
      ...axisStyle(theme, { name: String(metric) }),
      nameGap: 46,
    });
    for (const dist of dists) {
      const byN = new Map(td.filter((r) => r.dist === dist).map((r) => [r.N, r[metric] as number]));
      series.push({
        type: "bar",
        name: distLabel(dist),
        xAxisIndex: pi,
        yAxisIndex: pi,
        data: ns.map((n) => byN.get(n) ?? null),
        itemStyle: { color: distColor(dist), opacity: 0.85 },
      } as SeriesOption);
    }
  });
  return {
    width: 1350,
    height: 620,
    option: {
      backgroundColor: theme.bg,
      title: titles as never,
      grid: grids as never,
      xAxis: xAxes as never,
      yAxis: yAxes as never,
      legend: { top: 28, left: "center", textStyle: { color: theme.fg, fontSize: 10 } },
      series,
    },
  };
}

// ── Raw distribution shapes (violin / box / hist+KDE) ────────────────────────

function rawGroups(raw: RawWaste): { dists: string[]; groups: Map<string, [string, number[]][]> } {
  const dists = [...new Set([...raw.values()].map((v) => v.dist))].sort();
  const groups = new Map<string, [string, number[]][]>(dists.map((d) => [d, []]));
  for (const { city, N, dist, values } of [...raw.values()].sort((a, b) => a.N - b.N)) {
    const short = city === REF_CITY ? "FFZ" : "RM";
    groups.get(dist)!.push([`${short}-${N}`, values]);
  }
  return { dists, groups };
}

interface RenderApi {
  coord: (v: [number, number]) => number[];
}

export function buildNpzViolin(raw: RawWaste, theme: GenTheme): ChartSpec {
  const { dists, groups } = rawGroups(raw);
  const { boxes, grids } = subGrid(1, dists.length);
  const titles: object[] = [
    supTitle("Raw Waste Distribution — Violin Plots (30-day horizon)", theme),
  ];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const series: SeriesOption[] = [];

  dists.forEach((dist, pi) => {
    titles.push(subTitle(distLabel(dist), boxes[pi], theme));
    const entries = groups.get(dist)!;
    xAxes.push({
      type: "value",
      gridIndex: pi,
      min: -0.7,
      max: entries.length - 0.3,
      interval: 1,
      axisLabel: {
        color: theme.axisLabelColor,
        fontSize: 9.5,
        formatter: (v: number) => (Number.isInteger(v) && entries[v] ? entries[v][0] : ""),
      },
      splitLine: { show: false },
      axisLine: { lineStyle: { color: theme.axisLabelColor } },
    });
    yAxes.push({
      type: "value",
      gridIndex: pi,
      scale: true,
      ...axisStyle(theme, { name: "Waste (kg/bin/day)" }),
      nameGap: 44,
    });
    entries.forEach(([, values], gi) => {
      const { x: kx, y: ky } = gaussianKde(values, 80);
      const maxD = Math.max(...ky, 1e-9);
      const half = 0.42;
      const poly: [number, number][] = [
        ...kx.map((v, i) => [gi + (ky[i] / maxD) * half, v] as [number, number]),
        ...[...kx.keys()].reverse().map((i) => [gi - (ky[i] / maxD) * half, kx[i]] as [number, number]),
      ];
      const sorted = [...values].sort((a, b) => a - b);
      const q1 = percentile(sorted, 25);
      const med = percentile(sorted, 50);
      const q3 = percentile(sorted, 75);
      const lo = sorted[0];
      const hi = sorted[sorted.length - 1];
      series.push({
        type: "custom",
        xAxisIndex: pi,
        yAxisIndex: pi,
        silent: true,
        data: [0],
        renderItem: (_p: unknown, apiArg: unknown) => {
          const api = apiArg as RenderApi;
          const line = (p1: [number, number], p2: [number, number], color: string, w: number) => {
            const a = api.coord(p1);
            const b = api.coord(p2);
            return {
              type: "line" as const,
              shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] },
              style: { stroke: color, lineWidth: w },
            };
          };
          return {
            type: "group",
            children: [
              {
                type: "polygon",
                shape: { points: poly.map(([px, py]) => api.coord([px, py])) },
                style: { fill: distColor(dist), opacity: 0.7, stroke: theme.guideLine, lineWidth: 0.6 },
              },
              line([gi, lo], [gi, hi], theme.guideLine, 1),
              line([gi - 0.18, med], [gi + 0.18, med], theme.accentLine, 1.6),
              line([gi - 0.14, q1], [gi + 0.14, q1], theme.accentLine, 1.2),
              line([gi - 0.14, q3], [gi + 0.14, q3], theme.accentLine, 1.2),
            ],
          };
        },
        z: 2,
      } as unknown as SeriesOption);
    });
  });

  return {
    width: Math.max(900 * dists.length, 900),
    height: 720,
    option: {
      backgroundColor: theme.bg,
      title: titles as never,
      grid: grids as never,
      xAxis: xAxes as never,
      yAxis: yAxes as never,
      series,
    },
  };
}

export function buildNpzBox(raw: RawWaste, theme: GenTheme): ChartSpec {
  const { dists, groups } = rawGroups(raw);
  const { boxes, grids } = subGrid(1, dists.length);
  const titles: object[] = [supTitle("Raw Waste Distribution — Box Plots (30-day horizon)", theme)];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const series: SeriesOption[] = [];

  dists.forEach((dist, pi) => {
    titles.push(subTitle(distLabel(dist), boxes[pi], theme));
    const entries = groups.get(dist)!;
    xAxes.push({
      type: "category",
      gridIndex: pi,
      data: entries.map(([lbl]) => lbl),
      axisLabel: { color: theme.axisLabelColor, fontSize: 9.5 },
    });
    yAxes.push({
      type: "value",
      gridIndex: pi,
      scale: true,
      ...axisStyle(theme, { name: "Waste (kg/bin/day)" }),
      nameGap: 44,
    });
    const boxData: number[][] = [];
    const outliers: [number, number][] = [];
    entries.forEach(([, values], gi) => {
      const sorted = [...values].sort((a, b) => a - b);
      const q1 = percentile(sorted, 25);
      const med = percentile(sorted, 50);
      const q3 = percentile(sorted, 75);
      const iqr = q3 - q1;
      const loFence = q1 - 1.5 * iqr;
      const hiFence = q3 + 1.5 * iqr;
      const inliers = sorted.filter((v) => v >= loFence && v <= hiFence);
      boxData.push([
        inliers[0] ?? sorted[0],
        q1,
        med,
        q3,
        inliers[inliers.length - 1] ?? sorted[sorted.length - 1],
      ]);
      // cap outlier markers to a manageable sample
      const outs = sorted.filter((v) => v < loFence || v > hiFence);
      const step = Math.max(1, Math.floor(outs.length / 400));
      for (let i = 0; i < outs.length; i += step) outliers.push([gi, outs[i]]);
    });
    series.push(
      {
        type: "boxplot",
        xAxisIndex: pi,
        yAxisIndex: pi,
        data: boxData,
        itemStyle: {
          color: `${distColor(dist)}b3`,
          borderColor: theme.guideLine,
        },
      } as SeriesOption,
      {
        type: "scatter",
        xAxisIndex: pi,
        yAxisIndex: pi,
        data: outliers,
        symbolSize: 3,
        silent: true,
        itemStyle: { color: theme.muted, opacity: 0.35 },
      } as SeriesOption
    );
  });

  return {
    width: Math.max(900 * dists.length, 900),
    height: 720,
    option: {
      backgroundColor: theme.bg,
      title: titles as never,
      grid: grids as never,
      xAxis: xAxes as never,
      yAxis: yAxes as never,
      series,
    },
  };
}

const KDE_PALETTE = ["#4e88d9", "#e09020", "#20b2aa", "#a060e0", "#e05c5c", "#5cb85c"];

export function buildNpzHistKde(raw: RawWaste, theme: GenTheme): ChartSpec {
  const { dists, groups } = rawGroups(raw);
  const { boxes, grids } = subGrid(1, dists.length, { legendPad: 4 });
  const titles: object[] = [supTitle("Raw Waste Histograms with KDE (30-day horizon)", theme)];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const series: SeriesOption[] = [];

  dists.forEach((dist, pi) => {
    titles.push(subTitle(distLabel(dist), boxes[pi], theme));
    const entries = groups.get(dist)!;
    const pooled = entries.flatMap(([, v]) => v);
    const { counts, edges } = histogram(pooled, 60);
    const total = pooled.length;
    const binW = edges[1] - edges[0] || 1;
    xAxes.push({
      type: "value",
      gridIndex: pi,
      scale: true,
      ...axisStyle(theme, { name: "Waste (kg/bin/day)" }),
    });
    yAxes.push({
      type: "value",
      gridIndex: pi,
      ...axisStyle(theme, { name: "Density" }),
      nameGap: 48,
    });
    // density histogram as bars on value axis
    series.push({
      type: "custom",
      xAxisIndex: pi,
      yAxisIndex: pi,
      name: "All sizes (pooled)",
      silent: true,
      data: counts.map((c, i) => [edges[i], edges[i + 1], c / (total * binW)]),
      renderItem: (_p: unknown, apiArg: unknown) => {
        const api = apiArg as RenderApi & { value: (i: number) => number };
        const x0 = api.coord([api.value(0), 0]);
        const x1 = api.coord([api.value(1), api.value(2)]);
        return {
          type: "rect",
          shape: { x: x0[0], y: x1[1], width: x1[0] - x0[0], height: x0[1] - x1[1] },
          style: { fill: distColor(dist), opacity: 0.35 },
        };
      },
      z: 1,
    } as unknown as SeriesOption);
    entries.forEach(([lbl, values], i) => {
      const kde = gaussianKde(values, 200, Math.min(...pooled), Math.max(...pooled));
      series.push({
        type: "line",
        name: `KDE ${lbl}`,
        xAxisIndex: pi,
        yAxisIndex: pi,
        data: kde.x.map((x, j) => [x, kde.y[j]]),
        showSymbol: false,
        lineStyle: { color: KDE_PALETTE[i % KDE_PALETTE.length], width: 1.8 },
        itemStyle: { color: KDE_PALETTE[i % KDE_PALETTE.length] },
        z: 2,
      } as SeriesOption);
    });
  });

  return {
    width: Math.max(900 * dists.length, 900),
    height: 720,
    option: {
      backgroundColor: theme.bg,
      title: titles as never,
      grid: grids as never,
      xAxis: xAxes as never,
      yAxis: yAxes as never,
      legend: { bottom: 4, left: "center", textStyle: { color: theme.fg, fontSize: 9.5 } },
      series,
    },
  };
}
