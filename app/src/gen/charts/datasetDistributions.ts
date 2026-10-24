/**
 * Raw waste distribution shape chart builders: Violin, Box plot, Hist+KDE (§H.2).
 */
import type { SeriesOption } from "echarts";
import { DATASET_CFG, type GenTheme } from "../config";
import { axisStyle, type ChartSpec } from "./common";
import {
  gaussianKde,
  percentile,
  histogram,
  type RawWaste,
} from "../data/dataset";

const DIST_LABELS = DATASET_CFG.dist_labels;
const DIST_COLORS = DATASET_CFG.dist_colors;

const REF_CITY = "figueiradafoz";

const distColor = (d: string) => DIST_COLORS[d] ?? "#a0a0a0";
const distLabel = (d: string) => DIST_LABELS[d] ?? d;

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
