/**
 * KPI bar (single/combined) and km violin chart builders (§H.2).
 */
import type { SeriesOption } from "echarts";
import { META, KGKM_LABEL, type GenTheme } from "../config";
import {
  panelGrid,
  panelTitle,
  axisStyle,
  errorBarSeries,
  legendPlaceholderSeries,
  symlogVals,
  symlogAxisLabelFormatter,
  HATCH_DECALS,
  type ChartSpec,
  type PanelBox,
} from "./common";
import { gaussianKde } from "../data/dataset";
import {
  scenSub,
  regionLabel,
  metricValues,
  mean,
  type AnalysisCtx,
  type Scenario,
  type SimRow,
  type SimMetric,
} from "../data/simulation";
import { strategyColor, type Scale } from "./simulationScatter";

interface KpiPanelInput {
  box: PanelBox;
  gridIndex: number;
  scenarios: Scenario[];
  useLog: boolean;
}

function kpiPanelSeries(
  dfm: SimRow[],
  metric: SimMetric,
  ctx: AnalysisCtx,
  theme: GenTheme,
  panel: KpiPanelInput
): { series: SeriesOption[]; categories: string[] } {
  const { strategies, improvers } = ctx;
  const groups: [Scenario, string][] = panel.scenarios.flatMap((s) =>
    strategies.map((strat) => [s, strat] as [Scenario, string])
  );
  const categories = groups.map(([s, strat]) => `${regionLabel(s.city, s.N)}\n${strat}`);
  const series: SeriesOption[] = [];
  improvers.forEach((imp, ii) => {
    const means: (number | null)[] = [];
    const err: { x: number; low: number; high: number }[] = [];
    groups.forEach(([s, strat], gi) => {
      const vals = metricValues(
        scenSub(dfm.filter((r) => r.improver === imp && r.strategy === strat), s),
        metric
      );
      if (!vals.length) {
        means.push(null);
        return;
      }
      const m = mean(vals);
      means.push(panel.useLog ? symlogVals([m], true)[0] : m);
      const minV = Math.min(...vals);
      const maxV = Math.max(...vals);
      err.push({
        x: gi,
        low: panel.useLog ? symlogVals([minV], true)[0] : minV,
        high: panel.useLog ? symlogVals([maxV], true)[0] : maxV,
      });
    });
    series.push({
      type: "bar",
      xAxisIndex: panel.gridIndex,
      yAxisIndex: panel.gridIndex,
      barGap: "8%",
      data: means.map((m, gi) => ({
        value: m,
        itemStyle: {
          color: strategyColor(groups[gi][1]),
          opacity: ii === 0 ? 0.9 : 0.6,
          decal: HATCH_DECALS[ii % HATCH_DECALS.length] as never,
          borderColor: theme.fg,
          borderWidth: 0.4,
        },
      })),
      z: 2,
    } as SeriesOption);
    // whiskers must be positioned on the grouped-bar offsets
    const nImp = improvers.length;
    const w = 0.8 / nImp;
    series.push(
      errorBarSeries(
        err.map((e) => ({ x: e.x + (ii - (nImp - 1) / 2) * w, low: e.low, high: e.high })),
        theme.fg,
        { xAxisIndex: panel.gridIndex, yAxisIndex: panel.gridIndex }
      )
    );
  });
  return { series, categories };
}

function kpiAxes(
  categories: string[],
  ylabel: string,
  theme: GenTheme,
  gridIndex: number,
  useLog: boolean
): { x: object[]; y: object } {
  return {
    x: [
      {
        type: "category",
        gridIndex,
        data: categories,
        axisLabel: { color: theme.axisLabelColor, fontSize: 9.5, interval: 0, fontWeight: "bold" },
        axisTick: { alignWithLabel: true },
        axisLine: { lineStyle: { color: theme.axisLabelColor } },
      },
      {
        type: "value",
        gridIndex,
        min: -0.5,
        max: categories.length - 0.5,
        show: false,
      },
    ],
    y: {
      type: "value",
      gridIndex,
      ...axisStyle(theme, { name: ylabel }),
      nameGap: 42,
      axisLabel: {
        color: theme.axisLabelColor,
        fontSize: 11,
        formatter: symlogAxisLabelFormatter(useLog),
      },
    },
  };
}

function retargetErrorBars(series: SeriesOption[], numericAxisIndexOf: (gridIndex: number) => number): void {
  for (const s of series) {
    if ((s as { type?: string }).type === "custom") {
      const cur = (s as { xAxisIndex?: number }).xAxisIndex ?? 0;
      (s as { xAxisIndex?: number }).xAxisIndex = numericAxisIndexOf(cur);
    }
  }
}

export function buildKpiBar(
  dfm: SimRow[],
  metric: SimMetric,
  ylabel: string,
  ctx: AnalysisCtx,
  theme: GenTheme,
  yscale: Scale = "linear"
): ChartSpec {
  const { dists } = ctx;
  const useLog = yscale !== "linear";
  const grid = panelGrid(dists.length, { legendRows: 1, panelW: 720, panelH: 430 });
  const series: SeriesOption[] = [];
  const titles: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];

  dists.forEach((dist, pi) => {
    const box = grid.boxes[pi];
    titles.push(panelTitle(dist, box, theme));
    const scen = ctx.scenarios.filter((s) => s.dist === dist);
    const { series: ps, categories } = kpiPanelSeries(dfm, metric, ctx, theme, {
      box,
      gridIndex: pi,
      scenarios: scen,
      useLog,
    });
    retargetErrorBars(ps, (gi) => gi * 2 + 1);
    series.push(...ps);
    const { x, y } = kpiAxes(categories, ylabel, theme, pi, useLog);
    xAxes.push(...x);
    yAxes.push(y);
  });

  const legendData = ctx.improvers.map((imp) => ({
    name: imp,
    icon: "roundRect",
    itemStyle: { color: META.improver_colors[imp] ?? "#808080" },
  }));

  return {
    width: grid.width,
    height: grid.height,
    option: {
      backgroundColor: theme.bg,
      title: titles,
      grid: grid.boxes.map((b) => ({
        left: `${b.left}%`,
        top: `${b.top}%`,
        width: `${b.width}%`,
        height: `${b.height}%`,
      })),
      xAxis: xAxes as never,
      yAxis: yAxes as never,
      legend: { bottom: 4, data: legendData as never, textStyle: { color: theme.fg, fontSize: 12 } },
      series: [...series, ...legendPlaceholderSeries(legendData)],
    },
  };
}

export function buildKpiCombined(dfm: SimRow[], ctx: AnalysisCtx, theme: GenTheme): ChartSpec {
  const { dists } = ctx;
  const metrics: [SimMetric, string, boolean][] = [
    ["overflows", "Mean overflows", true],
    ["kgkm", `Efficiency (${KGKM_LABEL})`, false],
  ];
  const grid = panelGrid(dists.length * 2, { legendRows: 1, panelW: 760, panelH: 360 });
  const series: SeriesOption[] = [];
  const titles: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];

  metrics.forEach(([metric, ylabel, useLog], rowI) => {
    dists.forEach((dist, colI) => {
      const pi = rowI * dists.length + colI;
      const box = grid.boxes[pi];
      titles.push(panelTitle(`${dist} — ${ylabel}`, box, theme, 12));
      const scen = ctx.scenarios.filter((s) => s.dist === dist);
      const { series: ps, categories } = kpiPanelSeries(dfm, metric, ctx, theme, {
        box,
        gridIndex: pi,
        scenarios: scen,
        useLog,
      });
      retargetErrorBars(ps, (gi) => gi * 2 + 1);
      series.push(...ps);
      const { x, y } = kpiAxes(categories, ylabel, theme, pi, useLog);
      xAxes.push(...x);
      yAxes.push(y);
    });
  });

  const legendData = ctx.improvers.map((imp) => ({
    name: imp,
    icon: "roundRect",
    itemStyle: { color: META.improver_colors[imp] ?? "#808080" },
  }));

  return {
    width: grid.width,
    height: grid.height,
    option: {
      backgroundColor: theme.bg,
      title: titles,
      grid: grid.boxes.map((b) => ({
        left: `${b.left}%`,
        top: `${b.top}%`,
        width: `${b.width}%`,
        height: `${b.height}%`,
      })),
      xAxis: xAxes as never,
      yAxis: yAxes as never,
      legend: { bottom: 4, data: legendData as never, textStyle: { color: theme.fg, fontSize: 14 } },
      series: [...series, ...legendPlaceholderSeries(legendData)],
    },
  };
}

// ── km violin (ports gen_km_violin) ──────────────────────────────────────────

interface RenderApi {
  coord: (v: [number, number]) => number[];
}

function violinSeries(
  groups: { label: string; values: number[]; color: string }[],
  gridIndex: number,
  theme: GenTheme
): SeriesOption[] {
  const line = (api: RenderApi, p1: [number, number], p2: [number, number], color: string, w: number) => {
    const a = api.coord(p1);
    const b = api.coord(p2);
    return {
      type: "line" as const,
      shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] },
      style: { stroke: color, lineWidth: w },
    };
  };
  const series: SeriesOption[] = [];
  groups.forEach((g, gi) => {
    let vals = g.values;
    if (!vals.length) vals = [0, 0];
    if (vals.length === 1) vals = [vals[0], vals[0]];
    const { x: kx, y: ky } = gaussianKde(vals, 60);
    const maxDensity = Math.max(...ky, 1e-9);
    const half = 0.42;
    const poly: [number, number][] = [
      ...kx.map((v, i) => [gi + (ky[i] / maxDensity) * half, v] as [number, number]),
      ...[...kx.keys()].reverse().map((i) => [gi - (ky[i] / maxDensity) * half, kx[i]] as [number, number]),
    ];
    const sorted = [...vals].sort((a, b) => a - b);
    const med = sorted[Math.floor(sorted.length / 2)];
    const lo = sorted[0];
    const hi = sorted[sorted.length - 1];
    series.push({
      type: "custom",
      xAxisIndex: gridIndex,
      yAxisIndex: gridIndex,
      silent: true,
      data: [0],
      renderItem: (_p: unknown, apiArg: unknown) => {
        const api = apiArg as RenderApi;
        return {
          type: "group",
          children: [
            {
              type: "polygon",
              shape: { points: poly.map(([px, py]) => api.coord([px, py])) },
              style: { fill: g.color, opacity: 0.7, stroke: theme.guideLine, lineWidth: 0.6 },
            },
            line(api, [gi, lo], [gi, hi], theme.guideLine, 1),
            line(api, [gi - 0.18, med], [gi + 0.18, med], theme.accentLine, 1.6),
            line(api, [gi - 0.12, lo], [gi + 0.12, lo], theme.guideLine, 1),
            line(api, [gi - 0.12, hi], [gi + 0.12, hi], theme.guideLine, 1),
          ],
        };
      },
      z: 2,
    } as unknown as SeriesOption);
  });
  return series;
}

export function buildKmViolin(df: SimRow[], ctx: AnalysisCtx, theme: GenTheme): ChartSpec {
  const { dists, strategies } = ctx;
  const grid = panelGrid(dists.length, { legendRows: 1, panelW: 720, panelH: 430 });
  const series: SeriesOption[] = [];
  const titles: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];

  dists.forEach((dist, pi) => {
    const box = grid.boxes[pi];
    titles.push(panelTitle(`Distance Distribution — ${dist}`, box, theme, 13));
    const scen = ctx.scenarios.filter((s) => s.dist === dist);
    const groups: { label: string; values: number[]; color: string }[] = [];
    for (const strat of strategies) {
      for (const s of scen) {
        groups.push({
          label: `${strat}\n${regionLabel(s.city, s.N)}`,
          values: metricValues(scenSub(df.filter((r) => r.strategy === strat), s), "km"),
          color: strategyColor(strat),
        });
      }
    }
    xAxes.push({
      type: "value",
      gridIndex: pi,
      min: -0.7,
      max: groups.length - 0.3,
      interval: 1,
      axisLabel: {
        color: theme.axisLabelColor,
        fontSize: 8.5,
        formatter: (v: number) =>
          Number.isInteger(v) && groups[v] ? groups[v].label.replace("\n", " ") : "",
      },
      splitLine: { show: false },
      axisLine: { lineStyle: { color: theme.axisLabelColor } },
    });
    yAxes.push({
      type: "value",
      gridIndex: pi,
      scale: true,
      ...axisStyle(theme, { name: `Total km (${ctx.nDays} days)` }),
      nameGap: 46,
    });
    series.push(...violinSeries(groups, pi, theme));
  });

  const legendData = strategies.map((s) => ({
    name: s,
    icon: "roundRect",
    itemStyle: { color: strategyColor(s) },
  }));

  return {
    width: grid.width,
    height: grid.height,
    option: {
      backgroundColor: theme.bg,
      title: titles,
      grid: grid.boxes.map((b) => ({
        left: `${b.left}%`,
        top: `${b.top}%`,
        width: `${b.width}%`,
        height: `${b.height}%`,
      })),
      xAxis: xAxes as never,
      yAxis: yAxes as never,
      legend: { bottom: 4, data: legendData as never, textStyle: { color: theme.fg, fontSize: 12 } },
      series: [...series, ...legendPlaceholderSeries(legendData)],
    },
  };
}
