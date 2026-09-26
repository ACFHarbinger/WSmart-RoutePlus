/**
 * Simulation analysis chart library (§H.2) — native ECharts ports of every
 * figure generator in `logic/gen/gen_simulation_analysis.py`.
 *
 * Each builder returns a ChartSpec (option + pixel size) used identically for
 * in-app preview and PNG export.
 */
import type { SeriesOption } from "echarts";
import { META, KGKM_LABEL, disp, dispAll, type GenTheme } from "../config";
import {
  panelGrid,
  panelTitle,
  axisStyle,
  errorBarSeries,
  legendPlaceholderSeries,
  stepFront,
  symlogVals,
  symlogAxisLabelFormatter,
  rdYlGnColors,
  MARKER_CYCLE,
  HATCH_DECALS,
  type ChartSpec,
  type PanelBox,
} from "./common";
import { gaussianKde } from "../data/dataset";
import {
  scenSub,
  scenarioLabel,
  regionLabel,
  variantColor,
  paretoIndices,
  metricValues,
  mean,
  groupSpans,
  type AnalysisCtx,
  type Scenario,
  type SimRow,
  type SimMetric,
  type HorizonData,
} from "../data/simulation";

type Scale = "linear" | "symlog";

const strategyColor = (s: string) => META.strategy_colors[s] ?? "#a0a0a0";

function regionMarkers(regions: [string, number][]): Map<string, string> {
  const m = new Map<string, string>();
  regions.forEach(([city, N], i) => m.set(`${city}|${N}`, MARKER_CYCLE[i % MARKER_CYCLE.length]));
  return m;
}

// ── Pareto scatter (ports gen_pareto_scatter) ────────────────────────────────

export function buildParetoScatter(
  df: SimRow[],
  ctx: AnalysisCtx,
  theme: GenTheme,
  xscale: Scale = "linear"
): ChartSpec {
  const { scenarios, dists, improvers, regions } = ctx;
  const variants = [...new Set(df.map((r) => r.variant))].sort();
  const vcolors = new Map(variants.map((v, i) => [v, variantColor(v, i)]));
  const markers = regionMarkers(regions);
  const frontColors = new Map(
    scenarios.map((s, i) => [
      `${s.city}|${s.N}|${s.dist}`,
      META.scenario_colors[i % META.scenario_colors.length],
    ])
  );
  const log = xscale !== "linear";
  const grid = panelGrid(dists.length, { legendRows: 3, panelW: 680, panelH: 500 });
  const series: SeriesOption[] = [];
  const titles: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];

  dists.forEach((dist, pi) => {
    const box = grid.boxes[pi];
    titles.push(panelTitle(dist, box, theme));
    xAxes.push({
      type: "value",
      gridIndex: pi,
      ...axisStyle(theme, { name: `Overflows (${ctx.nDays} days)` }),
      axisLabel: {
        color: theme.axisLabelColor,
        fontSize: 13,
        fontWeight: "bold",
        formatter: symlogAxisLabelFormatter(log),
      },
      scale: true,
    });
    yAxes.push({
      type: "value",
      gridIndex: pi,
      ...axisStyle(theme, { name: `Efficiency (${KGKM_LABEL})` }),
      nameGap: 38,
      axisLabel: { color: theme.axisLabelColor, fontSize: 13, fontWeight: "bold" },
      scale: true,
    });
    for (const s of scenarios.filter((sc) => sc.dist === dist)) {
      const sub = scenSub(df.filter((r) => r.dist === dist), s);
      if (!sub.length) continue;
      const xs = symlogVals(metricValues(sub, "overflows"), log);
      const ys = metricValues(sub, "kgkm");
      const front = paretoIndices(metricValues(sub, "overflows"), metricValues(sub, "kgkm"));
      const pts: object[] = [];
      sub.forEach((row, i) => {
        if (ctx.paretoPoints === "front" && !front.has(i)) return;
        const color = vcolors.get(row.variant)!;
        const filled = row.improver === improvers[0];
        pts.push({
          value: [xs[i], ys[i]],
          symbol: markers.get(`${s.city}|${s.N}`),
          symbolSize: front.has(i) ? 13 : 9,
          itemStyle: filled
            ? { color, borderColor: color, borderWidth: 1.4, opacity: 0.85 }
            : { color: "transparent", borderColor: color, borderWidth: 1.4, opacity: 0.85 },
          _meta: `${row.variant} · ${disp(row.constructor)} · ${row.improver}\n${scenarioLabel(s)}`,
          _raw: [sub[i].overflows, ys[i]],
        });
      });
      series.push({
        type: "scatter",
        xAxisIndex: pi,
        yAxisIndex: pi,
        data: pts,
        z: 3,
      } as SeriesOption);
      const frontPts = stepFront(
        [...front].map((i) => [xs[i], ys[i]] as [number, number])
      );
      if (frontPts.length) {
        series.push({
          type: "line",
          xAxisIndex: pi,
          yAxisIndex: pi,
          data: frontPts,
          showSymbol: false,
          lineStyle: {
            type: "dashed",
            width: 2,
            color: frontColors.get(`${s.city}|${s.N}|${s.dist}`),
            opacity: 0.9,
          },
          z: 2,
          silent: true,
        } as SeriesOption);
      }
    }
  });

  // Legend: variants (colour), regions (shape), improvers (fill), fronts (scenario)
  const legendData = [
    ...variants.map((v) => ({ name: v, icon: "roundRect", itemStyle: { color: vcolors.get(v) } })),
    ...regions.map(([city, N]) => ({
      name: regionLabel(city, N),
      icon: markers.get(`${city}|${N}`),
      itemStyle: { color: theme.muted },
    })),
    ...(improvers.length > 1
      ? [
          { name: `${improvers[0]} (filled)`, icon: "circle", itemStyle: { color: theme.fg } },
          {
            name: `${improvers[improvers.length - 1]} (open)`,
            icon: "circle",
            itemStyle: { color: "transparent", borderColor: theme.fg, borderWidth: 1.2 },
          },
        ]
      : []),
    ...scenarios.map((s) => ({
      name: `Front — ${scenarioLabel(s)}`,
      // closed dash shapes: a stroke-only path has a zero-height bounding box,
      // which turns the legend layout into NaNs
      icon: "path://M0,4L8,4L8,6L0,6ZM12,4L20,4L20,6L12,6Z",
      itemStyle: { color: frontColors.get(`${s.city}|${s.N}|${s.dist}`) },
    })),
  ];

  return {
    width: grid.width,
    height: grid.height + 70,
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
      legend: {
        bottom: 4,
        data: legendData as never,
        textStyle: { color: theme.fg, fontSize: 13, fontWeight: "bold" },
        itemWidth: 18,
        type: "plain",
      },
      // legend entries are decorative (series are per scenario), keep silent
      tooltip: {
        trigger: "item",
        formatter: (raw: unknown) => {
          const p = raw as { data?: { _meta?: string; _raw?: [number, number] } };
          return p.data?._meta
            ? `${p.data._meta.replace("\n", "<br>")}<br>overflows: ${p.data._raw?.[0].toFixed(1)} · kg/km: ${p.data._raw?.[1].toFixed(3)}`
            : "";
        },
      },
      series: [...series, ...legendPlaceholderSeries(legendData)],
    },
  };
}

// ── KPI bars (ports gen_kpi_bar / _kpi_panel / gen_kpi_combined) ─────────────

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
      err.push({
        x: gi,
        low: panel.useLog ? symlogVals([Math.min(...vals)], true)[0] : Math.min(...vals),
        high: panel.useLog ? symlogVals([Math.max(...vals)], true)[0] : Math.max(...vals),
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

/** Value axis paired with a hidden numeric axis so whiskers can sit between bars. */
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

/** Re-target error bars onto the hidden numeric x-axis of a panel. */
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
    const axes = kpiAxes(categories, `${ylabel} (${ctx.nDays} days)`, theme, pi, useLog);
    xAxes.push(...axes.x);
    yAxes.push(axes.y);
    retargetErrorBars(ps, (gi) => gi * 2 + 1);
    // bars reference the category axis (even indexes)
    for (const s of ps) {
      if ((s as { type?: string }).type === "bar") (s as { xAxisIndex?: number }).xAxisIndex = pi * 2;
    }
    series.push(...ps);
  });

  const legendData = [
    ...ctx.strategies.map((s) => ({ name: s, icon: "roundRect", itemStyle: { color: strategyColor(s) } })),
    ...ctx.improvers.map((imp, i) => ({
      name: imp,
      icon: "roundRect",
      itemStyle: { color: "#909090", decal: HATCH_DECALS[i % HATCH_DECALS.length] as never },
    })),
  ];

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

/** 2×n_dists combined figure: overflows (symlog) on top, kg/km (linear) below. */
export function buildKpiCombined(dfm: SimRow[], ctx: AnalysisCtx, theme: GenTheme): ChartSpec {
  const { dists } = ctx;
  const metrics: [SimMetric, string, boolean][] = [
    ["overflows", "Overflow Count", true],
    ["kgkm", `${KGKM_LABEL} Efficiency`, false],
  ];
  const ncols = dists.length;
  const nrows = 2;
  const width = 760 * ncols;
  const height = 470 * nrows + 80;
  const series: SeriesOption[] = [];
  const titles: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const boxes: PanelBox[] = [];

  let gridIndex = 0;
  metrics.forEach(([metric, ylabel, useLog], mi) => {
    dists.forEach((dist, di) => {
      const box: PanelBox = {
        index: gridIndex,
        left: 7 + di * (93 / ncols),
        top: 7 + mi * 45,
        width: 93 / ncols - 8,
        height: 36,
      };
      boxes.push(box);
      titles.push(panelTitle(dist, box, theme, 15));
      const scen = ctx.scenarios.filter((s) => s.dist === dist);
      const { series: ps, categories } = kpiPanelSeries(dfm, metric, ctx, theme, {
        box,
        gridIndex,
        scenarios: scen,
        useLog,
      });
      const axes = kpiAxes(categories, ylabel, theme, gridIndex, useLog);
      xAxes.push(...axes.x);
      yAxes.push(axes.y);
      retargetErrorBars(ps, (gi) => gi * 2 + 1);
      for (const s of ps) {
        if ((s as { type?: string }).type === "bar") (s as { xAxisIndex?: number }).xAxisIndex = gridIndex * 2;
      }
      series.push(...ps);
      gridIndex++;
    });
  });

  const legendData = [
    ...ctx.strategies.map((s) => ({ name: s, icon: "roundRect", itemStyle: { color: strategyColor(s) } })),
    ...ctx.improvers.map((imp, i) => ({
      name: imp,
      icon: "roundRect",
      itemStyle: { color: "#909090", decal: HATCH_DECALS[i % HATCH_DECALS.length] as never },
    })),
  ];

  return {
    width,
    height,
    option: {
      backgroundColor: theme.bg,
      title: titles,
      grid: boxes.map((b) => ({
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

// ── Policy × scenario heatmap (ports gen_policy_scenario_heatmap) ────────────

function policyRows(df: SimRow[]): [string, string, string][] {
  const seen = new Set<string>();
  const combos: [string, string, string][] = [];
  for (const r of df) {
    const key = `${r.variant}|${r.constructor}|${r.improver}`;
    if (!seen.has(key)) {
      seen.add(key);
      combos.push([r.variant, r.constructor, r.improver]);
    }
  }
  return combos.sort(
    (a, b) => a[0].localeCompare(b[0]) || a[1].localeCompare(b[1]) || a[2].localeCompare(b[2])
  );
}

export function buildPolicyScenarioHeatmap(
  df: SimRow[],
  ctx: AnalysisCtx,
  theme: GenTheme,
  metric: "overflows" | "kgkm"
): ChartSpec {
  const rows = policyRows(df);
  const rowLabels = rows.map(([v, c, i]) => `${v} · ${disp(c)} · ${i}`);
  const colLabels = ctx.scenarios.map(scenarioLabel);
  const mlabel = metric === "overflows" ? "Overflow Count" : `${KGKM_LABEL} Efficiency`;
  const useLog = metric === "overflows";

  const data: [number, number, number][] = [];
  const rawVals = new Map<string, number>();
  rows.forEach(([v, c, i], ri) => {
    const subP = df.filter((r) => r.variant === v && r.constructor === c && r.improver === i);
    ctx.scenarios.forEach((s, ci) => {
      const vals = metricValues(scenSub(subP, s), metric);
      if (vals.length) {
        const raw = mean(vals);
        rawVals.set(`${ci}|${ri}`, raw);
        data.push([ci, ri, useLog ? symlogVals([raw], true)[0] : raw]);
      }
    });
  });
  const transformed = data.map((d) => d[2]);
  const vmin = Math.min(...transformed);
  const vmax = Math.max(...transformed);

  const width = Math.max(1200, 120 * ctx.scenarios.length);
  const height = Math.max(900, 34 * rows.length) + 140;
  return {
    width,
    height,
    option: {
      backgroundColor: theme.bg,
      title: {
        text: `Policy × Scenario Heatmap — ${mlabel} (${ctx.nDays} days)`,
        left: "center",
        top: 8,
        textStyle: { color: theme.fg, fontSize: 17, fontWeight: "bold" },
      },
      grid: { left: 250, right: 110, top: 60, bottom: 90 },
      xAxis: {
        type: "category",
        data: colLabels,
        axisLabel: { color: theme.axisLabelColor, fontSize: 12, fontWeight: "bold", rotate: 30, interval: 0 },
        splitArea: { show: false },
      },
      yAxis: {
        type: "category",
        data: rowLabels,
        inverse: true,
        axisLabel: { color: theme.axisLabelColor, fontSize: 11, fontWeight: "bold" },
      },
      visualMap: {
        min: vmin,
        max: vmax,
        calculable: false,
        orient: "vertical",
        right: 12,
        top: "center",
        inRange: { color: rdYlGnColors(metric === "overflows") },
        text: [mlabel, ""],
        textStyle: { color: theme.fg, fontSize: 11 },
        formatter: useLog
          ? (v: unknown) => symlogAxisLabelFormatter(true)!(Number(v))
          : undefined,
      },
      tooltip: {
        formatter: (rawP: unknown) => {
          const p = rawP as { data?: [number, number, number] };
          if (!p.data) return "";
          const raw = rawVals.get(`${p.data[0]}|${p.data[1]}`);
          return `${rowLabels[p.data[1]]}<br>${colLabels[p.data[0]]}<br>${mlabel}: ${raw?.toFixed(metric === "overflows" ? 1 : 3)}`;
        },
      },
      series: [{ type: "heatmap", data, emphasis: { disabled: true } }],
    },
  };
}

// ── Scenario × constructor heatmaps (ports gen_scenario_constructor_heatmap) ─

export function buildScenarioConstructorHeatmap(
  dfm: SimRow[],
  ctx: AnalysisCtx,
  theme: GenTheme,
  sharedAxisLabels = true,
  scenarioFilter?: (s: Scenario) => boolean
): ChartSpec {
  const scenarios = scenarioFilter ? ctx.scenarios.filter(scenarioFilter) : ctx.scenarios;
  const { strategies, improvers, constructors } = ctx;
  const combos: [string, string][] = strategies.flatMap((s) => improvers.map((i) => [s, i] as [string, string]));
  const comboLabels = combos.map(([s, i]) => `${s}·${i}`);
  const metrics: ["overflows" | "kgkm", boolean, string][] = [
    ["overflows", true, "OVERFLOWS"],
    ["kgkm", false, KGKM_LABEL.toUpperCase()],
  ];
  const n = scenarios.length;
  const width = Math.max(400 * n, 800);
  const height = 860;

  const grids: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const titles: object[] = [];
  const visualMaps: object[] = [];
  const series: SeriesOption[] = [];

  let gi = 0;
  metrics.forEach(([metric, useLog, mlabel], rowI) => {
    scenarios.forEach((s, colI) => {
      const left = 6 + colI * (90 / n);
      const top = 10 + rowI * 44;
      grids.push({ left: `${left}%`, top: `${top}%`, width: `${90 / n - 7}%`, height: "32%" });
      if (rowI === 0) {
        titles.push({
          text: scenarioLabel(s),
          left: `${left + (90 / n - 7) / 2}%`,
          top: "3.5%",
          textAlign: "center",
          textStyle: { color: theme.fg, fontSize: 12, fontWeight: "bold" },
        });
      }
      if (colI === n - 1) {
        titles.push({
          text: mlabel,
          right: 4,
          top: `${top + 14}%`,
          textStyle: { color: theme.fg, fontSize: 12, fontWeight: "bold" },
        });
      }
      xAxes.push({
        type: "category",
        gridIndex: gi,
        data: comboLabels,
        axisLabel: sharedAxisLabels
          ? { color: theme.axisLabelColor, fontSize: 9.5, fontWeight: "bold", interval: 0 }
          : { show: false },
        axisTick: { show: false },
      });
      yAxes.push({
        type: "category",
        gridIndex: gi,
        data: dispAll(constructors),
        inverse: true,
        axisLabel:
          sharedAxisLabels && colI === 0
            ? { color: theme.axisLabelColor, fontSize: 9.5, fontWeight: "bold" }
            : { show: false },
        axisTick: { show: false },
      });
      const sub = scenSub(dfm, s);
      const data: [number, number, number][] = [];
      let vmin = Infinity;
      let vmax = -Infinity;
      constructors.forEach((con, ci) => {
        combos.forEach(([strat, imp], bi) => {
          const vals = metricValues(
            sub.filter((r) => r.constructor === con && r.strategy === strat && r.improver === imp),
            metric
          );
          if (vals.length) {
            const raw = mean(vals);
            const v = useLog ? symlogVals([raw], true)[0] : raw;
            vmin = Math.min(vmin, v);
            vmax = Math.max(vmax, v);
            data.push([bi, ci, v]);
          }
        });
      });
      visualMaps.push({
        min: vmin === Infinity ? 0 : vmin,
        max: vmax === -Infinity ? 1 : vmax,
        show: false,
        seriesIndex: series.length,
        inRange: { color: rdYlGnColors(metric === "overflows") },
      });
      series.push({
        type: "heatmap",
        xAxisIndex: gi,
        yAxisIndex: gi,
        data,
        emphasis: { disabled: true },
      } as SeriesOption);
      gi++;
    });
  });

  return {
    width,
    height,
    option: {
      backgroundColor: theme.bg,
      title: titles as never,
      grid: grids as never,
      xAxis: xAxes as never,
      yAxis: yAxes as never,
      visualMap: visualMaps as never,
      series,
    },
  };
}

// ── Bubble charts (ports gen_strategy_bubble / gen_improver_bubble) ──────────

const LABEL_OFFSETS: [number, number][] = [
  [8, -7], [8, 20], [-58, -7], [-58, 20], [8, -24], [-58, -24], [8, 36], [-58, 36],
  [30, -34], [30, 40], [-80, -7], [8, -40],
];

function placeLabel(placed: [number, number][], nx: number, ny: number): [number, number] {
  for (const [dx, dy] of LABEL_OFFSETS) {
    const onx = nx + dx / 420;
    const ony = ny + dy / 230;
    if (placed.every(([px, py]) => (onx - px) ** 2 + (ony - py) ** 2 > 0.018)) {
      placed.push([onx, ony]);
      return [dx, dy];
    }
  }
  const [dx, dy] = LABEL_OFFSETS[0];
  placed.push([nx + dx / 480, ny + dy / 260]);
  return [dx, dy];
}

function buildBubble(
  dfm: SimRow[],
  ctx: AnalysisCtx,
  theme: GenTheme,
  mode: "strategy" | "improver",
  xscale: Scale
): ChartSpec {
  const { dists, regions } = ctx;
  const keys = mode === "strategy" ? ctx.strategies : ctx.improvers;
  const colorOf = (k: string) =>
    mode === "strategy" ? strategyColor(k) : META.improver_colors[k] ?? "#a0a0a0";
  const markers = regionMarkers(regions);
  const log = xscale !== "linear";
  const grid = panelGrid(dists.length, { legendRows: 1, panelW: 640, panelH: 500 });
  const series: SeriesOption[] = [];
  const titles: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];

  dists.forEach((dist, pi) => {
    const box = grid.boxes[pi];
    titles.push(panelTitle(dist, box, theme, 15));
    xAxes.push({
      type: "value",
      gridIndex: pi,
      scale: true,
      ...axisStyle(theme, { name: "Overflows", nameSize: 15 }),
      axisLabel: {
        color: theme.axisLabelColor,
        fontSize: 12,
        fontWeight: "bold",
        formatter: symlogAxisLabelFormatter(log),
      },
    });
    yAxes.push({
      type: "value",
      gridIndex: pi,
      scale: true,
      ...axisStyle(theme, { name: `${KGKM_LABEL} Efficiency`, nameSize: 15 }),
      nameGap: 42,
    });

    const scen = ctx.scenarios.filter((sc) => sc.dist === dist);
    // gather points first for label placement normalisation
    const pts: { x: number; y: number; key: string; s: Scenario; pair?: string }[] = [];
    for (const s of scen) {
      for (const k of keys) {
        const sub = scenSub(
          dfm.filter((r) => (mode === "strategy" ? r.strategy === k : r.improver === k)),
          s
        );
        if (!sub.length) continue;
        const ov = mean(metricValues(sub, "overflows"));
        pts.push({ x: log ? symlogVals([ov], true)[0] : ov, y: mean(metricValues(sub, "kgkm")), key: k, s });
      }
    }
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const xmin = Math.min(...xs);
    const xmax = Math.max(...xs);
    const ymin = Math.min(...ys);
    const ymax = Math.max(...ys);
    const placed: [number, number][] = [];

    // improver mode: connect pairs per scenario
    if (mode === "improver") {
      for (const s of scen) {
        const pair = pts.filter((p) => p.s === s);
        if (pair.length === 2) {
          series.push({
            type: "line",
            xAxisIndex: pi,
            yAxisIndex: pi,
            data: pair.map((p) => [p.x, p.y]),
            showSymbol: false,
            silent: true,
            lineStyle: { color: theme.guideLine, width: 0.8, opacity: 0.6 },
            z: 1,
          } as SeriesOption);
        }
      }
    }

    series.push({
      type: "scatter",
      xAxisIndex: pi,
      yAxisIndex: pi,
      data: pts.map((p) => {
        const nx = (p.x - xmin) / (xmax - xmin + 1e-9);
        const ny = (p.y - ymin) / (ymax - ymin + 1e-9);
        const [dx, dy] = placeLabel(placed, nx, ny);
        return {
          value: [p.x, p.y],
          symbol: markers.get(`${p.s.city}|${p.s.N}`),
          symbolSize: Math.sqrt(120 + 2 * p.s.N) * 1.9,
          itemStyle: {
            color: colorOf(p.key),
            opacity: 0.78,
            borderColor: theme.accentLine,
            borderWidth: 0.5,
          },
          label: {
            show: true,
            position: [dx, dy] as [number, number],
            formatter: regionLabel(p.s.city, p.s.N),
            color: "#111111",
            fontSize: 10.5,
            fontWeight: "bold",
            backgroundColor: "rgba(255,255,255,0.8)",
            borderRadius: 3,
            padding: [1.5, 3],
          },
          _meta: `${p.key} — ${scenarioLabel(p.s)}`,
        };
      }),
      z: 3,
    } as SeriesOption);
  });

  const legendData = keys.map((k) => ({ name: k, icon: "roundRect", itemStyle: { color: colorOf(k) } }));

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
      legend: {
        bottom: 4,
        data: legendData as never,
        textStyle: { color: theme.fg, fontSize: 14, fontWeight: "bold" },
      },
      tooltip: {
        trigger: "item",
        formatter: (raw: unknown) => (raw as { data?: { _meta?: string } }).data?._meta ?? "",
      },
      series: [...series, ...legendPlaceholderSeries(legendData)],
    },
  };
}

export function buildStrategyBubble(
  dfm: SimRow[],
  ctx: AnalysisCtx,
  theme: GenTheme,
  xscale: Scale = "linear"
): ChartSpec {
  return buildBubble(dfm, ctx, theme, "strategy", xscale);
}

export function buildImproverBubble(
  dfm: SimRow[],
  ctx: AnalysisCtx,
  theme: GenTheme,
  xscale: Scale = "linear"
): ChartSpec {
  return buildBubble(dfm, ctx, theme, "improver", xscale);
}

// ── Constructor ranking (ports gen_constructor_ranking) ─────────────────────

const RANK_METRICS: [SimMetric, string, boolean][] = [
  ["overflows", "Overflows", true],
  ["kgkm", KGKM_LABEL, false],
  ["km", "km", true],
  ["profit", "Profit", false],
];

function averageRanks(dfm: SimRow[], constructors: string[]): Map<string, Map<SimMetric, number>> {
  const acc = new Map<string, Map<SimMetric, number[]>>(
    constructors.map((c) => [c, new Map(RANK_METRICS.map(([m]) => [m, [] as number[]]))])
  );
  const groups = new Map<string, SimRow[]>();
  for (const r of dfm) {
    const key = [r.city, r.N, r.dist, r.strategy, r.improver].join("|");
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(r);
  }
  for (const [metric, , asc] of RANK_METRICS) {
    for (const grp of groups.values()) {
      const sorted = [...grp].sort((a, b) => (asc ? a[metric] - b[metric] : b[metric] - a[metric]));
      // average rank for ties
      const ranks = new Map<string, number[]>();
      sorted.forEach((r, i) => {
        (ranks.get(r.constructor) ?? ranks.set(r.constructor, []).get(r.constructor)!).push(i + 1);
      });
      for (const [con, rs] of ranks) {
        acc.get(con)?.get(metric)?.push(rs.reduce((a, b) => a + b, 0) / rs.length);
      }
    }
  }
  const out = new Map<string, Map<SimMetric, number>>();
  for (const [con, per] of acc) {
    const m = new Map<SimMetric, number>();
    for (const [metric, vals] of per) m.set(metric, vals.length ? mean(vals) : NaN);
    out.set(con, m);
  }
  return out;
}

export function buildConstructorRanking(dfm: SimRow[], ctx: AnalysisCtx, theme: GenTheme): ChartSpec {
  const { constructors } = ctx;
  const ranks = averageRanks(dfm, constructors);
  const series: SeriesOption[] = RANK_METRICS.map(([metric, label], mi) => ({
    type: "bar",
    name: label,
    data: constructors.map((c) => {
      const v = ranks.get(c)?.get(metric);
      return v !== undefined && Number.isFinite(v) ? Number(v.toFixed(3)) : null;
    }),
    itemStyle: { color: META.metric_colors[mi % META.metric_colors.length], opacity: 0.85 },
  }));
  const width = Math.max(1200, 180 * constructors.length);
  return {
    width,
    height: 640,
    option: {
      backgroundColor: theme.bg,
      grid: { left: 80, right: 30, top: 50, bottom: 70 },
      legend: { top: 8, left: 80, textStyle: { color: theme.fg, fontSize: 12 } },
      xAxis: {
        type: "category",
        data: dispAll(constructors),
        axisLabel: { color: theme.axisLabelColor, fontSize: 11, rotate: 15, interval: 0 },
      },
      yAxis: {
        type: "value",
        ...axisStyle(theme, { name: "Average rank (lower = better)" }),
        nameGap: 38,
      },
      tooltip: { trigger: "axis" },
      series,
    },
  };
}

// ── Radar (ports _render_radar / gen_radar / gen_radar_combined) ─────────────

const RADAR_BG = "#1a1a2e";
const RADAR_AX_BG = "#16213e";
const RADAR_MUTED = "#a0a0b0";
const RADAR_FAINT = "#2d2d4e";

export function buildRadar(dfm: SimRow[], key: string[], title: string): ChartSpec {
  const metrics: [SimMetric, string, boolean][] = [
    ["overflows", "Overflows\n(fewer ↓)", true],
    ["kgkm", `${KGKM_LABEL}\n(higher ↑)`, false],
    ["km", "km\n(fewer ↓)", true],
    ["profit", "Profit\n(higher ↑)", false],
  ];
  const allVals = new Map<SimMetric, number[]>(
    metrics.map(([m]) => [m, metricValues(dfm, m)])
  );
  const scores = key.map((c) => {
    const sub = dfm.filter((r) => r.constructor === c);
    return metrics.map(([metric, , inv]) => {
      const vals = allVals.get(metric)!;
      const v = sub.length ? mean(metricValues(sub, metric)) : mean(vals);
      const mn = Math.min(...vals);
      const mx = Math.max(...vals);
      const norm = mx > mn ? (v - mn) / (mx - mn + 1e-9) : 0.5;
      return inv ? 1 - norm : norm;
    });
  });

  return {
    width: 980,
    height: 900,
    background: RADAR_BG,
    option: {
      backgroundColor: RADAR_BG,
      title: {
        text: title,
        left: "center",
        top: 14,
        textStyle: { color: "#ffffff", fontSize: 18, fontWeight: "bold" },
      },
      legend: {
        top: 70,
        right: 20,
        orient: "vertical",
        textStyle: { color: "#ffffff", fontSize: 14 },
      },
      radar: {
        center: ["50%", "56%"],
        radius: "62%",
        indicator: metrics.map(([, label]) => ({ name: label.replace("\n", " "), max: 1 })),
        splitNumber: 4,
        axisName: { color: "#ffffff", fontSize: 15 },
        splitLine: { lineStyle: { color: RADAR_FAINT, type: "dashed" } },
        splitArea: { areaStyle: { color: [RADAR_AX_BG] } },
        axisLine: { lineStyle: { color: RADAR_MUTED } },
      },
      series: [
        {
          type: "radar",
          symbol: "circle",
          symbolSize: 5,
          data: key.map((c, i) => ({
            name: disp(c),
            value: scores[i],
            lineStyle: { color: META.constructor_colors[c] ?? "#e0e0e0", width: 2.5 },
            itemStyle: { color: META.constructor_colors[c] ?? "#e0e0e0" },
            areaStyle: { color: META.constructor_colors[c] ?? "#e0e0e0", opacity: 0.08 },
          })),
        },
      ],
    },
  };
}

// ── Improver delta heatmap (ports gen_improver_delta) ────────────────────────

export function buildImproverDelta(dfm: SimRow[], ctx: AnalysisCtx, theme: GenTheme): ChartSpec | null {
  const { improvers, scenarios, strategies, constructors } = ctx;
  if (improvers.length < 2) return null;
  const impA = improvers[0];
  const impB = improvers[improvers.length - 1];
  const configs: [Scenario, string][] = scenarios.flatMap((s) =>
    strategies.map((strat) => [s, strat] as [Scenario, string])
  );
  const colLabels = configs.map(
    ([s, strat]) => `${regionLabel(s.city, s.N)} ${s.dist.slice(0, 3)} ${strat}`
  );
  const metrics: ["overflows" | "kgkm", boolean][] = [
    ["overflows", true],
    ["kgkm", false],
  ];

  const grids: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const titles: object[] = [];
  const visualMaps: object[] = [];
  const series: SeriesOption[] = [];

  metrics.forEach(([metric, reversed], pi) => {
    grids.push({
      left: `${6 + pi * 47}%`,
      top: "12%",
      width: "40%",
      height: "68%",
    });
    titles.push({
      text: `Δ ${metric}`,
      left: `${6 + pi * 47 + 20}%`,
      top: "4%",
      textAlign: "center",
      textStyle: { color: theme.fg, fontSize: 13, fontWeight: "bold" },
    });
    xAxes.push({
      type: "category",
      gridIndex: pi,
      data: colLabels,
      axisLabel: { color: theme.axisLabelColor, fontSize: 7, rotate: 45, interval: 0 },
    });
    yAxes.push({
      type: "category",
      gridIndex: pi,
      data: dispAll(constructors),
      inverse: true,
      axisLabel: { color: theme.axisLabelColor, fontSize: 9.5 },
    });
    const data: [number, number, number][] = [];
    const abs: number[] = [];
    constructors.forEach((c, ci) => {
      configs.forEach(([s, strat], cfi) => {
        const sub = scenSub(dfm.filter((r) => r.strategy === strat && r.constructor === c), s);
        const a = sub.find((r) => r.improver === impA);
        const b = sub.find((r) => r.improver === impB);
        if (a && b) {
          const d = b[metric] - a[metric];
          data.push([cfi, ci, d]);
          abs.push(Math.abs(d));
        }
      });
    });
    abs.sort((x, y) => x - y);
    const vmax = abs.length ? abs[Math.min(abs.length - 1, Math.floor(abs.length * 0.95))] : 1;
    visualMaps.push({
      min: -vmax,
      max: vmax,
      show: false,
      seriesIndex: pi,
      inRange: { color: rdYlGnColors(reversed) },
    });
    series.push({
      type: "heatmap",
      xAxisIndex: pi,
      yAxisIndex: pi,
      data,
      emphasis: { disabled: true },
    } as SeriesOption);
  });

  return {
    width: 2400,
    height: 860,
    option: {
      backgroundColor: theme.bg,
      title: titles as never,
      grid: grids as never,
      xAxis: xAxes as never,
      yAxis: yAxes as never,
      visualMap: visualMaps as never,
      series,
    },
  };
}

// ── Horizon comparison charts (ports gen_horizon_comparison) ─────────────────

function horizonConfigMeans(
  dfm: SimRow[],
  configs: [Scenario, string][],
  metric: SimMetric
): (number | null)[] {
  return configs.map(([s, strat]) => {
    const vals = metricValues(scenSub(dfm.filter((r) => r.strategy === strat), s), metric);
    return vals.length ? mean(vals) : null;
  });
}

export function buildHorizonComparison(
  horizons: HorizonData[],
  ctx: AnalysisCtx,
  theme: GenTheme,
  metric: "overflows" | "kgkm"
): ChartSpec {
  const configs: [Scenario, string][] = ctx.scenarios.flatMap((s) =>
    ctx.strategies.map((strat) => [s, strat] as [Scenario, string])
  );
  const colLabels = configs.map(
    ([s, strat]) => `${regionLabel(s.city, s.N)} ${s.dist.slice(0, 3)} ${strat}`
  );
  const ylabel = metric === "overflows" ? "Mean overflows" : "Mean kg/km";
  const series: SeriesOption[] = horizons.map((h, hi) => ({
    type: "bar",
    name: `${h.days} days`,
    data: horizonConfigMeans(h.dfm, configs, metric),
    itemStyle: { color: META.horizon_colors[hi % META.horizon_colors.length], opacity: 0.85 },
  }));
  return {
    width: Math.max(1800, 90 * configs.length),
    height: 640,
    option: {
      backgroundColor: theme.bg,
      grid: { left: 80, right: 30, top: 50, bottom: 120 },
      legend: { top: 8, left: 80, textStyle: { color: theme.fg, fontSize: 12 } },
      xAxis: {
        type: "category",
        data: colLabels,
        axisLabel: { color: theme.axisLabelColor, fontSize: 8, rotate: 45, interval: 0 },
      },
      yAxis: { type: "value", ...axisStyle(theme, { name: ylabel }), nameGap: 44 },
      tooltip: { trigger: "axis" },
      series,
    },
  };
}

export function buildHorizonDelta(
  horizons: HorizonData[],
  ctx: AnalysisCtx,
  theme: GenTheme
): ChartSpec {
  const configs: [Scenario, string][] = ctx.scenarios.flatMap((s) =>
    ctx.strategies.map((strat) => [s, strat] as [Scenario, string])
  );
  const colLabels = configs.map(
    ([s, strat]) => `${regionLabel(s.city, s.N)} ${s.dist.slice(0, 3)} ${strat}`
  );
  const first = horizonConfigMeans(horizons[0].dfm, configs, "overflows");
  const last = horizonConfigMeans(horizons[horizons.length - 1].dfm, configs, "overflows");
  const data = configs.map((_, i) => {
    const a = first[i];
    const b = last[i];
    if (a == null || b == null || a <= 0) return null;
    return ((b - a) / a) * 100;
  });
  return {
    width: Math.max(1800, 90 * configs.length),
    height: 640,
    option: {
      backgroundColor: theme.bg,
      grid: { left: 80, right: 30, top: 40, bottom: 120 },
      xAxis: {
        type: "category",
        data: colLabels,
        axisLabel: { color: theme.axisLabelColor, fontSize: 8, rotate: 45, interval: 0 },
      },
      yAxis: { type: "value", ...axisStyle(theme, { name: "Δ overflows (%)" }), nameGap: 44 },
      series: [
        {
          type: "bar",
          data: data.map((v) => ({
            value: v,
            itemStyle: { color: v != null && v > 0 ? "#e05c5c" : "#5cb85c", opacity: 0.85 },
          })),
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: theme.fg, type: "dashed", width: 0.8 },
            data: [{ yAxis: 0 }],
          },
        },
      ],
    },
  };
}

export function buildHorizonConstructorRanking(
  horizons: HorizonData[],
  theme: GenTheme
): ChartSpec {
  const allCons = [
    ...new Set(horizons.flatMap((h) => h.dfm.map((r) => r.constructor))),
  ].sort();
  const series: SeriesOption[] = horizons.map((h, hi) => {
    const ranks = averageRanks(h.dfm, allCons);
    return {
      type: "bar",
      name: `${h.days} days`,
      data: allCons.map((c) => {
        const per = ranks.get(c);
        if (!per) return null;
        const vals = RANK_METRICS.map(([m]) => per.get(m)).filter(
          (v): v is number => v !== undefined && Number.isFinite(v)
        );
        return vals.length ? Number(mean(vals).toFixed(3)) : null;
      }),
      itemStyle: { color: META.horizon_colors[hi % META.horizon_colors.length], opacity: 0.85 },
    } as SeriesOption;
  });
  return {
    width: Math.max(1200, 180 * allCons.length),
    height: 640,
    option: {
      backgroundColor: theme.bg,
      grid: { left: 80, right: 30, top: 50, bottom: 70 },
      legend: { top: 8, left: 80, textStyle: { color: theme.fg, fontSize: 12 } },
      xAxis: {
        type: "category",
        data: dispAll(allCons),
        axisLabel: { color: theme.axisLabelColor, fontSize: 11, rotate: 15, interval: 0 },
      },
      yAxis: {
        type: "value",
        ...axisStyle(theme, { name: "Average rank (lower = better)" }),
        nameGap: 38,
      },
      tooltip: { trigger: "axis" },
      series,
    },
  };
}

export { groupSpans };
