/**
 * Pareto scatter and strategy/improver bubble chart builders (§H.2).
 */
import type { SeriesOption } from "echarts";
import { META, KGKM_LABEL, disp, type GenTheme } from "../config";
import {
  panelGrid,
  panelTitle,
  axisStyle,
  legendPlaceholderSeries,
  stepFront,
  symlogVals,
  symlogAxisLabelFormatter,
  MARKER_CYCLE,
  type ChartSpec,
} from "./common";
import {
  scenSub,
  scenarioLabel,
  regionLabel,
  variantColor,
  paretoIndices,
  metricValues,
  mean,
  type AnalysisCtx,
  type SimRow,
} from "../data/simulation";

export type Scale = "linear" | "symlog";

export const strategyColor = (s: string): string => META.strategy_colors[s] ?? "#a0a0a0";

export function regionMarkers(regions: [string, number][]): Map<string, string> {
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
  const { dists } = ctx;
  const groups = mode === "strategy" ? ctx.strategies : ctx.improvers;
  const useLog = xscale !== "linear";
  const grid = panelGrid(dists.length, { legendRows: 1, panelW: 700, panelH: 520 });

  const series: SeriesOption[] = [];
  const titles: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];

  dists.forEach((dist, pi) => {
    const box = grid.boxes[pi];
    titles.push(panelTitle(`Policy Trade-offs — ${dist}`, box, theme, 13));
    xAxes.push({
      type: "value",
      gridIndex: pi,
      ...axisStyle(theme, { name: "Overflow Count" }),
      nameGap: 30,
      axisLabel: {
        color: theme.axisLabelColor,
        fontSize: 10,
        formatter: symlogAxisLabelFormatter(useLog),
      },
    });
    yAxes.push({
      type: "value",
      gridIndex: pi,
      ...axisStyle(theme, { name: "Distance (km)" }),
      nameGap: 42,
      axisLabel: { color: theme.axisLabelColor, fontSize: 10 },
    });

    const scen = ctx.scenarios.filter((s) => s.dist === dist);
    const subDist = dfm.filter((r) => scen.some((s) => s.city === r.city && s.N === r.N));
    const pts: {
      group: string;
      con: string;
      ov: number;
      km: number;
      kgkm: number;
    }[] = [];

    for (const g of groups) {
      for (const con of ctx.constructors) {
        const sub = subDist.filter(
          (r) => (mode === "strategy" ? r.strategy === g : r.improver === g) && r.constructor === con
        );
        const ovs = metricValues(sub, "overflows");
        const kms = metricValues(sub, "km");
        const kgs = metricValues(sub, "kgkm");
        if (!ovs.length || !kms.length) continue;
        pts.push({
          group: g,
          con,
          ov: mean(ovs),
          km: mean(kms),
          kgkm: kgs.length ? mean(kgs) : 1,
        });
      }
    }

    const ovVals = useLog ? symlogVals(pts.map((p) => p.ov), true) : pts.map((p) => p.ov);
    const minOv = Math.min(...ovVals, 0);
    const maxOv = Math.max(...ovVals, 1);
    const minKm = Math.min(...pts.map((p) => p.km), 0);
    const maxKm = Math.max(...pts.map((p) => p.km), 1);

    const placedLabels: [number, number][] = [];
    const colorOf = (g: string) => (mode === "strategy" ? strategyColor(g) : META.improver_colors[g] ?? "#808080");

    const scatterData = pts.map((p, i) => {
      const nx = (ovVals[i] - minOv) / (maxOv - minOv + 1e-9);
      const ny = (p.km - minKm) / (maxKm - minKm + 1e-9);
      const [ox, oy] = placeLabel(placedLabels, nx, ny);
      return {
        value: [ovVals[i], p.km, p.kgkm],
        symbolSize: Math.max(10, Math.min(32, p.kgkm * 0.45)),
        itemStyle: { color: colorOf(p.group), opacity: 0.8 },
        label: {
          show: true,
          formatter: disp(p.con),
          color: theme.fg,
          fontSize: 8.5,
          offset: [ox, oy],
        },
        _raw: [p.ov, p.km, p.kgkm, p.group, p.con],
      };
    });

    series.push({
      type: "scatter",
      xAxisIndex: pi,
      yAxisIndex: pi,
      data: scatterData,
    });
  });

  const legendData = groups.map((g) => ({
    name: g,
    icon: "circle",
    itemStyle: { color: mode === "strategy" ? strategyColor(g) : META.improver_colors[g] ?? "#808080" },
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
      tooltip: {
        trigger: "item",
        formatter: (raw: unknown) => {
          const d = raw as { data?: { _raw?: [number, number, number, string, string] } };
          if (!d.data?._raw) return "";
          const [ov, km, kgkm, g, con] = d.data._raw;
          return `${g} · ${disp(con)}<br>overflows: ${ov.toFixed(1)}<br>distance: ${km.toFixed(0)} km<br>kg/km: ${kgkm.toFixed(3)}`;
        },
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
