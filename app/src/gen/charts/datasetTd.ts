/**
 * Training Data (TD) chart builders and TD vs NPZ alignment (§H.2).
 */
import type { SeriesOption } from "echarts";
import type { GenTheme } from "../config";
import { axisStyle, type ChartSpec } from "./common";
import type { NpzStatRow, TdStatRow } from "../data/dataset";
import {
  distColor,
  distLabel,
  linePlusRef,
  lineGridChart,
  subGrid,
  supTitle,
  subTitle,
  LINE_CITY,
  REF_CITY,
} from "./dataset";

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
