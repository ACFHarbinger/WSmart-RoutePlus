/**
 * Dataset Interactive HTML charts: npz_stats, waste_distribution, city_network_comparison.
 */
import { DATASET_CFG, type GenTheme } from "../config";
import type { NpzStatRow } from "../data/dataset";
import type { InteractivePage } from "./interactiveHtml";

const CITY_LABELS = DATASET_CFG.city_labels;
const DIST_LABELS = DATASET_CFG.dist_labels;
const CITY_COLORS = DATASET_CFG.city_colors;
const DIST_COLORS = DATASET_CFG.dist_colors;

function baseAxes(theme: GenTheme, xName: string, yName: string) {
  const axis = (name: string) => ({
    type: "value",
    name,
    nameLocation: "middle",
    nameGap: 32,
    scale: true,
    nameTextStyle: { color: theme.axisLabelColor, fontSize: 13 },
    axisLabel: { color: theme.axisLabelColor },
    axisLine: { lineStyle: { color: theme.axisLabelColor } },
    splitLine: { lineStyle: { color: theme.gridColor, opacity: theme.gridAlpha } },
  });
  return { xAxis: axis(xName), yAxis: axis(yName) };
}

export function buildNpzStatsInteractive(npz: NpzStatRow[], theme: GenTheme): InteractivePage {
  const sub30 = npz.filter((r) => r.horizon === 30);
  const cities = [...new Set(sub30.map((r) => r.city))].sort();
  const dists = [...new Set(sub30.map((r) => r.dist))].sort();
  const series: object[] = [];
  for (const city of cities) {
    for (const dist of dists) {
      const rows = sub30.filter((r) => r.city === city && r.dist === dist);
      if (!rows.length) continue;
      series.push({
        type: "scatter",
        name: `${CITY_LABELS[city] ?? city} — ${DIST_LABELS[dist] ?? dist}`,
        data: rows.map((r) => ({
          value: [r.mean_kg, r.std_kg],
          name:
            `${CITY_LABELS[city] ?? city}<br>N=${r.N}<br>Dist: ${DIST_LABELS[dist] ?? dist}<br>` +
            `Mean kg: ${r.mean_kg.toFixed(3)}<br>Std kg: ${r.std_kg.toFixed(3)}<br>` +
            `Max kg: ${r.max_kg.toFixed(1)}<br>Skewness: ${r.skewness.toFixed(3)}`,
          symbol: (DIST_LABELS[dist] ?? dist) === "Empirical" ? "circle" : "rect",
          symbolSize: r.N / 20 + 10,
          itemStyle: { color: CITY_COLORS[city] ?? "#a0a0a0", opacity: 0.85, borderColor: theme.fg, borderWidth: 1 },
        })),
      });
    }
  }
  const { xAxis, yAxis } = baseAxes(theme, "Mean waste (kg/bin/day)", "Std waste (kg/bin/day)");
  return {
    title: "NPZ Dataset Statistics — Mean vs Std Waste (30-day horizon)",
    height: 600,
    option: {
      backgroundColor: theme.bg,
      tooltip: { trigger: "item", formatter: "{b}" },
      legend: { bottom: 4, textStyle: { color: theme.fg, fontSize: 11 } },
      grid: { left: 80, right: 30, top: 30, bottom: 70 },
      xAxis,
      yAxis,
      series,
    },
  };
}

export function buildWasteDistributionInteractive(npz: NpzStatRow[], theme: GenTheme): InteractivePage {
  const sub30 = npz.filter((r) => r.horizon === 30);
  const cities = [...new Set(sub30.map((r) => r.city))].sort();
  const dists = [...new Set(sub30.map((r) => r.dist))].sort();
  const grids: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const titles: object[] = [];
  const series: object[] = [];
  cities.forEach((city, ci) => {
    const left = 7 + ci * (90 / cities.length);
    grids.push({ left: `${left}%`, top: "14%", width: `${90 / cities.length - 6}%`, height: "70%" });
    titles.push({
      text: CITY_LABELS[city] ?? city,
      left: `${left + (90 / cities.length - 6) / 2}%`,
      top: "6%",
      textAlign: "center",
      textStyle: { color: theme.fg, fontSize: 13 },
    });
    const rows = sub30.filter((r) => r.city === city).sort((a, b) => a.N - b.N);
    const ns = [...new Set(rows.map((r) => r.N))];
    xAxes.push({
      type: "category",
      gridIndex: ci,
      data: ns.map((n) => `N=${n}`),
      axisLabel: { color: theme.axisLabelColor, fontSize: 10 },
    });
    yAxes.push({
      type: "value",
      gridIndex: ci,
      name: ci === 0 ? "Mean waste (kg/bin/day)" : "",
      nameTextStyle: { color: theme.axisLabelColor },
      axisLabel: { color: theme.axisLabelColor },
      splitLine: { lineStyle: { color: theme.gridColor, opacity: theme.gridAlpha } },
    });
    for (const dist of dists) {
      const byN = new Map(rows.filter((r) => r.dist === dist).map((r) => [r.N, r]));
      series.push({
        type: "bar",
        name: DIST_LABELS[dist] ?? dist,
        xAxisIndex: ci,
        yAxisIndex: ci,
        data: ns.map((n) => {
          const r = byN.get(n);
          return r
            ? {
                value: r.mean_kg,
                name:
                  `${CITY_LABELS[city] ?? city}, N=${n}, ${DIST_LABELS[dist] ?? dist}<br>` +
                  `Mean: ${r.mean_kg.toFixed(3)} kg<br>Std: ${r.std_kg.toFixed(3)} kg`,
              }
            : null;
        }),
        itemStyle: { color: DIST_COLORS[dist] ?? "gray", opacity: 0.85 },
      });
    }
  });
  return {
    title: "Waste Distribution Statistics per City and Network Size",
    height: 550,
    option: {
      backgroundColor: theme.bg,
      tooltip: { trigger: "item", formatter: "{b}" },
      legend: { bottom: 2, textStyle: { color: theme.fg, fontSize: 11 } },
      title: titles,
      grid: grids,
      xAxis: xAxes,
      yAxis: yAxes,
      series,
    },
  };
}

export function buildCityNetworkComparisonInteractive(npz: NpzStatRow[], theme: GenTheme): InteractivePage {
  const sub30 = npz.filter((r) => r.horizon === 30);
  const cities = [...new Set(sub30.map((r) => r.city))].sort();
  const dists = [...new Set(sub30.map((r) => r.dist))].sort();
  const metrics: [keyof NpzStatRow, string][] = [
    ["mean_kg", "Mean Waste (kg/bin)"],
    ["std_kg", "Std Waste (kg/bin)"],
    ["skewness", "Skewness"],
    ["max_kg", "Max Waste (kg)"],
  ];
  const grids: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const titles: object[] = [];
  const series: object[] = [];
  const categories = sub30
    .sort((a, b) => a.city.localeCompare(b.city) || a.N - b.N)
    .map((r) => `${(CITY_LABELS[r.city] ?? r.city).slice(0, 3)} N=${r.N}`)
    .filter((v, i, arr) => arr.indexOf(v) === i);

  metrics.forEach(([metric, label], mi) => {
    const col = mi % 2;
    const row = Math.floor(mi / 2);
    grids.push({ left: `${7 + col * 47}%`, top: `${11 + row * 45}%`, width: "40%", height: "33%" });
    titles.push({
      text: label,
      left: `${7 + col * 47 + 20}%`,
      top: `${5 + row * 45}%`,
      textAlign: "center",
      textStyle: { color: theme.fg, fontSize: 12 },
    });
    xAxes.push({
      type: "category",
      gridIndex: mi,
      data: categories,
      axisLabel: { color: theme.axisLabelColor, fontSize: 9, rotate: 30, interval: 0 },
    });
    yAxes.push({
      type: "value",
      gridIndex: mi,
      scale: true,
      axisLabel: { color: theme.axisLabelColor, fontSize: 10 },
      splitLine: { lineStyle: { color: theme.gridColor, opacity: theme.gridAlpha } },
    });
    for (const city of cities) {
      for (const dist of dists) {
        const rows = sub30.filter((r) => r.city === city && r.dist === dist);
        if (!rows.length) continue;
        const byCat = new Map(
          rows.map((r) => [`${(CITY_LABELS[city] ?? city).slice(0, 3)} N=${r.N}`, r])
        );
        series.push({
          type: "bar",
          name: `${CITY_LABELS[city] ?? city} ${DIST_LABELS[dist] ?? dist}`,
          xAxisIndex: mi,
          yAxisIndex: mi,
          data: categories.map((cat) => {
            const r = byCat.get(cat);
            return r
              ? {
                  value: r[metric] as number,
                  name:
                    `${CITY_LABELS[city] ?? city}<br>N=${r.N}<br>Dist: ${DIST_LABELS[dist] ?? dist}<br>` +
                    `${String(metric)}: ${(r[metric] as number).toFixed(3)}`,
                }
              : null;
          }),
          itemStyle: { color: CITY_COLORS[city] ?? "gray", opacity: dist === "emp" ? 0.85 : 0.55 },
        });
      }
    }
  });
  return {
    title: "City & Network Comparison — NPZ Dataset Statistics",
    height: 800,
    option: {
      backgroundColor: theme.bg,
      tooltip: { trigger: "item", formatter: "{b}" },
      legend: { type: "scroll", bottom: 2, textStyle: { color: theme.fg, fontSize: 10 } },
      title: titles,
      grid: grids,
      xAxis: xAxes,
      yAxis: yAxes,
      series,
    },
  };
}
