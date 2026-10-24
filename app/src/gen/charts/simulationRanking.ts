/**
 * Constructor ranking, radar charts, and horizon comparison builders (§H.2).
 */
import type { SeriesOption } from "echarts";
import { META, KGKM_LABEL, disp, dispAll, type GenTheme } from "../config";
import { axisStyle, type ChartSpec } from "./common";
import {
  scenSub,
  regionLabel,
  metricValues,
  mean,
  type AnalysisCtx,
  type Scenario,
  type SimRow,
  type SimMetric,
  type HorizonData,
} from "../data/simulation";

export const RANK_METRICS: [SimMetric, string, boolean][] = [
  ["overflows", "Overflows", true],
  ["kgkm", KGKM_LABEL, false],
  ["km", "km", true],
  ["profit", "Profit", false],
];

export function averageRanks(dfm: SimRow[], constructors: string[]): Map<string, Map<SimMetric, number>> {
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
  const n = constructors.length;

  const series: SeriesOption[] = RANK_METRICS.map(([metric, label], mi) => ({
    type: "bar",
    name: label,
    data: constructors.map((c) => {
      const v = ranks.get(c)?.get(metric);
      return v != null && Number.isFinite(v) ? Number(v.toFixed(3)) : null;
    }),
    itemStyle: { color: ["#5cb85c", "#f0ad4e", "#5bc0de", "#d9534f"][mi] },
  }));

  return {
    width: Math.max(900, 140 * n),
    height: 480,
    option: {
      backgroundColor: theme.bg,
      title: {
        text: "Average Rank by Constructor (lower = better)",
        left: "center",
        top: 8,
        textStyle: { color: theme.fg, fontSize: 13, fontWeight: "bold" },
      },
      grid: { left: 80, right: 30, top: 50, bottom: 60 },
      legend: { top: 8, right: 30, textStyle: { color: theme.fg, fontSize: 11 } },
      xAxis: {
        type: "category",
        data: dispAll(constructors),
        axisLabel: { color: theme.axisLabelColor, fontSize: 11 },
      },
      yAxis: {
        type: "value",
        ...axisStyle(theme, { name: "Average rank (1 = best)" }),
        min: 1,
        max: n,
        nameGap: 36,
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
