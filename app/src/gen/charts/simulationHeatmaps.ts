/**
 * Policy, scenario constructor, and improver delta heatmap builders (§H.2).
 */
import type { SeriesOption } from "echarts";
import { KGKM_LABEL, disp, dispAll, type GenTheme } from "../config";
import { symlogVals, rdYlGnColors, type ChartSpec } from "./common";
import {
  scenSub,
  scenarioLabel,
  regionLabel,
  metricValues,
  mean,
  type AnalysisCtx,
  type Scenario,
  type SimRow,
} from "../data/simulation";

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

  const vals = data.map((d) => d[2]);
  const minVal = vals.length ? Math.min(...vals) : 0;
  const maxVal = vals.length ? Math.max(...vals) : 1;

  const width = Math.max(900, 110 * colLabels.length + 260);
  const height = Math.max(700, 26 * rowLabels.length + 140);

  return {
    width,
    height,
    option: {
      backgroundColor: theme.bg,
      title: {
        text: `Policy × Scenario — ${mlabel}`,
        left: "center",
        top: 8,
        textStyle: { color: theme.fg, fontSize: 13, fontWeight: "bold" },
      },
      grid: { left: 240, right: 90, top: 50, bottom: 90 },
      xAxis: {
        type: "category",
        data: colLabels,
        axisLabel: { color: theme.axisLabelColor, fontSize: 10, rotate: 25, interval: 0 },
        splitArea: { show: true },
      },
      yAxis: {
        type: "category",
        data: rowLabels,
        inverse: true,
        axisLabel: { color: theme.axisLabelColor, fontSize: 9.5 },
        splitArea: { show: true },
      },
      visualMap: {
        min: minVal,
        max: maxVal,
        calculable: true,
        orient: "vertical",
        right: 14,
        top: "center",
        textStyle: { color: theme.fg, fontSize: 10 },
        inRange: { color: rdYlGnColors(metric === "overflows") },
      },
      tooltip: {
        trigger: "item",
        formatter: (raw: unknown) => {
          const d = raw as { data?: [number, number, number] };
          if (!d.data) return "";
          const [ci, ri] = d.data;
          const rawV = rawVals.get(`${ci}|${ri}`);
          return `${rowLabels[ri]}<br>${colLabels[ci]}<br>${metric}: ${rawV?.toFixed(2) ?? "—"}`;
        },
      },
      series: [
        {
          type: "heatmap",
          data,
          label: {
            show: true,
            formatter: (p: unknown) => {
              const d = (p as { data?: [number, number, number] }).data;
              if (!d) return "";
              const rawV = rawVals.get(`${d[0]}|${d[1]}`);
              if (rawV == null) return "";
              return metric === "overflows" ? rawV.toFixed(1) : rawV.toFixed(2);
            },
            color: "#1a1a1a",
            fontSize: 8.5,
          },
          emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.5)" } },
        },
      ],
    },
  };
}

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
