/**
 * Figure rendering pipeline for simulation analysis report (§H.5).
 */
import { SIM_CFG, type GenTheme } from "../config";
import { joinPath, writeBinaryFile } from "../io";
import { renderChartPng, type ChartSpec } from "../charts/common";
import {
  buildParetoScatter,
  buildKpiBar,
  buildKpiCombined,
  buildKmViolin,
  buildPolicyScenarioHeatmap,
  buildScenarioConstructorHeatmap,
  buildStrategyBubble,
  buildImproverBubble,
  buildConstructorRanking,
  buildRadar,
  buildImproverDelta,
} from "../charts/simulation";
import { KGKM_LABEL, type HorizonData } from "../data/simulation";
import { buildBinMapChart, loadBinCoords, loadSelectedBinCoords, SELECTED_MAP_SPECS } from "./binMaps";
import type { SimReportOptions, Progress } from "./simulationReport";

export interface HorizonRender {
  data: HorizonData;
  csv: string;
  figuresDir: string;
  privateDir: string;
  interactive: boolean;
  flags: {
    paretoLog: boolean;
    overflowLog: boolean;
    bubbleLog: boolean;
    improverDelta: boolean;
  };
}

export async function saveFigure(
  projectRoot: string,
  dir: string,
  name: string,
  spec: ChartSpec | null,
  progress: Progress
): Promise<boolean> {
  if (!spec) return false;
  const png = await renderChartPng(spec);
  await writeBinaryFile(joinPath(projectRoot, `${dir}/${name}`), png);
  progress(`Saved: ${name}`);
  return true;
}

export async function renderHorizonFigures(
  h: HorizonData,
  figuresDir: string,
  theme: GenTheme,
  opts: SimReportOptions,
  progress: Progress
): Promise<HorizonRender["flags"]> {
  const root = opts.projectRoot;
  const charts = SIM_CFG.charts;
  const on = (name: string) => (charts[name]?.enabled as boolean | undefined) ?? true;
  const scales = (name: string, key: string): string[] => {
    const v = charts[name]?.[key];
    if (!v) return ["linear"];
    return Array.isArray(v) ? (v as string[]) : [String(v)];
  };
  const flags = { paretoLog: false, overflowLog: false, bubbleLog: false, improverDelta: false };
  const { rows, dfm, ctx } = h;

  if (on("pareto_scatter")) {
    const sc = scales("pareto_scatter", "x_scale");
    await saveFigure(root, figuresDir, "pareto_scatter.png", buildParetoScatter(rows, ctx, theme, sc[0] as never), progress);
    if (sc.length > 1) {
      flags.paretoLog = await saveFigure(
        root, figuresDir, "pareto_scatter_log.png", buildParetoScatter(rows, ctx, theme, sc[1] as never), progress
      );
    }
  }
  if (on("overflow_bar")) {
    const sc = scales("overflow_bar", "y_scale");
    await saveFigure(root, figuresDir, "overflow_by_config.png", buildKpiBar(dfm, "overflows", "Overflow Count", ctx, theme, sc[0] as never), progress);
    if (sc.length > 1) {
      flags.overflowLog = await saveFigure(
        root, figuresDir, "overflow_by_config_log.png", buildKpiBar(dfm, "overflows", "Overflow Count", ctx, theme, sc[1] as never), progress
      );
    }
  }
  if (on("kgkm_bar")) {
    await saveFigure(root, figuresDir, "kgkm_by_config.png", buildKpiBar(dfm, "kgkm", `${KGKM_LABEL} Efficiency`, ctx, theme), progress);
  }
  if (on("kpi_combined")) {
    await saveFigure(root, figuresDir, "kpi_combined.png", buildKpiCombined(dfm, ctx, theme), progress);
  }
  if (on("km_violin")) {
    await saveFigure(root, figuresDir, "km_violin.png", buildKmViolin(rows, ctx, theme), progress);
  }
  if (on("policy_scenario_heatmap")) {
    await saveFigure(root, figuresDir, "policy_scenario_heatmap_overflows.png", buildPolicyScenarioHeatmap(rows, ctx, theme, "overflows"), progress);
    await saveFigure(root, figuresDir, "policy_scenario_heatmap_kgkm.png", buildPolicyScenarioHeatmap(rows, ctx, theme, "kgkm"), progress);
  }
  if (on("scenario_constructor_heatmap")) {
    const mode = opts.heatmapLabels ?? "both";
    if (mode === "both" || mode === "show") {
      await saveFigure(root, figuresDir, "scenario_constructor_heatmap.png", buildScenarioConstructorHeatmap(dfm, ctx, theme, true), progress);
    }
    if (mode === "both" || mode === "hide") {
      await saveFigure(root, figuresDir, "scenario_constructor_heatmap_full.png", buildScenarioConstructorHeatmap(dfm, ctx, theme, false), progress);
    }
    if (ctx.scenarios.some((s) => s.dist.toLowerCase() === "empirical")) {
      await saveFigure(
        root, figuresDir, "scenario_constructor_heatmap_empirical.png",
        buildScenarioConstructorHeatmap(dfm, ctx, theme, true, (s) => s.dist.toLowerCase() === "empirical"),
        progress
      );
    }
  }
  if (on("strategy_bubble")) {
    const sc = scales("strategy_bubble", "x_scale");
    await saveFigure(root, figuresDir, "strategy_bubble.png", buildStrategyBubble(dfm, ctx, theme, sc[0] as never), progress);
    if (sc.length > 1) {
      flags.bubbleLog = await saveFigure(
        root, figuresDir, "strategy_bubble_log.png", buildStrategyBubble(dfm, ctx, theme, sc[1] as never), progress
      );
    }
  }
  if (on("improver_bubble")) {
    const sc = scales("improver_bubble", "x_scale");
    await saveFigure(root, figuresDir, "improver_bubble.png", buildImproverBubble(dfm, ctx, theme, sc[0] as never), progress);
    if (sc.length > 1) {
      await saveFigure(root, figuresDir, "improver_bubble_log.png", buildImproverBubble(dfm, ctx, theme, sc[1] as never), progress);
    }
  }
  if (on("constructor_ranking")) {
    await saveFigure(root, figuresDir, "constructor_ranking.png", buildConstructorRanking(dfm, ctx, theme), progress);
  }
  if (on("radar")) {
    const curated = (charts.radar?.constructors as string[] | undefined)?.filter((c) =>
      ctx.constructors.includes(c)
    );
    const key = curated?.length ? curated : ctx.constructors.slice(0, 4);
    await saveFigure(
      root, figuresDir, "policy_radar.png",
      buildRadar(dfm, key, "Policy Performance Radar (normalised; outer = better)"),
      progress
    );
  }
  if (on("radar_combined")) {
    await saveFigure(
      root, figuresDir, "policy_radar_combined.png",
      buildRadar(dfm, ctx.constructors, "Policy Performance Radar — All Constructors (normalised; outer = better)"),
      progress
    );
  }
  if (on("improver_delta") && ctx.improvers.length > 1) {
    flags.improverDelta = await saveFigure(root, figuresDir, "improver_delta.png", buildImproverDelta(dfm, ctx, theme), progress);
  }
  return flags;
}

export async function renderBinMaps(root: string, figuresDir: string, theme: GenTheme, progress: Progress): Promise<void> {
  for (const [city, outName] of [
    ["Rio Maior", "riomaior_map.png"],
    ["Figueira da Foz", "figueiradafoz_map.png"],
  ] as const) {
    const coords = await loadBinCoords(root, city);
    if (!coords) {
      progress(`[WARN] Bin coordinates unavailable for ${city}`);
      continue;
    }
    await saveFigure(root, figuresDir, outName, buildBinMapChart(coords, `${city} — ${coords.lat.length} Waste Bins`, theme), progress);
  }
  for (const spec of SELECTED_MAP_SPECS) {
    const coords = await loadSelectedBinCoords(root, spec.csvName);
    if (coords) {
      await saveFigure(
        root, figuresDir, spec.outName,
        buildBinMapChart(coords, `${spec.city} — ${spec.nBins} Selected Bins`, theme),
        progress
      );
    } else {
      const all = await loadBinCoords(root, spec.city);
      if (all) {
        await saveFigure(
          root, figuresDir, spec.outName,
          buildBinMapChart(all, `${spec.city} — ${all.lat.length} Waste Bins`, theme),
          progress
        );
        progress(`[WARN] No selected-bin coords for ${spec.city} N=${spec.nBins}; used all-bins map`);
      }
    }
  }
}
