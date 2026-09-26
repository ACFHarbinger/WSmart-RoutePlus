/**
 * Simulation analysis report generator (§H.5) — native port of the
 * orchestration + Jinja template of `logic/gen/gen_simulation_analysis.py`.
 *
 * Loads horizon CSVs (or parses raw output trees), renders every §H.2 chart
 * to PNG figures, builds all tables, renders the markdown template, and
 * writes report + figures through the Rust file bridge.
 */
import { SIM_CFG, loadTheme, type GenTheme, type HorizonSpec } from "../config";
import { joinPath, pathExists, writeBinaryFile, writeTextFile } from "../io";
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
  buildHorizonComparison,
  buildHorizonDelta,
  buildHorizonConstructorRanking,
} from "../charts/simulation";
import {
  aggregate,
  buildCtx,
  buildFullResultsTableAllHorizons,
  buildKpiTable,
  buildParetoFrontTable,
  buildStrategyBest,
  buildFullResultsMatrix,
  renderFullResultsTableMd,
  filterData,
  loadHorizonCsv,
  parseOutputDir,
  regionLabel,
  scenarioLabel,
  rowsToCsv,
  KGKM_LABEL,
  type AnalysisCtx,
  type HorizonData,
  type SimRow,
} from "../data/simulation";
import { finalizeMarkdown, toRel, PLACEHOLDER } from "./markdown";
import {
  buildInteractiveHtmlPage,
  buildParetoInteractive,
  buildPolicyHeatmapInteractive,
  buildStrategyBubbleInteractive,
} from "./interactiveHtml";
import { buildBinMapChart, loadBinCoords, loadSelectedBinCoords, SELECTED_MAP_SPECS } from "./binMaps";

export interface SimReportOptions {
  projectRoot: string;
  theme?: "dark" | "light";
  paretoPoints?: "all" | "front";
  horizons?: HorizonSpec[];
  scenarios?: { city: string; N: number; dist: string }[] | null;
  strategies?: string[] | null;
  constructors?: string[] | null;
  improvers?: string[] | null;
  acceptance?: string[] | null;
  outMd?: string;
  figuresDir?: string;
  privateDir?: string;
  force?: boolean;
  figuresOnly?: boolean;
  heatmapLabels?: "both" | "show" | "hide";
  /** Write self-contained interactive HTML charts (default: config charts.interactive.enabled). */
  interactive?: boolean;
}

export type Progress = (message: string) => void;

async function saveFigure(
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

interface HorizonRender {
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

async function renderHorizonFigures(
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

async function renderBinMaps(root: string, figuresDir: string, theme: GenTheme, progress: Progress): Promise<void> {
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
      // fallback: re-render the all-bins map under the selected-map name
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

// ── Template (ports jinja/simulation_analysis.md.j2) ─────────────────────────

interface HorizonTemplateCtx {
  days: number;
  sectionNum: number;
  nLogs: number;
  constructors: string[];
  figuresRel: string;
  privateRel: string;
  interactive: boolean;
  paretoTable: string;
  overflowTable: string;
  efficiencyTable: string;
  kmTable: string;
  strategyBest: string;
  fullResultsTable: string;
  flags: HorizonRender["flags"];
}

function horizonSection(h: HorizonTemplateCtx, improvers: string[], strategies: string[]): string {
  const n = h.sectionNum;
  const f = h.figuresRel;
  return `
## ${n}. ${h.days}-Day Horizon Results

> **Logs analysed:** ${h.nLogs}
> **Constructors available:** ${h.constructors.join(", ")}

### ${n}.1 Analytics Comparison — Pareto View

![Overflow vs Efficiency — Pareto Front](${f}/pareto_scatter.png)

*Scatter of all ${h.days}-day runs in the overflows–kg/km space, one panel per waste distribution. Colour encodes the mandatory selection variant, marker shape encodes the scenario (region/N), filled markers = ${improvers[0]}, open markers = ${improvers[improvers.length - 1]}. Dashed lines = Pareto fronts, one colour per scenario (region × N × distribution).*
${h.flags.paretoLog ? `
![Overflow vs Efficiency — Pareto Front (log scale)](${f}/pareto_scatter_log.png)

*Same chart with symlog X-axis — spreads the densely clustered low-overflow region.*
` : ""}${h.interactive ? `
**[Interactive version](${h.privateRel}/pareto_scatter_interactive.html)**
` : ""}
#### Pareto-Front Policy Catalogue (${h.days} days)

_TABCAP_: Pareto-optimal policy configurations over the ${h.days}-day horizon — each unique (selection variant, constructor, improver) that appeared on the Pareto front of at least one scenario, sorted by scenario count; metrics averaged across those scenarios.

${h.paretoTable}

${PLACEHOLDER}

### ${n}.2 Summary KPI Analysis

#### Overflow Performance

![Overflow Count by Configuration](${f}/overflow_by_config.png)

*Mean overflow count per scenario and selection strategy (mean ± min/max range across route constructors); route improvers shown as paired bars within each configuration.*
${h.flags.overflowLog ? `
![Overflow Count by Configuration (log scale)](${f}/overflow_by_config_log.png)

*Same chart with symlog Y axis — reveals structure compressed in the linear scale.*
` : ""}
_TABCAP_: Overflow counts by configuration over ${h.days} days — min/max/mean across route constructors, per route improver.

${h.overflowTable}

${PLACEHOLDER}

#### Route Efficiency (kg/km)

![kg/km Efficiency by Configuration](${f}/kgkm_by_config.png)

*Mean kg/km efficiency per scenario and selection strategy, with min–max whiskers across constructors; improvers as paired bars.*

_TABCAP_: Route efficiency (kg/km) by configuration over ${h.days} days — min/max/mean across route constructors, per route improver.

${h.efficiencyTable}

${PLACEHOLDER}

#### Distance Driven (km)

![Vehicle Distance by Strategy](${f}/km_violin.png)

*Distribution of total vehicle distance (km) per selection strategy and scenario (all constructors and improvers pooled), one panel per waste distribution.*

_TABCAP_: Vehicle distance driven (km) by configuration over ${h.days} days — min/max/mean across route constructors, per route improver.

${h.kmTable}

${PLACEHOLDER}

### ${n}.3 Policy × Scenario Heatmaps

![Policy × Scenario Heatmap — Overflows](${f}/policy_scenario_heatmap_overflows.png)

*Overflow count heatmap: each row is a full policy configuration (selection variant + constructor + improver), each column a simulation scenario (region × N × distribution).*

![Policy × Scenario Heatmap — Efficiency](${f}/policy_scenario_heatmap_kgkm.png)

*kg/km efficiency heatmap with the same layout (rows = policy configurations, columns = scenarios).*

![Per-Scenario Constructor Heatmaps](${f}/scenario_constructor_heatmap.png)

*One panel per scenario: route constructors on the rows, selection strategy × route improver combinations on the columns.*
${h.interactive ? `
**[Interactive heatmap](${h.privateRel}/policy_heatmap_interactive.html)**
` : ""}
${PLACEHOLDER}

### ${n}.4 Selection Strategy Comparison (${strategies.join(" vs ")})

![Strategy Trade-off Bubble Chart](${f}/strategy_bubble.png)

*One panel per waste distribution. Each bubble = one (strategy, scenario) combination, averaged over constructors and improvers; bubble size ∝ N.*
${h.flags.bubbleLog ? `
![Strategy Trade-off Bubble Chart (log X scale)](${f}/strategy_bubble_log.png)

*Same chart with symlog X axis.*
` : ""}${h.interactive ? `
**[Interactive bubble chart](${h.privateRel}/strategy_bubble_interactive.html)**
` : ""}
${h.strategyBest}

${PLACEHOLDER}

### ${n}.5 Route Improver Comparison (${improvers.join(" vs ")})

![Improver Trade-off Bubble Chart](${f}/improver_bubble.png)

*Each bubble = one (improver, scenario) combination averaged over strategies and constructors — contrasts the route improvers directly.*
${h.flags.improverDelta ? `
![Improver Delta Heatmap](${f}/improver_delta.png)

*Delta heatmap (${improvers[improvers.length - 1]} − ${improvers[0]}) per constructor × configuration.*
` : ""}
${PLACEHOLDER}

### ${n}.6 Key Findings

![Policy Performance Radar](${f}/policy_radar.png)

*Overlaid radar chart for key constructors. Outer = better on all axes.*

![Policy Performance Radar — All Constructors](${f}/policy_radar_combined.png)

*Same normalised radar, overlaying every route constructor instead of the curated subset above.*

![Route Constructor Average Rank](${f}/constructor_ranking.png)

*Average rank of each route constructor across all scenarios and strategies (improvers pooled). Bars grow upward — shorter = better.*

${PLACEHOLDER}

### ${n}.7 Full Results Table

_TABCAP_: Full results over the ${h.days}-day horizon — rows: graph size × region × data distribution; columns: mandatory selection strategy × route constructor × route improver. Each cell reports mean±std overflows and mean±std kg/km.

${h.fullResultsTable}

${PLACEHOLDER}

---
`;
}

function comparisonSection(cmp: {
  sectionNum: number;
  label: string;
  figuresRel: string;
  firstDays: number;
  lastDays: number;
  fullResultsTable: string;
}): string {
  const f = cmp.figuresRel;
  return `
## ${cmp.sectionNum}. Horizon Comparison (${cmp.label})

This section compares results across the simulation horizons to identify which patterns are
robust across time scales and which shift as the evaluation window extends.

### Overflow Across Horizons

![Overflow Horizon Comparison](${f}/horizon_overflow_comparison.png)

*Side-by-side overflow bars for every configuration, one bar colour per horizon.
Growth across horizons indicates that overflow pressure accumulates over time.*

![Overflow Relative Delta](${f}/horizon_overflow_delta.png)

*Relative change in mean overflows between the shortest and longest horizon: (${cmp.lastDays}d − ${cmp.firstDays}d) / ${cmp.firstDays}d × 100.
Red bars = more overflows on the longer horizon; green bars = fewer.*

${PLACEHOLDER}

### Efficiency Across Horizons

![kg/km Horizon Comparison](${f}/horizon_kgkm_comparison.png)

*Side-by-side kg/km efficiency comparison.
Consistent efficiency across horizons suggests the routing policy scales well.*

${PLACEHOLDER}

### Constructor Rankings Across Horizons

![Constructor Ranking Across Horizons](${f}/horizon_constructor_ranking.png)

*Average constructor rank (lower = better) compared across horizons.
Constructors with stable ranks are robust; those that improve or regress warrant deeper investigation.*

${PLACEHOLDER}

### Key Observations

${PLACEHOLDER}

### Full Results Table — All Horizons

_TABCAP_: Full results across all simulation horizons — rows: graph size × region × data distribution; columns: horizon × mandatory selection strategy × route constructor × route improver. Each cell reports mean±std overflows and mean±std kg/km.

${cmp.fullResultsTable}

${PLACEHOLDER}

---
`;
}

// ── Orchestration (ports main) ───────────────────────────────────────────────

export async function generateSimulationReport(
  opts: SimReportOptions,
  progress: Progress = () => {}
): Promise<{ outMd: string | null; figuresDirs: string[] }> {
  const root = opts.projectRoot;
  const theme = loadTheme(opts.theme ?? SIM_CFG.theme);
  const filter = {
    scenarios: opts.scenarios ?? SIM_CFG.scenarios,
    policies: {
      strategies: opts.strategies ?? SIM_CFG.policies.strategies,
      constructors: opts.constructors ?? SIM_CFG.policies.constructors,
      improvers: opts.improvers ?? SIM_CFG.policies.improvers,
      acceptance: opts.acceptance ?? SIM_CFG.policies.acceptance,
    },
  };
  const paretoPoints = opts.paretoPoints ?? ((SIM_CFG.charts.pareto_scatter?.points as "all" | "front") ?? "all");
  const specs = opts.horizons?.length ? opts.horizons : SIM_CFG.horizons;
  const figuresBase = opts.figuresDir ?? SIM_CFG.figures_dir;
  const privateBase = opts.privateDir ?? SIM_CFG.private_dir;
  const wantInteractive =
    opts.interactive ?? ((SIM_CFG.charts.interactive?.enabled as boolean | undefined) ?? true);
  const outMdRel = opts.outMd ?? SIM_CFG.out_md;

  // load horizons
  const horizons: HorizonData[] = [];
  for (const spec of [...specs].sort((a, b) => a.days - b.days)) {
    let rows: SimRow[] | null = null;
    if (spec.csv && (await pathExists(joinPath(root, spec.csv)))) {
      progress(`Loading ${spec.days}d CSV: ${spec.csv}`);
      rows = await loadHorizonCsv(root, spec.csv);
    } else if (spec.output_dir) {
      progress(`Parsing ${spec.days}d output tree: ${spec.output_dir}`);
      rows = await parseOutputDir(joinPath(root, spec.output_dir));
    }
    if (!rows?.length) {
      progress(`[WARN] No data for the ${spec.days}d horizon — skipped`);
      continue;
    }
    const filtered = filterData(rows, filter);
    const ctx = buildCtx(filtered, spec.days, paretoPoints);
    horizons.push({ days: spec.days, rows: filtered, dfm: aggregate(filtered), ctx, nLogs: filtered.length });
  }
  if (!horizons.length) throw new Error("No horizon data could be loaded.");
  const multi = horizons.length > 1;

  // figures
  const renders: HorizonRender[] = [];
  const figuresDirs: string[] = [];
  for (let i = 0; i < horizons.length; i++) {
    const h = horizons[i];
    const figDir = multi ? `${figuresBase}/${h.days}d` : figuresBase;
    figuresDirs.push(figDir);
    progress(`Generating ${h.days}d figures → ${figDir}`);
    const flags = await renderHorizonFigures(h, figDir, theme, opts, progress);
    await renderBinMaps(root, figDir, theme, progress);
    const privDir = multi ? `${privateBase}/${h.days}d` : privateBase;
    if (wantInteractive) {
      progress(`Generating ${h.days}d interactive HTML → ${privDir}`);
      const pages: [string, ReturnType<typeof buildParetoInteractive>][] = [
        ["pareto_scatter_interactive.html", buildParetoInteractive(h.rows, h.ctx, theme)],
        ["strategy_bubble_interactive.html", buildStrategyBubbleInteractive(h.dfm, h.ctx, theme)],
        ["policy_heatmap_interactive.html", buildPolicyHeatmapInteractive(h.rows, h.ctx, theme)],
      ];
      for (const [name, page] of pages) {
        await writeTextFile(joinPath(root, `${privDir}/${name}`), buildInteractiveHtmlPage(page, theme));
        progress(`Saved: ${name}`);
      }
    }
    renders.push({
      data: h,
      csv: specs.find((s) => s.days === h.days)?.csv ?? "",
      figuresDir: figDir,
      privateDir: privDir,
      interactive: wantInteractive,
      flags,
    });
  }
  let cmpDir: string | null = null;
  if (multi) {
    cmpDir = `${figuresBase}/compare`;
    progress(`Generating horizon comparison figures → ${cmpDir}`);
    const cmpCtx: AnalysisCtx = horizons[horizons.length - 1].ctx;
    await saveFigure(root, cmpDir, "horizon_overflow_comparison.png", buildHorizonComparison(horizons, cmpCtx, theme, "overflows"), progress);
    await saveFigure(root, cmpDir, "horizon_kgkm_comparison.png", buildHorizonComparison(horizons, cmpCtx, theme, "kgkm"), progress);
    await saveFigure(root, cmpDir, "horizon_overflow_delta.png", buildHorizonDelta(horizons, cmpCtx, theme), progress);
    await saveFigure(root, cmpDir, "horizon_constructor_ranking.png", buildHorizonConstructorRanking(horizons, theme), progress);
  }

  if (opts.figuresOnly) {
    progress("figures-only: skipping markdown generation");
    return { outMd: null, figuresDirs };
  }
  const outMdAbs = joinPath(root, outMdRel);
  if (!opts.force && (await pathExists(outMdAbs))) {
    progress(`${outMdRel} already exists — enable force to regenerate. Markdown skipped.`);
    return { outMd: null, figuresDirs };
  }

  // markdown
  const ctx0 = horizons[0].ctx;
  const acceptance = [...new Set(horizons.flatMap((h) => h.rows.map((r) => r.acceptance)))].filter(Boolean).sort();
  const tocItems = ["1. [Experimental Setup](#1-experimental-setup)"];
  const hCtxs: HorizonTemplateCtx[] = renders.map((r, i) => {
    const n = i + 2;
    const anchor = `${n}-${r.data.days}-day-horizon-results`;
    tocItems.push(`${n}. [${r.data.days}-Day Horizon Results](#${anchor})`);
    return {
      days: r.data.days,
      sectionNum: n,
      nLogs: r.data.nLogs,
      constructors: r.data.ctx.constructors,
      figuresRel: toRel(r.figuresDir),
      privateRel: toRel(r.privateDir),
      interactive: r.interactive,
      paretoTable: buildParetoFrontTable(r.data.rows, r.data.ctx),
      overflowTable: buildKpiTable(r.data.dfm, r.data.ctx, "overflows", 1),
      efficiencyTable: buildKpiTable(r.data.dfm, r.data.ctx, "kgkm", 2),
      kmTable: buildKpiTable(r.data.dfm, r.data.ctx, "km", 0),
      strategyBest: buildStrategyBest(r.data.dfm, r.data.ctx, PLACEHOLDER),
      fullResultsTable: renderFullResultsTableMd(buildFullResultsMatrix(r.data.dfm, r.data.ctx)),
      flags: r.flags,
    };
  });

  let comparisonMd = "";
  if (multi && cmpDir) {
    const sec = horizons.length + 2;
    const label = horizons.map((h) => `${h.days}d`).join(" vs ");
    tocItems.push(
      `${sec}. [Horizon Comparison (${label})](#${sec}-horizon-comparison-${label.replace(/ /g, "-").toLowerCase()})`
    );
    comparisonMd = comparisonSection({
      sectionNum: sec,
      label,
      figuresRel: toRel(cmpDir),
      firstDays: horizons[0].days,
      lastDays: horizons[horizons.length - 1].days,
      fullResultsTable: buildFullResultsTableAllHorizons(horizons),
    });
  }

  const scope =
    `${horizons.map((h) => `${h.days}-day`).join(" and ")} simulation runs across ` +
    `${ctx0.regions.length} region/network configurations × ${ctx0.dists.length} distributions × ` +
    `${ctx0.strategies.length} selection strategies × ${ctx0.improvers.length} route improvers × ` +
    `${ctx0.constructors.length} route constructors`;

  const header = `# WSmart+ Route — Simulation Analysis Report

> **Scope:** ${scope}
${horizons.map((h) => `> **Total logs analysed (${h.days}d):** ${h.nLogs}`).join("\n")}
> **Scenarios:** ${ctx0.scenarios.map(scenarioLabel).join(", ")}
> **Generated:** ${new Date().toISOString().slice(0, 10)}

---

## Table of Contents

${tocItems.join("\n")}

---

## 1. Experimental Setup

### Configuration Space

_TABCAP_: Configuration space — experimental dimensions and the values tested in this study.

| Dimension | Values |
|-----------|--------|
| **Scenarios (region / N)** | ${ctx0.regions.map(([c, n]) => `${regionLabel(c, n)} (N=${n})`).join(", ")} |
| **Waste distribution** | ${ctx0.dists.join(", ")} |
| **Mandatory selection strategy** | ${ctx0.strategies.join(", ")} |
| **Route constructors** | ${ctx0.constructors.join(", ")} |
${acceptance.length ? `| **Acceptance criteria** | ${acceptance.join(", ")} |\n` : ""}| **Route improvers** | ${ctx0.improvers.join(", ")} |
| **Simulation horizons (days)** | ${horizons.map((h) => h.days).join(", ")} |

### Policy Naming Convention

Each log file encodes the full pipeline as:
\`{mandatory_selection}_{route_constructor}[_{acceptance_criterion}]_{route_improver}\`

For Last-Minute (LM), two critical fill threshold variants are tested: **CF70** (70% fill triggers mandatory collection) and **CF90** (90% threshold). Service-Level (SL) tests two service level targets: **SL1** and **SL2**. Results in this report aggregate CF70 and CF90 under **LM**, and SL1/SL2 under **SL**, unless otherwise specified.

### Metrics Tracked

_TABCAP_: Metrics tracked per simulation run, their optimisation direction, and a brief description.

| Metric | Direction | Description |
|--------|-----------|-------------|
| \`overflows\` | ↓ lower better | Bins exceeding 100% capacity during simulation |
| \`kg\` | ↑ higher better | Total waste collected (kg) over the simulation horizon |
| \`km\` | ↓ lower better | Total vehicle distance driven (km) |
| \`kg/km\` | ↑ higher better | Route efficiency (waste per unit distance) |
| \`ncol\` | contextual | Number of collection events |
| \`kg_lost\` | ↓ lower better | Waste that overflowed and was not collected |
| \`profit\` | ↑ higher better | Revenue from collection minus operational cost |
| \`days\` | contextual | Active collection days in the simulation horizon |

---
`;

  const interactiveFooter = renders.some((r) => r.interactive)
    ? "\n## Interactive Charts\n" +
      renders
        .filter((r) => r.interactive)
        .map(
          (r) =>
            `\n### ${r.data.days}-Day Horizon\n\n` +
            `- [Overflow vs Efficiency — Pareto View](${toRel(r.privateDir)}/pareto_scatter_interactive.html)\n` +
            `- [Strategy Trade-off Bubble Chart](${toRel(r.privateDir)}/strategy_bubble_interactive.html)\n` +
            `- [Policy Configuration Heatmap](${toRel(r.privateDir)}/policy_heatmap_interactive.html)\n`
        )
        .join("") +
      "\n"
    : "";

  const footer = `
*Figures are stored under \`${toRel(figuresBase)}/\`.*
*Raw simulation data: ${renders.filter((r) => r.csv).map((r) => `\`${r.csv}\``).join(", ")}.*
${interactiveFooter}`;

  const md = finalizeMarkdown(
    header +
      hCtxs.map((h) => horizonSection(h, ctx0.improvers, ctx0.strategies)).join("") +
      comparisonMd +
      footer
  );
  await writeTextFile(outMdAbs, md);
  progress(`Written: ${outMdRel} (${md.length} chars)`);
  return { outMd: outMdRel, figuresDirs };
}

/** Native --parse-output mode: raw output tree → summary CSV. */
export async function parseOutputToCsv(
  projectRoot: string,
  outputDir: string,
  outCsv: string,
  progress: Progress = () => {}
): Promise<number> {
  progress(`Parsing: ${outputDir}`);
  const rows = await parseOutputDir(joinPath(projectRoot, outputDir));
  if (!rows.length) throw new Error("No log files found — check the output directory structure.");
  progress(`Rows: ${rows.length}`);
  for (const col of ["city", "dist", "strategy", "improver", "constructor"] as const) {
    progress(`${col[0].toUpperCase()}${col.slice(1)}s: ${[...new Set(rows.map((r) => String(r[col])))].sort().join(", ")}`);
  }
  await writeTextFile(joinPath(projectRoot, outCsv), rowsToCsv(rows));
  progress(`Written: ${outCsv}`);
  return rows.length;
}
