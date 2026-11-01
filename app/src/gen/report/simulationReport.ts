/**
 * Simulation analysis report generator (§H.5) — native port of the
 * orchestration + Jinja template of `logic/gen/gen_simulation_analysis.py`.
 *
 * Loads horizon CSVs (or parses raw output trees), renders every §H.2 chart
 * to PNG figures, builds all tables, renders the markdown template, and
 * writes report + figures through the Rust file bridge.
 */
import { SIM_CFG, loadTheme, type HorizonSpec } from "../config";
import { joinPath, pathExists, writeTextFile } from "../io";
import {
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
import {
  saveFigure,
  renderHorizonFigures,
  renderBinMaps,
  type HorizonRender,
} from "./simulationReportFigures";
import {
  horizonSection,
  comparisonSection,
  type HorizonTemplateCtx,
} from "./simulationReportTemplate";

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
    `${ctx0.strategies.length} packing algorithms × ${ctx0.improvers.length} route improvers × ` +
    `${ctx0.constructors.length} routing algorithms`;

  const header = `# WSmart Route+ — Simulation Analysis Report

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
| **Packing algorithm** | ${ctx0.strategies.join(", ")} |
| **Routing algorithms** | ${ctx0.constructors.join(", ")} |
${acceptance.length ? `| **Acceptance criteria** | ${acceptance.join(", ")} |\n` : ""}| **Route improvers** | ${ctx0.improvers.join(", ")} |
| **Simulation horizons (days)** | ${horizons.map((h) => h.days).join(", ")} |

### Policy Naming Convention

Each log file encodes the full pipeline as:
\`{packing}_{routing_algorithm}[_{acceptance_criterion}]_{route_improver}\`

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
