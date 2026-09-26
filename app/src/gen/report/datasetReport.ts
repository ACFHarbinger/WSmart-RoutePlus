/**
 * Dataset analysis report generator (§H.5) — native port of the
 * orchestration + Jinja template of `logic/gen/gen_dataset_analysis.py`.
 */
import { DATASET_CFG, loadTheme } from "../config";
import { joinPath, pathExists, writeBinaryFile, writeTextFile } from "../io";
import { renderChartPng, type ChartSpec } from "../charts/common";
import {
  buildNpzStatsBar,
  buildNpzSizeScaling,
  buildNpzHorizonComparison,
  buildNpzCityComparison,
  buildNpzTdAlignment,
  buildTdStatsComparison,
  buildTdWasteDistributions,
  buildNpzViolin,
  buildNpzBox,
  buildNpzHistKde,
  buildNpzExtendedStats,
} from "../charts/dataset";
import {
  loadNpzStats,
  loadTdStats,
  loadRawWaste,
  buildExtendedRows,
  buildNpzTable,
  buildExtendedTable,
  buildTdTable,
  type NpzStatRow,
} from "../data/dataset";
import { finalizeMarkdown, toRel, PLACEHOLDER } from "./markdown";
import {
  buildCityNetworkComparisonInteractive,
  buildInteractiveHtmlPage,
  buildNpzStatsInteractive,
  buildWasteDistributionInteractive,
} from "./interactiveHtml";
import type { Progress } from "./simulationReport";

export interface DatasetReportOptions {
  projectRoot: string;
  theme?: "dark" | "light";
  npzCsv?: string;
  tdCsv?: string;
  npzDir?: string;
  outMd?: string;
  figuresDir?: string;
  privateDir?: string;
  force?: boolean;
  figuresOnly?: boolean;
  /** Write self-contained interactive HTML charts (default true). */
  interactive?: boolean;
}

export async function generateDatasetReport(
  opts: DatasetReportOptions,
  progress: Progress = () => {}
): Promise<{ outMd: string | null; figuresDir: string }> {
  const root = opts.projectRoot;
  const theme = loadTheme(opts.theme ?? DATASET_CFG.theme);
  const npzCsv = opts.npzCsv ?? DATASET_CFG.npz_csv;
  const tdCsv = opts.tdCsv ?? DATASET_CFG.td_csv;
  const npzDir = opts.npzDir ?? DATASET_CFG.npz_dir;
  const figuresDir = opts.figuresDir ?? DATASET_CFG.figures_dir;
  const privateDir = opts.privateDir ?? DATASET_CFG.private_dir;
  const wantInteractive = opts.interactive ?? true;
  const outMdRel = opts.outMd ?? DATASET_CFG.out_md;

  progress(`Loading NPZ stats: ${npzCsv}`);
  const npz = await loadNpzStats(root, npzCsv);
  if (!npz.length) throw new Error(`No NPZ statistics found in ${npzCsv}`);
  let td: Awaited<ReturnType<typeof loadTdStats>> = [];
  if (await pathExists(joinPath(root, tdCsv))) {
    progress(`Loading TD stats: ${tdCsv}`);
    td = await loadTdStats(root, tdCsv);
  }
  progress(`Loading raw NPZ waste matrices from ${npzDir} …`);
  const raw = await loadRawWaste(root, npzDir, npz, 30);
  progress(`Raw NPZ entries loaded: ${raw.size}`);
  const ext = buildExtendedRows(raw);

  const save = async (name: string, spec: ChartSpec | null) => {
    if (!spec) return false;
    const png = await renderChartPng(spec);
    await writeBinaryFile(joinPath(root, `${figuresDir}/${name}`), png);
    progress(`Saved: ${name}`);
    return true;
  };

  // figures
  const hasTd = td.length > 0;
  if (hasTd) {
    await save("td_stats_comparison.png", buildTdStatsComparison(td, theme));
    await save("td_waste_distributions.png", buildTdWasteDistributions(td, theme));
  }
  await save("npz_stats_bar.png", buildNpzStatsBar(npz, ext, theme));
  await save("npz_size_scaling.png", buildNpzSizeScaling(npz, theme));
  const hasMultipleHorizons = await save(
    "npz_horizon_comparison.png",
    buildNpzHorizonComparison(npz, theme)
  );
  const hasRaw = raw.size > 0;
  if (hasRaw) {
    await save("npz_violin.png", buildNpzViolin(raw, theme));
    await save("npz_box.png", buildNpzBox(raw, theme));
    await save("npz_hist_kde.png", buildNpzHistKde(raw, theme));
    await save("npz_extended_stats.png", buildNpzExtendedStats(ext, theme));
  }
  const cities = [...new Set(npz.map((r) => r.city))].sort();
  const hasCityCmp = cities.length > 1;
  if (hasCityCmp) await save("npz_city_comparison.png", buildNpzCityComparison(npz, ext, theme));
  if (hasTd) await save("npz_td_alignment.png", buildNpzTdAlignment(npz, td, theme));

  if (wantInteractive) {
    progress(`Generating interactive HTML → ${privateDir}`);
    const pages: [string, ReturnType<typeof buildNpzStatsInteractive>][] = [
      ["npz_stats_interactive.html", buildNpzStatsInteractive(npz, theme)],
      ["waste_distribution_interactive.html", buildWasteDistributionInteractive(npz, theme)],
      ["city_network_comparison_interactive.html", buildCityNetworkComparisonInteractive(npz, theme)],
    ];
    for (const [name, page] of pages) {
      await writeTextFile(joinPath(root, `${privateDir}/${name}`), buildInteractiveHtmlPage(page, theme));
      progress(`Saved: ${name}`);
    }
  }

  if (opts.figuresOnly) {
    progress("figures-only: skipping markdown generation");
    return { outMd: null, figuresDir };
  }
  const outMdAbs = joinPath(root, outMdRel);
  if (!opts.force && (await pathExists(outMdAbs))) {
    progress(`${outMdRel} already exists — enable force to regenerate. Markdown skipped.`);
    return { outMd: null, figuresDir };
  }

  // markdown (ports jinja/dataset_analysis.md.j2)
  const f = toRel(figuresDir);
  const cityLabels = cities.map((c) => DATASET_CFG.city_labels[c] ?? c);
  const distLabels = [...new Set(npz.map((r) => r.dist))].sort().map((d) => DATASET_CFG.dist_labels[d] ?? d);
  const horizons = [...new Set(npz.map((r) => r.horizon))].sort((a, b) => a - b);
  const horizonStr = horizons.map((h) => `${h} days`).join(", ");

  let section = hasTd ? 2 : 1;
  const tocItems: string[] = [];
  if (hasTd) tocItems.push("1. [Training Data (TD)](#1-training-data-td)");

  const citySections: string[] = [];
  for (const city of cities) {
    const label = DATASET_CFG.city_labels[city] ?? city;
    const anchor = `${section}-${label.toLowerCase().replace(/ /g, "-")}-npz-datasets`;
    tocItems.push(`${section}. [${label} NPZ Datasets](#${anchor})`);
    const ns = [...new Set(npz.filter((r) => r.city === city).map((r) => r.N))].sort((a, b) => a - b);
    citySections.push(`
## ${section}. ${label} NPZ Datasets

**Network sizes:** N = ${ns.join(", ")}  **Distributions:** ${distLabels.join(", ")}  **Horizons:** ${horizonStr}

![NPZ Statistics Bar Chart](${f}/npz_stats_bar.png)

*Mean, median, std and max waste per city and distribution (30-day horizon).*

![Statistics vs Network Size](${f}/npz_size_scaling.png)

*How mean waste, std, and skewness vary with network size — Rio Maior N=20…170 (lines) plus the Figueira da Foz N=350 reference point (diamonds).*
${hasMultipleHorizons ? `
![Horizon Comparison](${f}/npz_horizon_comparison.png)

*Comparison of horizon statistics across network sizes, including the N=350 reference point.*
` : ""}
### Statistics Summary — ${label} (30-day horizon)

_TABCAP_: NPZ dataset statistics for ${label} — mean, median, std, max waste and IQR per network size and distribution (30-day horizon).

${buildNpzTable(npz.filter((r: NpzStatRow) => r.city === city), ext.filter((e) => e.city === city))}

${PLACEHOLDER}
`);
    section++;
  }

  const shapesSection = section++;
  tocItems.push(`${shapesSection}. [Waste Distribution Shapes](#${shapesSection}-waste-distribution-shapes)`);
  let cityCmpSection = 0;
  if (hasCityCmp) {
    cityCmpSection = section++;
    tocItems.push(`${cityCmpSection}. [City Comparison](#${cityCmpSection}-city-comparison)`);
  }
  let alignmentSection = 0;
  if (hasTd) {
    alignmentSection = section++;
    tocItems.push(`${alignmentSection}. [TD vs NPZ Alignment](#${alignmentSection}-td-vs-npz-alignment)`);
  }

  const md = finalizeMarkdown(`# WSmart+ Route — Dataset Analysis Report

> **Scope:** NPZ simulator datasets and TensorDict training datasets
> **Cities:** ${cityLabels.join(", ")}
> **Distributions:** ${distLabels.join(", ")}
> **Horizons analysed:** ${horizonStr}
> **Total NPZ dataset entries:** ${npz.length}
> **Generated:** ${new Date().toISOString().slice(0, 10)}

---

## Table of Contents

${tocItems.join("\n")}

---
${hasTd ? `
## 1. Training Data (TD)

Training data used for supervised learning models (stored as TensorDict \`.td\` files).
Each entry contains normalised waste values in [0, 1] (divide by 100 to convert to kg/kg).

![Waste Statistics Comparison](${f}/td_stats_comparison.png)

*Mean, std, and skewness of training waste values per network size and distribution.*

![Training Data Waste Distributions](${f}/td_waste_distributions.png)

*Bar chart of mean and std waste fractions per network size.*

### TD Statistics Summary

_TABCAP_: Training data (TD) statistics — mean, std, and skewness of normalised waste values per network size and distribution.

${buildTdTable(td)}

${PLACEHOLDER}

---
` : ""}${citySections.join("")}
---

## ${shapesSection}. Waste Distribution Shapes

![Waste Distribution Violin Plots](${f}/npz_violin.png)

*Violin plots of raw daily waste values (kg/bin/day) per network size and distribution, with embedded quartile markers.*

![Waste Distribution Box Plots](${f}/npz_box.png)

*Box plots showing median, quartiles, interquartile range, outlier fences and outliers of raw waste values.*

![Waste Histograms with KDE](${f}/npz_hist_kde.png)

*Histograms with kernel density estimates of raw waste values per distribution — reveals modes and tail behaviour.*

![Extended Statistics](${f}/npz_extended_stats.png)

*Median, variance, interquartile range, minimum, outlier fences and mode per network size and distribution.*

### Extended Statistics Summary (30-day horizon)

_TABCAP_: Extended NPZ dataset statistics — median, variance, IQR, minimum, outlier fences and mode of raw waste values per city, network size and distribution (30-day horizon).

${buildExtendedTable(ext)}

${PLACEHOLDER}

---
${hasCityCmp ? `
## ${cityCmpSection}. City Comparison

![City Comparison Overview](${f}/npz_city_comparison.png)

*Key statistics across cities and distributions.*

### Statistics Summary — All Cities (30-day horizon)

_TABCAP_: NPZ dataset statistics across all cities — mean, median, std, max waste and IQR per city and distribution (30-day horizon).

${buildNpzTable(npz, ext)}

${PLACEHOLDER}

---
` : ""}${hasTd ? `
## ${alignmentSection}. TD vs NPZ Alignment

![Training (TD) vs Simulator (NPZ) Mean Waste Alignment](${f}/npz_td_alignment.png)

*Comparison of mean waste levels between TD training data (normalised × 100) and NPZ simulator
data, including the Figueira da Foz N=350 reference point. Close alignment validates that
training distribution matches simulation.*

${PLACEHOLDER}

---
` : ""}
*Figures are stored in \`${f}/\`.*
*Raw statistics: \`${tdCsv}\` and \`${npzCsv}\`.*
${wantInteractive ? `
## Interactive Charts

- [NPZ Statistics — Mean vs Std Scatter](${toRel(privateDir)}/npz_stats_interactive.html)
- [Waste Distribution by City and Network Size](${toRel(privateDir)}/waste_distribution_interactive.html)
- [City & Network Comparison](${toRel(privateDir)}/city_network_comparison_interactive.html)
` : ""}`);

  await writeTextFile(outMdAbs, md);
  progress(`Written: ${outMdRel} (${md.length} chars)`);
  return { outMd: outMdRel, figuresDir };
}
