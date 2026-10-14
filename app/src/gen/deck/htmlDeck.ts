/**
 * Self-contained HTML deck export (§H.6) — a keyboard-navigable slideshow
 * mirroring the 21-slide PPTX structure, in one offline file: figures,
 * logos, icons and equations embed as data URLs; the native-shape diagrams
 * (pipeline chevrons, policy grid, taxonomy, DoE tree, objective flow,
 * simulator scene) are reproduced as HTML/CSS; the results table renders as
 * a real HTML table with merged header spans and best-cell highlighting.
 *
 * Navigation: ←/→/Space/PageUp/PageDown, Home/End, `N` toggles speaker
 * notes, slide counter bottom-right, location hash tracks the slide index.
 */
import { CONTENT, type ContentSlide } from "../config";
import { GEN_IMAGES, assetToDataUrl } from "../assets";
import { joinPath, pathExists, readBinaryDataUrl, writeTextFile } from "../io";
import { renderEquationPng } from "./equations";
import { resolveIllustration, type ImageMode } from "./illustrations";
import {
  computeGlobalBest,
  loadResultsTable,
  splitMatrix,
  type ResultsTableMode,
  type ResultsTableSplit,
} from "./resultsTable";
import { cellKey, groupSpans, type ResultsMatrix, type RowKey } from "../data/simulation";
import type { Progress } from "../report/simulationReport";

const ACCENT = "#2E74B5";
const DARK = "#1F2D3D";
const MUTED = "#5A6A7A";
const LIGHT_TXT = "#C9D6E4";
const GREEN = "#3E8E41";
const ORANGE = "#B06A2E";
const LIGHT_FILL = "#F0F4FA";

export interface HtmlDeckOptions {
  projectRoot: string;
  figuresDir: string;
  /** Output path for the .html file. */
  out: string;
  author?: string;
  coauthors?: string[];
  groups?: string[];
  resultsTable?: ResultsTableMode;
  resultsTableSplit?: ResultsTableSplit;
  imageMode?: ImageMode;
}

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Deck stylesheet — shared by the slideshow page and the PDF rasteriser (§H.6). */
export const DECK_CSS = `  html, body { margin: 0; height: 100%; background: #0c0c18; font-family: Helvetica, Arial, sans-serif; }
  .frame { display: none; width: 100vw; height: 100vh; align-items: center; justify-content: center; }
  .frame.active { display: flex; }
  .slide { position: relative; width: min(100vw, 177.78vh); aspect-ratio: 16 / 9; background: #ffffff; overflow: hidden; display: flex; flex-direction: column; }
  .slide.dark { background: ${DARK}; color: #ffffff; }
  .titlebar { background: ${DARK}; color: #fff; font-size: 1.7em; font-weight: bold; padding: 0.55em 0.8em; flex: none; }
  .body { flex: 1; padding: 0.8em 1.1em; overflow: hidden; display: flex; flex-direction: column; gap: 0.5em; color: ${DARK}; min-height: 0; }
  .bullets { margin: 0.2em 0; padding-left: 1.2em; color: ${DARK}; font-size: 1.05em; }
  .bullets.small { font-size: 0.9em; }
  .bullets li { margin-bottom: 0.45em; }
  .figrow { flex: 1; display: flex; gap: 0.6em; align-items: center; justify-content: center; min-height: 0; }
  .figrow img { max-width: 100%; max-height: 100%; min-height: 0; object-fit: contain; flex: 1 1 0; }
  .split { flex: 1; display: flex; gap: 1em; min-height: 0; }
  .split > div { flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; min-height: 0; }
  .eqband { background: ${LIGHT_FILL}; border-radius: 10px; padding: 0.7em; display: flex; flex-direction: column; align-items: center; gap: 0.45em; }
  .eqband img.eq { max-width: 100%; object-fit: contain; }
  .eq-fallback { font-family: 'Cambria Math', serif; color: ${DARK}; }
  .caption { font-size: 0.75em; font-style: italic; color: ${MUTED}; margin: 0.2em 0 0; flex: none; }
  /* cover */
  .cover { justify-content: center; padding: 0 6%; }
  .cover h1 { font-size: 2.1em; margin: 0 0 0.5em; }
  .cover .band { height: 5px; background: ${ACCENT}; margin: 0.4em 0 0.9em; }
  .cover .author { font-size: 1.25em; font-weight: bold; margin: 0.2em 0; }
  .cover .coauthors { color: ${LIGHT_TXT}; margin: 0.2em 0; }
  .cover .groups { color: #8A9BB0; font-style: italic; font-size: 0.8em; }
  .cover .logos { display: flex; gap: 8%; align-items: center; margin-top: 1.2em; }
  .cover .logos img { height: 3.2em; object-fit: contain; }
  .cover .conf-logo { position: absolute; top: 2%; right: 2%; height: 4.2em; }
  /* agenda */
  .agenda-title { font-size: 2em; margin: 0.8em 1em 0.4em; }
  .agenda { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6em; padding: 0 1.6em 1.6em; flex: 1; }
  .agenda-card { display: flex; align-items: center; gap: 0.7em; background: #273749; border: 2px solid; border-radius: 10px; padding: 0.5em 0.8em; font-weight: bold; font-size: 1.05em; }
  .agenda-card .badge { flex: none; width: 1.9em; height: 1.9em; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.05em; }
  /* pipeline */
  .pipeline { display: flex; gap: 0.5em; flex: none; }
  .stage-wrap { flex: 1; text-align: center; }
  .chevron { background: ${ACCENT}; color: #fff; font-weight: bold; padding: 0.9em 0.4em; clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 50%, calc(100% - 14px) 100%, 0 100%, 14px 50%); font-size: 0.9em; }
  .chevron.optional { background: #8A9BB0; }
  .stage-cap { color: ${MUTED}; font-size: 0.75em; margin-top: 0.4em; }
  /* policy grid + taxonomy */
  .policy-grid, .taxonomy { display: flex; gap: 0.8em; flex: 1; min-height: 0; }
  .pg-col, .tax-col { flex: 1; display: flex; flex-direction: column; gap: 0.4em; }
  .pg-head { background: ${DARK}; color: #fff; font-weight: bold; text-align: center; border-radius: 8px; padding: 0.45em; }
  .pg-item { background: ${ACCENT}; color: #fff; font-weight: bold; text-align: center; border-radius: 8px; padding: 0.55em 0.3em; flex: 1; display: flex; align-items: center; justify-content: center; font-size: 0.85em; }
  .pg-group { border: 1.5px dashed ${MUTED}; border-radius: 6px; padding: 0.4em; flex: 1; }
  .pg-group-label { color: ${ACCENT}; font-weight: bold; text-align: center; font-size: 0.85em; }
  .pg-group-items { text-align: center; font-size: 0.8em; color: ${DARK}; margin-top: 0.25em; }
  .tax-item { background: ${LIGHT_FILL}; border: 1.5px solid; border-radius: 8px; color: ${DARK}; font-weight: bold; text-align: center; font-size: 0.78em; padding: 0.5em 0.3em; flex: 1; display: flex; align-items: center; justify-content: center; }
  /* DoE tree */
  .doe { text-align: center; flex: none; }
  .doe-root { display: inline-block; background: ${DARK}; color: #fff; font-weight: bold; border-radius: 8px; padding: 0.4em 1.2em; margin-bottom: 0.6em; }
  .doe-row { display: flex; gap: 1em; }
  .doe-h { flex: 1; }
  .doe-hbox { background: ${ACCENT}; color: #fff; font-weight: bold; border-radius: 8px; padding: 0.4em; font-size: 0.85em; }
  .doe-scen { display: flex; gap: 0.4em; margin-top: 0.5em; }
  .doe-s { flex: 1; }
  .doe-sbox { background: #8A9BB0; color: #fff; font-weight: bold; padding: 0.3em; font-size: 0.75em; }
  .doe-d { font-size: 0.65em; color: ${DARK}; margin-top: 0.25em; }
  /* simulator scene */
  .sim-scene { position: relative; flex: 1; min-height: 0; }
  .sim-scene .bin { position: absolute; width: 4%; text-align: center; }
  .sim-scene .bin img { width: 100%; }
  .sim-scene .bin span { position: absolute; left: 0; right: 0; bottom: 18%; font-size: 0.6em; font-weight: bold; color: ${DARK}; }
  .sim-scene .bin.selected { outline: 3px solid red; }
  .sim-scene .truck { position: absolute; left: 72%; top: 78%; width: 10%; }
  .sim-scene .brace { position: absolute; left: 58%; top: 20%; font-size: 7em; color: ${MUTED}; }
  /* objective flow */
  .obj-flow { display: flex; align-items: center; gap: 0.7em; flex: none; margin-bottom: 0.6em; }
  .obj-inputs { display: flex; flex-direction: column; gap: 0.4em; }
  .obj-box { color: #fff; font-weight: bold; border-radius: 8px; padding: 0.5em 0.9em; text-align: center; font-size: 0.85em; }
  .obj-sim { background: ${DARK}; color: #fff; font-weight: bold; border-radius: 10px; padding: 0.9em 1.1em; text-align: center; }
  .obj-sim small { display: block; color: ${LIGHT_TXT}; font-weight: normal; font-size: 0.7em; margin-top: 0.4em; }
  .obj-arrow { font-size: 1.6em; color: #8A9BB0; font-weight: bold; }
  /* results table */
  .results { border-collapse: collapse; font-size: 0.52em; margin: auto; max-width: 100%; }
  .results th, .results td { border: 1px solid #C8D0DA; padding: 0.25em 0.4em; text-align: center; }
  .results th { background: ${DARK}; color: #fff; }
  .results td.rowlabel { font-weight: bold; }
  .results td.best { background: #C6EFCE; color: #1F6B2C; font-weight: bold; }
  .results tbody tr:nth-child(even) td:not(.best) { background: #EEF2F7; }
  /* statement slides */
  .statement { align-items: center; justify-content: center; text-align: center; padding: 0 8%; }
  .statement h2 { font-size: 1.9em; }
  .statement p { color: ${LIGHT_TXT}; font-size: 1.05em; }
  .statement img.fct { height: 4.5em; margin-top: 2em; }
  .qa { flex-direction: row; gap: 4%; text-align: left; }
  .qa .qa-fig { max-height: 80%; max-width: 45%; object-fit: contain; }
  /* chrome */
  #counter { position: fixed; bottom: 10px; right: 14px; color: #8A9BB0; font-size: 13px; z-index: 10; }
  #notes-panel { position: fixed; left: 0; right: 0; bottom: 0; max-height: 30vh; overflow: auto; background: rgba(12,12,24,0.94); color: ${LIGHT_TXT}; padding: 12px 18px; font-size: 13px; white-space: pre-wrap; display: none; border-top: 2px solid ${ACCENT}; z-index: 9; }
  #help { position: fixed; bottom: 10px; left: 14px; color: #4a4a68; font-size: 11px; z-index: 10; }`;

export interface HtmlSlide {
  html: string;
  notes: string;
  dark?: boolean;
}

class HtmlDeckBuilder {
  private slides: HtmlSlide[] = [];
  private figureCache = new Map<string, string | null>();

  constructor(
    private opts: HtmlDeckOptions,
    private progress: Progress
  ) {}

  private get content() {
    return CONTENT;
  }

  private get author(): string {
    return this.opts.author || this.content.author;
  }

  private async figure(name: string, figuresDir?: string): Promise<string | null> {
    const dir = figuresDir ?? this.opts.figuresDir;
    const key = `${dir}/${name}`;
    if (this.figureCache.has(key)) return this.figureCache.get(key)!;
    let data: string | null = null;
    try {
      const path = joinPath(this.opts.projectRoot, key);
      if (await pathExists(path)) {
        data = await readBinaryDataUrl(path);
      } else {
        data = await resolveIllustration(name, this.opts.imageMode ?? "native");
        if (!data) this.progress(`[WARN] Figure not found: ${key}`);
      }
    } catch (err) {
      this.progress(`[WARN] Failed to load figure ${name}: ${String(err)}`);
    }
    this.figureCache.set(key, data);
    return data;
  }

  private async figuresRow(names: string[], figuresDir?: string): Promise<string> {
    const imgs = (await Promise.all(names.map((n) => this.figure(n, figuresDir)))).filter(
      (d): d is string => d !== null
    );
    if (!imgs.length) return "";
    return `<div class="figrow">${imgs.map((d) => `<img src="${d}">`).join("")}</div>`;
  }

  private bulletsHtml(bullets: string[] | undefined, small = false): string {
    if (!bullets?.length) return "";
    return `<ul class="bullets${small ? " small" : ""}">${bullets
      .map((b) => `<li>${esc(b)}</li>`)
      .join("")}</ul>`;
  }

  private async equationsHtml(lines: string[], sizePt = 20): Promise<string> {
    const parts: string[] = [];
    for (const line of lines) {
      try {
        const eq = await renderEquationPng(line, { color: DARK });
        const h = Math.round(eq.height * (sizePt / 16));
        parts.push(`<img class="eq" style="height:${h}px" src="${eq.dataUrl}" alt="${esc(eq.fallback)}">`);
      } catch {
        parts.push(`<div class="eq-fallback">${esc(line)}</div>`);
      }
    }
    return `<div class="eqband">${parts.join("")}</div>`;
  }

  private push(title: string, bodyHtml: string, notes: string[], dark = false): void {
    this.slides.push({
      html: dark
        ? `<div class="slide dark">${bodyHtml}</div>`
        : `<div class="slide"><div class="titlebar">${esc(title)}</div><div class="body">${bodyHtml}</div></div>`,
      notes: notes.filter(Boolean).join("\n\n"),
      dark,
    });
  }

  // ── Slides ─────────────────────────────────────────────────────────────────

  private async cover(): Promise<void> {
    const coauthors = this.opts.coauthors ?? this.content.coauthors;
    const groups = this.opts.groups ?? this.content.research_groups;
    const logos: string[] = [];
    for (const rel of [
      "assets/images/logo-inescid.png",
      "assets/images/logo-ist.png",
      "assets/images/logo-cegist.png",
    ]) {
      const path = joinPath(this.opts.projectRoot, rel);
      if (await pathExists(path)) logos.push(await readBinaryDataUrl(path));
    }
    let conf = "";
    const confPath = joinPath(this.opts.projectRoot, "assets/images/logo-optimization2026.png");
    if (await pathExists(confPath)) conf = await readBinaryDataUrl(confPath);
    this.slides.push({
      dark: true,
      notes: `Presented by ${this.author}.${coauthors.length ? ` With ${coauthors.join(", ")}.` : ""}`,
      html: `<div class="slide dark cover">
${conf ? `<img class="conf-logo" src="${conf}">` : ""}
<h1>${esc(this.content.title)}</h1>
<div class="band"></div>
<p class="author">${esc(this.author)}</p>
${coauthors.length ? `<p class="coauthors">with ${esc(coauthors.join(", "))}</p>` : ""}
${groups.length ? `<p class="groups">${esc(groups.join("   ·   "))}</p>` : ""}
<div class="logos">${logos.map((d) => `<img src="${d}">`).join("")}</div>
</div>`,
    });
  }

  private agenda(): void {
    const items = this.content.agenda;
    const cardColors = [ACCENT, GREEN, ORANGE, "#8E3E7A"];
    this.slides.push({
      dark: true,
      notes: `Today's agenda: ${items.join("; ")}.`,
      html: `<div class="slide dark"><h2 class="agenda-title">Agenda</h2><div class="agenda">${items
        .map(
          (item, i) =>
            `<div class="agenda-card" style="border-color:${cardColors[i % cardColors.length]}">` +
            `<span class="badge" style="background:${cardColors[i % cardColors.length]}">${i + 1}</span>` +
            `<span>${esc(item)}</span></div>`
        )
        .join("")}</div></div>`,
    });
  }

  // native-shape diagrams as HTML/CSS
  private pipelineHtml(): string {
    const stages: [string, string][] = [
      ["Packing algorithm", "LA · LM · SL — which bins today?"],
      ["Routing algorithm", "exact · meta-h. · hyper-h. — build the routes"],
      ["Acceptance criterion (optional)", "constructor-dependent, e.g. BMC, OI"],
      ["Route improver (optional)", "CLS · FTSP — post-optimisation"],
    ];
    return `<div class="pipeline">${stages
      .map(
        ([name, cap]) =>
          `<div class="stage-wrap"><div class="chevron${name.includes("optional") ? " optional" : ""}">${esc(name)}</div><div class="stage-cap">${esc(cap)}</div></div>`
      )
      .join("")}</div>`;
  }

  private policyGridHtml(): string {
    const col = (header: string, body: string) =>
      `<div class="pg-col"><div class="pg-head">${esc(header)}</div>${body}</div>`;
    const flat = (items: string[]) =>
      items.map((i) => `<div class="pg-item">${esc(i)}</div>`).join("");
    const grouped = (groups: [string, string[]][]) =>
      groups
        .map(
          ([label, subs]) =>
            `<div class="pg-group"><div class="pg-group-label">${esc(label)}</div><div class="pg-group-items">${esc(subs.join("  ·  "))}</div></div>`
        )
        .join("");
    return `<div class="policy-grid">${col(
      "Packing",
      flat(["Look-Ahead (LA)", "Last-Minute (LM, CF70)", "Last-Minute (LM, CF90)", "Service-Level (SL1)", "Service-Level (SL2)"])
    )}${col(
      "Routing Algorithm",
      grouped([
        ["Exact Methods", ["BPC", "SWC-TCF"]],
        ["Meta-Heuristics", ["HGS", "ALNS", "SANS", "PG-CLNS", "PSOMA"]],
        ["Hyper-Heuristics", ["ACO-HH"]],
      ])
    )}${col("Route Improver", flat(["Fast-TSP", "Local Search (CLS)"]))}</div>`;
  }

  private taxonomyHtml(): string {
    const groups: [string, string, string[]][] = [
      ["Exact Methods", ACCENT, ["Branch-Price-and-Cut (BPC)", "Smart Waste Collection Two-Commodity Flow (SWC-TCF)"]],
      ["Meta-Heuristics", GREEN, [
        "Hybrid Genetic Search (HGS)",
        "Adaptive Large Neighborhood Search (ALNS)",
        "Simulated Annealing Neighborhood Search (SANS)",
        "Policy-Gradient Cooperative LNS (PG-CLNS)",
        "Particle Swarm Optimisation Memetic Algorithm (PSOMA)",
      ]],
      ["Hyper-Heuristics", ORANGE, ["Ant Colony Optimisation Hyper-Heuristic (ACO-HH)"]],
    ];
    return `<div class="taxonomy">${groups
      .map(
        ([header, color, items]) =>
          `<div class="tax-col"><div class="pg-head" style="background:${color}">${esc(header)}</div>${items
            .map((i) => `<div class="tax-item" style="border-color:${color}">${esc(i)}</div>`)
            .join("")}</div>`
      )
      .join("")}</div>`;
  }

  private doeTreeHtml(): string {
    const horizon = (label: string) =>
      `<div class="doe-h"><div class="doe-hbox">${esc(label)}</div><div class="doe-scen">${["RM-100", "RM-170", "FFZ-350"]
        .map((sc) => `<div class="doe-s"><div class="doe-sbox">${sc}</div><div class="doe-d">Empirical / Gamma-3</div></div>`)
        .join("")}</div></div>`;
    return `<div class="doe"><div class="doe-root">Simulation Runs</div><div class="doe-row">${horizon("30 Days")}${horizon("90 Days (Pareto-front policies only)")}</div></div>`;
  }

  private async simulatorSceneHtml(): Promise<string> {
    const binImg = await assetToDataUrl(GEN_IMAGES.waste_bin_icon);
    const truckImg = await assetToDataUrl(GEN_IMAGES.waste_truck_icon);
    const bin = (x: number, y: number, pct: string, selected = false) =>
      `<div class="bin${selected ? " selected" : ""}" style="left:${((x - 0.3) / 9.5) * 100}%;top:${((y - 4.2) / 3.4) * 100}%">` +
      `<img src="${binImg}"><span>${pct}</span></div>`;
    const left: [number, number, string][] = [
      [1.063, 4.912, "87%"], [0.431, 5.399, "60%"], [0.708, 6.182, "42%"], [1.405, 5.6, "50%"],
      [1.538, 4.57, "30%"], [2.063, 5.113, "80%"], [2.412, 6.26, "20%"], [2.851, 5.335, "55%"],
    ];
    const right: [number, number, string][] = [
      [7.424, 4.698, "87%"], [6.791, 5.185, "60%"], [7.068, 5.968, "42%"], [7.766, 5.386, "50%"],
      [7.899, 4.356, "30%"], [8.424, 4.899, "80%"], [8.773, 6.046, "20%"], [9.212, 5.121, "55%"],
    ];
    return `<div class="sim-scene">${left.map(([x, y, p], i) => bin(x, y, p, i === 0)).join("")}${right
      .map(([x, y, p]) => bin(x, y, p))
      .join("")}<img class="truck" src="${truckImg}"><div class="brace">⟩</div></div>`;
  }

  private async contentSlide(key: string): Promise<void> {
    const spec = this.content.slides[key];
    const figNames = spec.figures ?? (spec.figure ? [spec.figure] : []);
    const showBullets = spec.show_bullets ?? true;
    let body = "";
    if (spec.equation) {
      const eq = await this.equationsHtml(spec.equation, spec.eq_size_pt ?? 18);
      const figs = await this.figuresRow(figNames);
      body = figs
        ? `<div class="split"><div>${eq}${showBullets ? this.bulletsHtml(spec.bullets, true) : ""}</div><div>${figs}</div></div>`
        : `${eq}${showBullets ? this.bulletsHtml(spec.bullets) : ""}`;
    } else if (spec.diagram === "pipeline") {
      body = this.pipelineHtml() + (showBullets ? this.bulletsHtml(spec.bullets) : "");
    } else if (spec.diagram === "policy_grid") {
      body = this.policyGridHtml() + (showBullets ? this.bulletsHtml(spec.bullets, true) : "");
    } else if (spec.diagram === "algo_taxonomy") {
      const figs = await this.figuresRow(figNames);
      body = this.taxonomyHtml() + (showBullets ? this.bulletsHtml(spec.bullets, true) : "") + figs;
    } else if (spec.diagram === "metaheuristic_families") {
      body = (await this.figuresRow(figNames)) + (showBullets ? this.bulletsHtml(spec.bullets, true) : "");
    } else if (spec.diagram === "doe_tree") {
      body = this.doeTreeHtml() + (await this.figuresRow(figNames));
    } else if (figNames.length) {
      const figs = await this.figuresRow(figNames);
      body = showBullets
        ? `<div class="split"><div>${this.bulletsHtml(spec.bullets, true)}</div><div>${figs}</div></div>`
        : figs;
    } else if (showBullets) {
      body = this.bulletsHtml(spec.bullets);
    }
    this.push(spec.title, body, [spec.caption ?? "", ...(spec.bullets ?? []), ...(spec.speaker_notes ?? [])]);
  }

  private async simulator(): Promise<void> {
    const spec = this.content.slides.simulator;
    this.push(
      spec.title,
      this.pipelineHtml() + (await this.simulatorSceneHtml()),
      [...(spec.speaker_notes ?? []), ...(spec.bullets ?? [])]
    );
  }

  private objective(): void {
    const spec = this.content.slides.objective;
    const flow =
      `<div class="obj-flow">` +
      `<div class="obj-inputs">` +
      `<div class="obj-box" style="background:${ACCENT}">Exact Methods</div>` +
      `<div class="obj-box" style="background:${GREEN}">Meta-Heuristics</div>` +
      `<div class="obj-box" style="background:${ORANGE}">Hyper-Heuristics</div></div>` +
      `<div class="obj-arrow">→</div>` +
      `<div class="obj-sim">One Shared Simulator<small>Multi-day simulator: region × graph size × demand</small></div>` +
      `<div class="obj-arrow">→</div>` +
      `<div class="obj-box" style="background:${ACCENT}">Fair Benchmarks (KPIs)</div></div>`;
    this.push(spec.title, flow + this.taxonomyHtml(), spec.speaker_notes ?? []);
  }

  private async figureSlide(spec: ContentSlide): Promise<void> {
    const names = spec.figures ?? (spec.figure ? [spec.figure] : []);
    const figs = await this.figuresRow(names, spec.figures_dir);
    const caption = spec.caption ?? spec.note ?? "";
    this.push(spec.title, `${figs}${caption ? `<p class="caption">${esc(caption)}</p>` : ""}`, [caption]);
  }

  private resultsTableHtml(matrix: ResultsMatrix, globalBest: Map<string, ReturnType<typeof computeGlobalBest> extends Map<string, infer V> ? V : never>): string {
    const nLevels = matrix.colKeys[0]?.length ?? 0;
    const rowHeaders = ["Region", "N", "Dist"];
    let thead = "";
    for (let level = 0; level < nLevels; level++) {
      thead += "<tr>";
      if (level === 0) thead += rowHeaders.map((h) => `<th rowspan="${nLevels}">${h}</th>`).join("");
      for (const [start, end, label] of groupSpans(matrix.colKeys, level)) {
        thead += `<th colspan="${end - start}">${esc(String(label))}</th>`;
      }
      thead += "</tr>";
    }
    let tbody = "";
    let prev: RowKey | null = null;
    for (const rk of matrix.rowKeys) {
      tbody += "<tr>";
      for (let lvl = 0; lvl < rk.length; lvl++) {
        const same = prev !== null && prev.slice(0, lvl + 1).join("|") === rk.slice(0, lvl + 1).join("|");
        tbody += `<td class="rowlabel">${same ? "" : esc(String(rk[lvl]))}</td>`;
      }
      prev = rk;
      const best = globalBest.get(rk.join("|"));
      for (const ck of matrix.colKeys) {
        const raw = matrix.cells.get(cellKey(rk, ck)) ?? "—";
        const key = ck.join("|");
        const cls = best?.ov === key || best?.kg === key ? ' class="best"' : "";
        tbody += `<td${cls}>${raw.replace(" ov<br>", "<br>").replace(" kg/km", "")}</td>`;
      }
      tbody += "</tr>";
    }
    return `<table class="results"><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;
  }

  private async resultsTableSlides(): Promise<void> {
    const mode = this.opts.resultsTable ?? "30d";
    if (mode === "none") return;
    const data = await loadResultsTable(this.opts.projectRoot, mode);
    if (!data) {
      this.progress("[WARN] No data available for the full results table — skipping slide");
      return;
    }
    const globalBest = computeGlobalBest(data.matrix);
    for (const part of splitMatrix(data, this.opts.resultsTableSplit ?? "none")) {
      this.push(
        part.title,
        this.resultsTableHtml(part.matrix, globalBest) +
          `<p class="caption">Per cell: overflows (top, lower better) · KG / KM (bottom, higher better) — best per row highlighted.</p>`,
        [`CLS route improver results table.`]
      );
    }
  }

  private async acknowledgments(): Promise<void> {
    const ackText =
      "This work was supported by FCT – Foundation for Science and Technology, I.P., " +
      "under project 2022.04180.PTDC and individual PhD research grant - 2025.06860.BDANA";
    const fct = await assetToDataUrl(GEN_IMAGES.fct_b_horizontal_branco);
    this.slides.push({
      dark: true,
      notes: ackText,
      html: `<div class="slide dark statement"><h2>Acknowledgements</h2><p>${esc(ackText)}</p><img class="fct" src="${fct}"></div>`,
    });
  }

  private async qa(): Promise<void> {
    const spec = this.content.slides.qa;
    const fig = spec.figure ? await this.figure(spec.figure) : null;
    this.slides.push({
      dark: true,
      notes: (spec.bullets ?? []).join("\n\n"),
      html: `<div class="slide dark statement qa"><div><h2>${esc(spec.title)}</h2>${(spec.bullets ?? [])
        .map((b) => `<p>${esc(b)}</p>`)
        .join("")}</div>${fig ? `<img class="qa-fig" src="${fig}">` : ""}</div>`,
    });
  }

  // ── Page assembly ──────────────────────────────────────────────────────────

  private pageHtml(): string {
    const slidesHtml = this.slides
      .map(
        (s, i) =>
          `<section class="frame" data-idx="${i}">${s.html}<aside class="notes">${esc(s.notes)}</aside></section>`
      )
      .join("\n");
    return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>${esc(this.content.title)}</title>
<style>
${DECK_CSS}
</style>
</head>
<body>
${slidesHtml}
<div id="counter"></div>
<div id="notes-panel"></div>
<div id="help">←/→ navigate · N notes · Home/End</div>
<script>
  var frames = Array.from(document.querySelectorAll(".frame"));
  var idx = Math.min(Math.max(parseInt(location.hash.slice(1) || "0", 10) || 0, 0), frames.length - 1);
  var notesOn = false;
  function show(i) {
    idx = Math.min(Math.max(i, 0), frames.length - 1);
    frames.forEach(function (f, j) { f.classList.toggle("active", j === idx); });
    document.getElementById("counter").textContent = (idx + 1) + " / " + frames.length;
    var notes = frames[idx].querySelector(".notes");
    var panel = document.getElementById("notes-panel");
    panel.textContent = notes ? notes.textContent : "";
    panel.style.display = notesOn && panel.textContent ? "block" : "none";
    location.hash = String(idx);
  }
  document.addEventListener("keydown", function (e) {
    if (e.key === "ArrowRight" || e.key === " " || e.key === "PageDown") show(idx + 1);
    else if (e.key === "ArrowLeft" || e.key === "PageUp") show(idx - 1);
    else if (e.key === "Home") show(0);
    else if (e.key === "End") show(frames.length - 1);
    else if (e.key.toLowerCase() === "n") { notesOn = !notesOn; show(idx); }
  });
  document.addEventListener("click", function (e) {
    if (e.clientX > window.innerWidth / 2) show(idx + 1); else show(idx - 1);
  });
  show(idx);
</script>
</body>
</html>
`;
  }

  async buildSlides(): Promise<HtmlSlide[]> {
    this.progress("Building HTML deck …");
    await this.cover();
    this.agenda();
    await this.contentSlide("vrpp");
    this.objective();
    await this.simulator();
    await this.contentSlide("policy_overview");
    await this.contentSlide("strategies");
    await this.contentSlide("exact");
    await this.contentSlide("metaheuristics");
    await this.contentSlide("improvers");
    await this.contentSlide("design_of_experiments");
    for (const key of ["pareto", "kpi", "strategy_bubble", "scenario_heatmaps", "heatmaps", "improver_bubble"]) {
      await this.figureSlide(this.content.figure_slides[key]);
    }
    await this.resultsTableSlides();
    await this.contentSlide("conclusion");
    await this.acknowledgments();
    await this.qa();
    return this.slides;
  }

  async build(): Promise<string> {
    await this.buildSlides();
    const html = this.pageHtml();
    await writeTextFile(joinPath(this.opts.projectRoot, this.opts.out), html);
    this.progress(`Written: ${this.opts.out} (${this.slides.length} slides, ${(html.length / 1e6).toFixed(1)} MB)`);
    return this.opts.out;
  }
}

export async function generateHtmlDeck(opts: HtmlDeckOptions, progress: Progress = () => {}): Promise<string> {
  return await new HtmlDeckBuilder(opts, progress).build();
}

/** Build the slide list without writing the page (used by the PDF exporter). */
export async function buildHtmlDeckSlides(
  opts: HtmlDeckOptions,
  progress: Progress = () => {}
): Promise<HtmlSlide[]> {
  return await new HtmlDeckBuilder(opts, progress).buildSlides();
}
