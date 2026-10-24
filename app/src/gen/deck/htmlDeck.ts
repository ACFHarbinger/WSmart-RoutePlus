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
import {
  ACCENT,
  DARK,
  GREEN,
  ORANGE,
  DECK_CSS,
  esc,
  pipelineHtml,
  policyGridHtml,
  taxonomyHtml,
  doeTreeHtml,
  simulatorSceneHtml,
} from "./htmlDeckDiagrams";

export { DECK_CSS };

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
      body = pipelineHtml() + (showBullets ? this.bulletsHtml(spec.bullets) : "");
    } else if (spec.diagram === "policy_grid") {
      body = policyGridHtml() + (showBullets ? this.bulletsHtml(spec.bullets, true) : "");
    } else if (spec.diagram === "algo_taxonomy") {
      const figs = await this.figuresRow(figNames);
      body = taxonomyHtml() + (showBullets ? this.bulletsHtml(spec.bullets, true) : "") + figs;
    } else if (spec.diagram === "metaheuristic_families") {
      body = (await this.figuresRow(figNames)) + (showBullets ? this.bulletsHtml(spec.bullets, true) : "");
    } else if (spec.diagram === "doe_tree") {
      body = doeTreeHtml() + (await this.figuresRow(figNames));
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
      pipelineHtml() + (await simulatorSceneHtml()),
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
    this.push(spec.title, flow + taxonomyHtml(), spec.speaker_notes ?? []);
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
