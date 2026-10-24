/**
 * Results presentation deck builder (§H.6) — native pptxgenjs port of the
 * `DeckBuilder` class in `logic/gen/gen_presentation.py`.
 *
 * Reproduces the 21-slide structure: cover, agenda cards, content slides
 * (equations + figures + diagrams), native-shape objective/simulator slides,
 * results figure slides, the hierarchical results table (as a NATIVE PPTX
 * table with merged spans and best-cell highlighting), acknowledgements and
 * Q&A — plus the speaker-script DOCX and XLSX results workbook.
 */
import PptxGenJS from "pptxgenjs";
import { CONTENT, type ContentSlide } from "../config";
import { GEN_IMAGES, assetToDataUrl, imageSize } from "../assets";
import { joinPath, pathExists, readBinaryDataUrl } from "../io";
import { resolveIllustration, type ImageMode } from "./illustrations";
import {
  buildPptxTableRows,
  computeGlobalBest,
  loadResultsTable,
  splitMatrix,
  type ResultsTableMode,
  type ResultsTableSplit,
} from "./resultsTable";
import type { SlideScript } from "./speakerScript";
import type { Progress } from "../report/simulationReport";
import {
  SLIDE_W,
  SLIDE_H,
  ACCENT,
  DARK,
  MUTED,
  LIGHT_TXT,
  WHITE,
  GREEN,
  ORANGE,
  titleBar,
  bullets,
  placeFit,
  captionBox,
  equationFocus,
  figuresSideBySide,
  figuresGrid2x2,
  exportDeckArtifacts,
} from "./deckBuilderHelpers";
import {
  pipelineDiagram,
  policyGridDiagram,
  doeTreeDiagram,
  algoTaxonomyDiagram,
  renderObjectiveShapes,
  renderSimulatorShapes,
} from "./deckBuilderDiagrams";

export interface DeckOptions {
  projectRoot: string;
  figuresDir: string;
  out: string;
  author?: string;
  coauthors?: string[];
  groups?: string[];
  resultsTable?: ResultsTableMode;
  resultsTableSplit?: ResultsTableSplit;
  speakerScript?: boolean;
  speakerScriptOut?: string;
  imageMode?: ImageMode;
  excel?: boolean;
  /** Also export the self-contained HTML slideshow (§H.6). */
  html?: boolean;
  /** Also export the 16:9 PDF rendering of the deck (§H.6). */
  pdf?: boolean;
}

type Slide = ReturnType<PptxGenJS["addSlide"]>;

export class NativeDeckBuilder {
  private pptx: PptxGenJS;
  private scripts: SlideScript[] = [];
  private figCount = 0;
  private figureCache = new Map<string, { data: string; w: number; h: number } | null>();

  constructor(
    private opts: DeckOptions,
    private progress: Progress = () => {}
  ) {
    this.pptx = new PptxGenJS();
    this.pptx.defineLayout({ name: "WSR_WIDE", width: SLIDE_W, height: SLIDE_H });
    this.pptx.layout = "WSR_WIDE";
  }

  private get content() {
    return CONTENT;
  }

  private get author(): string {
    return this.opts.author || this.content.author;
  }

  private recordScript(title: string, paragraphs: string[]): void {
    this.scripts.push({
      number: this.scripts.length + 1,
      title,
      script: paragraphs.filter(Boolean).join("\n\n"),
    });
  }

  private figureCaption(slide: Slide, text: string, y?: number): void {
    this.figCount += 1;
    captionBox(slide, `Figure ${this.figCount}`, text, y);
  }

  /** Load a figure PNG from the figures dir (or an illustration builder). */
  private async figure(name: string, figuresDir?: string): Promise<{ data: string; w: number; h: number } | null> {
    const dir = figuresDir ?? this.opts.figuresDir;
    const key = `${dir}/${name}`;
    if (this.figureCache.has(key)) return this.figureCache.get(key)!;
    let result: { data: string; w: number; h: number } | null = null;
    const path = joinPath(this.opts.projectRoot, key);
    try {
      if (await pathExists(path)) {
        const data = await readBinaryDataUrl(path);
        const { w, h } = await imageSize(data);
        result = { data, w, h };
      } else {
        const built = await resolveIllustration(name, this.opts.imageMode ?? "native");
        if (built) {
          const { w, h } = await imageSize(built);
          result = { data: built, w, h };
        } else {
          this.progress(`[WARN] Figure not found: ${key}`);
        }
      }
    } catch (err) {
      this.progress(`[WARN] Failed to load figure ${name}: ${String(err)}`);
    }
    this.figureCache.set(key, result);
    return result;
  }

  // ── Slides ─────────────────────────────────────────────────────────────────

  private async cover(): Promise<void> {
    const slide = this.pptx.addSlide();
    slide.background = { color: DARK };
    const titleLen = this.content.title.length;
    const titleSz = titleLen > 80 ? 26 : titleLen > 55 ? 32 : 36;
    slide.addShape("rect", { x: 0, y: 4.55, w: SLIDE_W, h: 0.06, fill: { color: ACCENT } });
    slide.addText(this.content.title, { x: 0.9, y: 1.95, w: SLIDE_W - 1.8, h: 2.4, fontSize: titleSz, bold: true, color: WHITE, valign: "top" });
    const coauthors = this.opts.coauthors ?? this.content.coauthors;
    const groups = this.opts.groups ?? this.content.research_groups;
    const authorRuns: PptxGenJS.TextProps[] = [
      { text: this.author, options: { fontSize: 19, bold: true, color: WHITE, breakLine: true } },
    ];
    if (coauthors.length) {
      authorRuns.push({ text: `with ${coauthors.join(", ")}`, options: { fontSize: 14, color: LIGHT_TXT, breakLine: true, paraSpaceBefore: 6 } });
    }
    if (groups.length) {
      authorRuns.push({ text: groups.join("   ·   "), options: { fontSize: 11, italic: true, color: "8A9BB0", paraSpaceBefore: 10 } });
    }
    slide.addText(authorRuns, { x: 0.9, y: 4.8, w: SLIDE_W - 1.8, h: 2.4, valign: "top" });

    const logoSpecs: [string, number, number, number, number][] = [
      ["assets/images/logo-inescid.png", 0.9, 5.95, 1.83, 1.25],
      ["assets/images/logo-ist.png", 4.6, 5.82, 1.98, 1.69],
      ["assets/images/logo-cegist.png", 8.4, 5.97, 2.49, 1.38],
      ["assets/images/logo-optimization2026.png", 10.39, 0.15, 2.77, 1.35],
    ];
    for (const [rel, x, y, w, h] of logoSpecs) {
      const path = joinPath(this.opts.projectRoot, rel);
      if (await pathExists(path)) {
        const data = await readBinaryDataUrl(path);
        let fit = { x, y, w, h };
        try {
          const nat = await imageSize(data);
          if (nat.w > 0 && nat.h > 0) {
            const scale = Math.min(w / nat.w, h / nat.h);
            const fw = nat.w * scale;
            const fh = nat.h * scale;
            fit = { x: x + (w - fw) / 2, y: y + (h - fh) / 2, w: fw, h: fh };
          }
        } catch {}
        slide.addImage({ data, ...fit });
      }
    }
    this.recordScript(this.content.title, [
      `Presented by ${this.author}.${coauthors.length ? ` With ${coauthors.join(", ")}.` : ""}`,
    ]);
    slide.addNotes(this.scripts[this.scripts.length - 1].script);
  }

  private agenda(): void {
    const slide = this.pptx.addSlide();
    slide.background = { color: DARK };
    slide.addText("Agenda", { x: 0.9, y: 0.5, w: SLIDE_W - 1.8, h: 0.9, fontSize: 34, bold: true, color: WHITE });
    const items = this.content.agenda;
    const half = Math.ceil(items.length / 2);
    const cols = [items.slice(0, half), items.slice(half)];
    const top0 = 1.65;
    const availH = SLIDE_H - top0 - 0.5;
    const gapX = 0.4;
    const colW = (SLIDE_W - 1.2 - gapX) / 2;
    const cardColors = [ACCENT, GREEN, ORANGE, "8E3E7A"];
    cols.forEach((chunk, col) => {
      const left = 0.6 + col * (colW + gapX);
      const cardGap = 0.22;
      const cardH = (availH - cardGap * (chunk.length - 1)) / Math.max(chunk.length, 1);
      chunk.forEach((item, i) => {
        const idx = col * half + i;
        const top = top0 + i * (cardH + cardGap);
        const color = cardColors[idx % cardColors.length];
        slide.addShape("roundRect", { x: left, y: top, w: colW, h: cardH, fill: { color: "273749" }, line: { color, width: 1.25 }, rectRadius: 0.08 });
        const badgeD = Math.min(0.62, cardH - 0.15);
        slide.addShape("ellipse", { x: left + 0.18, y: top + (cardH - badgeD) / 2, w: badgeD, h: badgeD, fill: { color } });
        slide.addText(String(idx + 1), { x: left + 0.18, y: top + (cardH - badgeD) / 2, w: badgeD, h: badgeD, fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle" });
        slide.addText(item, { x: left + badgeD + 0.38, y: top, w: colW - badgeD - 0.55, h: cardH, fontSize: 19, bold: true, color: WHITE, valign: "middle" });
      });
    });
    this.recordScript("Agenda", [`Today's agenda: ${items.join("; ")}.`]);
    slide.addNotes(this.scripts[this.scripts.length - 1].script);
  }

  private async contentSlide(key: string): Promise<void> {
    const spec = this.content.slides[key];
    const slide = this.pptx.addSlide();
    titleBar(slide, spec.title);
    const showBullets = spec.show_bullets ?? true;
    const figNames = spec.figures ?? (spec.figure ? [spec.figure] : []);
    const figs = (await Promise.all(figNames.map((n) => this.figure(n)))).filter(
      (f): f is NonNullable<typeof f> => f !== null
    );
    const hasFig = figs.length > 0;

    if (spec.equation) {
      const colW = hasFig ? SLIDE_W / 2 - 0.4 : SLIDE_W - 1.2;
      const bulletsTop = await equationFocus(slide, spec.equation, {
        x: 0.6,
        w: colW,
        sizePt: spec.eq_size_pt ?? (hasFig ? 15 : 20),
        lineH: spec.eq_line_h ?? 0.62,
      }, this.progress);
      if (showBullets && spec.bullets) {
        bullets(slide, spec.bullets, { x: 0.6, y: bulletsTop, w: colW, size: hasFig ? 13 : 14 });
      }
      if (hasFig) {
        figuresSideBySide(slide, figs, SLIDE_W / 2 + 0.2, 1.2, SLIDE_W / 2 - 0.8, SLIDE_H - 1.5);
      }
    } else if (spec.diagram === "pipeline") {
      const bottom = pipelineDiagram(slide);
      if (showBullets && spec.bullets) bullets(slide, spec.bullets, { y: bottom + 0.1, size: 14 });
    } else if (spec.diagram === "policy_grid") {
      policyGridDiagram(slide);
      if (showBullets && spec.bullets) bullets(slide, spec.bullets, { y: 6.6, size: 13 });
    } else if (spec.diagram === "metaheuristic_families") {
      const gridBottom = SLIDE_H - 0.3;
      if (figs.length >= 3) {
        figuresGrid2x2(slide, figs, 0.4, 1.15, SLIDE_W - 0.8, gridBottom - 1.15);
      } else {
        figuresSideBySide(slide, figs, 0.4, 1.15, SLIDE_W - 0.8, gridBottom - 1.15);
      }
      if (showBullets && spec.bullets) bullets(slide, spec.bullets, { y: gridBottom, size: 12 });
    } else if (spec.diagram === "algo_taxonomy") {
      algoTaxonomyDiagram(slide);
      if (showBullets && spec.bullets) {
        bullets(slide, spec.bullets, { y: 4.35, w: hasFig ? SLIDE_W / 2 - 0.7 : undefined, size: 12 });
      }
      if (hasFig) {
        figuresSideBySide(slide, figs, SLIDE_W / 2 + 0.1, 1.2, SLIDE_W / 2 - 0.7, SLIDE_H - 1.5);
      }
    } else if (spec.diagram === "doe_tree") {
      const bottom = doeTreeDiagram(slide);
      const contentTop = bottom + 0.15;
      if (hasFig) {
        figuresSideBySide(slide, figs, 0.6, contentTop, SLIDE_W - 1.2, SLIDE_H - contentTop - 0.2);
      } else if (showBullets && spec.bullets) {
        bullets(slide, spec.bullets, { y: contentTop, size: 13 });
      }
    } else if (hasFig) {
      const figLeft = showBullets ? SLIDE_W / 2 : 0.5;
      const figW = showBullets ? SLIDE_W / 2 - 0.4 : SLIDE_W - 1.0;
      const hEach = SLIDE_H - 1.4;
      const wEach = figW / figs.length;
      figs.forEach((img, i) => placeFit(slide, img, figLeft + wEach * i, 1.2, wEach - 0.1, hEach));
      if (showBullets && spec.bullets) {
        bullets(slide, spec.bullets, { x: 0.5, y: 1.4, w: SLIDE_W / 2 - 0.7, size: 13 });
      }
    } else if (showBullets && spec.bullets) {
      bullets(slide, spec.bullets);
    }
    const scriptParts = [spec.caption ?? "", ...(spec.bullets ?? []), ...(spec.speaker_notes ?? [])];
    this.recordScript(spec.title, scriptParts);
    slide.addNotes(this.scripts[this.scripts.length - 1].script);
  }

  private objective(): void {
    const spec = this.content.slides.objective;
    const slide = this.pptx.addSlide();
    titleBar(slide, spec.title);
    renderObjectiveShapes(slide);
    this.recordScript(spec.title, spec.speaker_notes ?? []);
    slide.addNotes(this.scripts[this.scripts.length - 1].script);
  }

  private async simulator(): Promise<void> {
    const spec = this.content.slides.simulator;
    const slide = this.pptx.addSlide();
    titleBar(slide, spec.title);
    pipelineDiagram(slide);
    const binImg = await assetToDataUrl(GEN_IMAGES.waste_bin_icon);
    const truckImg = await assetToDataUrl(GEN_IMAGES.waste_truck_icon);
    renderSimulatorShapes(slide, binImg, truckImg);
    this.recordScript(spec.title, [...(spec.speaker_notes ?? []), ...(spec.bullets ?? [])]);
    slide.addNotes(this.scripts[this.scripts.length - 1].script);
  }

  private async figureSlide(spec: ContentSlide): Promise<void> {
    const slide = this.pptx.addSlide();
    titleBar(slide, spec.title);
    const names = spec.figures ?? (spec.figure ? [spec.figure] : []);
    const figs = (await Promise.all(names.map((n) => this.figure(n, spec.figures_dir)))).filter(
      (f): f is NonNullable<typeof f> => f !== null
    );
    const areaTop = 1.15;
    const caption = spec.caption ?? spec.note ?? "";
    const areaH = SLIDE_H - areaTop - (caption ? 0.65 : 0.2);
    figuresSideBySide(slide, figs, 0.3, areaTop, SLIDE_W - 0.6, areaH);
    if (caption) this.figureCaption(slide, caption);
    this.recordScript(spec.title, [spec.caption ?? spec.note ?? ""]);
    slide.addNotes(this.scripts[this.scripts.length - 1].script);
  }

  private async resultsTableSlide(): Promise<void> {
    const mode = this.opts.resultsTable ?? "30d";
    if (mode === "none") return;
    const data = await loadResultsTable(this.opts.projectRoot, mode);
    if (!data) {
      this.progress("[WARN] No data available for the full results table — skipping slide");
      return;
    }
    const partitions = splitMatrix(data, this.opts.resultsTableSplit ?? "none");
    const globalBest = computeGlobalBest(data.matrix);
    for (const part of partitions) {
      const slide = this.pptx.addSlide();
      titleBar(slide, part.title);
      const nCols = part.matrix.colKeys.length + 3;
      const fontSize = nCols > 22 ? 6 : nCols > 14 ? 7.5 : 9;
      const rows = buildPptxTableRows(part.matrix, globalBest, fontSize);
      slide.addTable(rows as never, {
        x: 0.25,
        y: 1.15,
        w: SLIDE_W - 0.5,
        border: { type: "solid", color: "C8D0DA", pt: 0.5 },
        align: "center",
        valign: "middle",
        autoPage: false,
      });
      slide.addText("LEFT: Overflows · RIGHT: KG / KM — best per row in green", {
        x: 0.25,
        y: SLIDE_H - 0.35,
        w: SLIDE_W - 0.5,
        h: 0.3,
        fontSize: 10,
        italic: true,
        color: MUTED,
      });
      const levelNames = (data.multi ? ["horizon"] : []).concat(["strategy", "constructor", "improver"]);
      this.recordScript(part.title, [
        `CLS route improver results table, columns grouped by ${levelNames.join(" × ")}. ` +
          `In each cell, top=overflows (lower is better), bottom=KG / KM (higher is better); best in green.`,
      ]);
      slide.addNotes(this.scripts[this.scripts.length - 1].script);
    }
  }

  private async acknowledgments(): Promise<void> {
    const slide = this.pptx.addSlide();
    slide.background = { color: DARK };
    slide.addText("Acknowledgements", { x: 1.0, y: 1.8, w: SLIDE_W - 2.0, h: 1.0, fontSize: 28, bold: true, color: WHITE, align: "center" });
    const ackText =
      "This work was supported by FCT – Foundation for Science and Technology, I.P., " +
      "under project 2022.04180.PTDC and individual PhD research grant - 2025.06860.BDANA";
    slide.addText(ackText, { x: 1.0, y: 3.0, w: SLIDE_W - 2.0, h: 1.8, fontSize: 17, color: LIGHT_TXT, align: "center", valign: "top" });
    const fct = await assetToDataUrl(GEN_IMAGES.fct_b_horizontal_branco);
    const { w, h } = await imageSize(fct);
    const logoW = 4.5;
    const logoH = (logoW * h) / w;
    slide.addImage({ data: fct, x: (SLIDE_W - logoW) / 2, y: 5.0, w: logoW, h: logoH });
    this.recordScript("Acknowledgements", [ackText]);
    slide.addNotes(ackText);
  }

  private async qa(): Promise<void> {
    const spec = this.content.slides.qa;
    const slide = this.pptx.addSlide();
    slide.background = { color: DARK };
    const fig = spec.figure ? await this.figure(spec.figure) : null;
    const hasFig = fig !== null;
    const textW = hasFig ? SLIDE_W / 2 - 1.1 : SLIDE_W - 1.8;
    slide.addText(
      [
        { text: spec.title, options: { fontSize: hasFig ? 30 : 36, bold: true, color: WHITE, breakLine: true } },
        ...(spec.bullets ?? []).map((line) => ({
          text: line,
          options: { fontSize: 16, color: LIGHT_TXT, breakLine: true, paraSpaceBefore: 8 },
        })),
      ],
      { x: 0.9, y: 2.8, w: textW, h: 2.4, align: hasFig ? "left" : "center", valign: "top" }
    );
    if (fig) {
      placeFit(slide, fig, SLIDE_W / 2 + 0.2, 0.6, SLIDE_W / 2 - 0.8, SLIDE_H - 1.2);
    }
    this.recordScript(spec.title, spec.bullets ?? []);
    slide.addNotes(this.scripts[this.scripts.length - 1].script);
  }

  // ── Build + export ─────────────────────────────────────────────────────────

  async build(): Promise<{ outputs: string[] }> {
    this.progress("Building deck: cover + agenda …");
    await this.cover(); // 1
    this.agenda(); // 2
    await this.contentSlide("vrpp"); // 3
    this.objective(); // 4
    await this.simulator(); // 5
    await this.contentSlide("policy_overview"); // 6
    await this.contentSlide("strategies"); // 7
    await this.contentSlide("exact"); // 8
    await this.contentSlide("metaheuristics"); // 9
    await this.contentSlide("improvers"); // 10
    await this.contentSlide("design_of_experiments"); // 11
    this.progress("Adding results figure slides …");
    await this.figureSlide(this.content.figure_slides.pareto); // 12
    await this.figureSlide(this.content.figure_slides.kpi); // 13
    await this.figureSlide(this.content.figure_slides.strategy_bubble); // 14
    await this.figureSlide(this.content.figure_slides.scenario_heatmaps); // 15
    await this.figureSlide(this.content.figure_slides.heatmaps); // 16
    await this.figureSlide(this.content.figure_slides.improver_bubble); // 17
    this.progress("Building results table slide …");
    await this.resultsTableSlide(); // 18
    await this.contentSlide("conclusion"); // 19
    await this.acknowledgments(); // 20
    await this.qa(); // 21

    const outputs = await exportDeckArtifacts(
      this.opts,
      this.pptx,
      this.content.title,
      this.author,
      this.scripts,
      this.progress
    );
    return { outputs };
  }
}

export async function generatePresentation(
  opts: DeckOptions,
  progress: Progress = () => {}
): Promise<{ outputs: string[] }> {
  return await new NativeDeckBuilder(opts, progress).build();
}
