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
import { joinPath, pathExists, readBinaryDataUrl, writeBinaryFile } from "../io";
import { renderEquationPng } from "./equations";
import { resolveIllustration, type ImageMode } from "./illustrations";
import {
  buildPptxTableRows,
  buildResultsWorkbook,
  computeGlobalBest,
  loadResultsTable,
  splitMatrix,
  type ResultsTableMode,
  type ResultsTableSplit,
} from "./resultsTable";
import { buildSpeakerScript, type SlideScript } from "./speakerScript";
import { generateHtmlDeck } from "./htmlDeck";
import { generateDeckPdf } from "./pdfExport";
import type { Progress } from "../report/simulationReport";

const SLIDE_W = 13.333;
const SLIDE_H = 7.5;

const ACCENT = "2E74B5";
const DARK = "1F2D3D";
const MUTED = "5A6A7A";
const LIGHT_TXT = "C9D6E4";
const WHITE = "FFFFFF";
const GREEN = "3E8E41";
const ORANGE = "B06A2E";
const LIGHT_FILL = "F0F4FA";
const ARROW_ORANGE = "ED7D31";

const PIPELINE_STAGES: [string, string][] = [
  ["Mandatory bin\nselection strategy", "LA · LM · SL\nwhich bins today?"],
  ["Route\nconstructor", "exact · meta-h. · hyper-h.\nbuild the routes"],
  ["Acceptance criterion\n(optional)", "constructor-dependent\ne.g. BMC, OI"],
  ["Route improver\n(optional)", "CLS · FTSP\npost-optimisation"],
];

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
  private eqCount = 0;
  private tabCount = 0;
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

  private titleBar(slide: Slide, title: string): void {
    slide.addShape("rect", { x: 0, y: 0, w: SLIDE_W, h: 1.0, fill: { color: DARK } });
    slide.addText(title, {
      x: 0.5,
      y: 0,
      w: SLIDE_W - 1.0,
      h: 1.0,
      fontSize: 24,
      bold: true,
      color: WHITE,
      valign: "middle",
    });
  }

  private bullets(
    slide: Slide,
    bullets: string[],
    box: { x?: number; y?: number; w?: number; h?: number; size?: number } = {}
  ): void {
    if (!bullets.length) return;
    const x = box.x ?? 0.6;
    const y = box.y ?? 1.3;
    const w = box.w ?? SLIDE_W - 1.2;
    slide.addText(
      bullets.map((text) => ({
        text,
        options: { bullet: { characterCode: "2022", indent: 14 }, fontSize: box.size ?? 16, color: DARK, paraSpaceAfter: 8 },
      })),
      { x, y, w, h: box.h ?? SLIDE_H - y - 0.4, valign: "top" }
    );
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

  private placeFit(
    slide: Slide,
    img: { data: string; w: number; h: number },
    x: number,
    y: number,
    maxW: number,
    maxH: number
  ): void {
    const ratio = Math.min(maxW / img.w, maxH / img.h);
    const w = img.w * ratio;
    const h = img.h * ratio;
    slide.addImage({ data: img.data, x: x + (maxW - w) / 2, y: y + (maxH - h) / 2, w, h });
  }

  private captionBox(slide: Slide, label: string, text: string, y?: number): void {
    slide.addText(
      [
        { text: `${label}: `, options: { bold: true, fontSize: 11, color: DARK } },
        { text, options: { italic: true, fontSize: 11, color: MUTED } },
      ],
      { x: 0.6, y: y ?? SLIDE_H - 0.55, w: SLIDE_W - 1.2, h: 0.5, valign: "top" }
    );
  }

  private figureCaption(slide: Slide, text: string, y?: number): void {
    this.figCount += 1;
    this.captionBox(slide, `Figure ${this.figCount}`, text, y);
  }

  /** Equation focus band with per-line MathJax images (ports _equation_focus). */
  private async equationFocus(
    slide: Slide,
    lines: string[],
    box: { x?: number; y?: number; w?: number; sizePt?: number; lineH?: number } = {}
  ): Promise<number> {
    this.eqCount += 1;
    const x = box.x ?? 0.6;
    const y = box.y ?? 1.25;
    const w = box.w ?? SLIDE_W - 1.2;
    const lineH = box.lineH ?? 0.62;
    const areaH = lineH * lines.length + 0.4;
    slide.addShape("roundRect", { x, y, w, h: areaH, fill: { color: LIGHT_FILL }, rectRadius: 0.08 });
    const targetH = lineH * ((box.sizePt ?? 22) / 22) * 0.9;
    for (let i = 0; i < lines.length; i++) {
      try {
        const eq = await renderEquationPng(lines[i], { color: `#${DARK}` });
        const scale = Math.min((targetH * 96) / eq.height, ((w - 0.6) * 96) / eq.width);
        const ew = (eq.width * scale) / 96;
        const eh = (eq.height * scale) / 96;
        slide.addImage({
          data: eq.dataUrl,
          x: x + (w - ew) / 2,
          y: y + 0.2 + i * lineH + (lineH - eh) / 2,
          w: ew,
          h: eh,
          altText: eq.fallback,
        });
      } catch (err) {
        this.progress(`[WARN] Equation render failed (${String(err)}); using text fallback`);
        slide.addText(lines[i], {
          x,
          y: y + 0.2 + i * lineH,
          w,
          h: lineH,
          fontSize: box.sizePt ?? 22,
          color: DARK,
          align: "center",
          fontFace: "Cambria Math",
        });
      }
    }
    return y + areaH + 0.2;
  }

  // ── Diagrams as native shapes ──────────────────────────────────────────────

  private pipelineDiagram(slide: Slide): number {
    const n = PIPELINE_STAGES.length;
    const gap = 0.25;
    const left0 = 0.5;
    const totalW = SLIDE_W - 1.0;
    const stageW = (totalW - gap * (n - 1)) / n;
    const top = 1.9;
    const h = 1.5;
    const subH = 1.1;
    PIPELINE_STAGES.forEach(([name, caption], i) => {
      const left = left0 + i * (stageW + gap);
      const optional = name.includes("optional");
      slide.addShape("chevron", {
        x: left,
        y: top,
        w: stageW,
        h,
        fill: { color: optional ? "8A9BB0" : ACCENT },
      });
      slide.addText(name.replace(/\n/g, "\n"), {
        x: left,
        y: top,
        w: stageW,
        h,
        fontSize: 15,
        bold: true,
        color: WHITE,
        align: "center",
        valign: "middle",
      });
      slide.addText(caption, {
        x: left,
        y: top + h + 0.15,
        w: stageW,
        h: subH,
        fontSize: 14,
        color: MUTED,
        align: "center",
        valign: "top",
      });
    });
    return top + h + 0.15 + subH;
  }

  private policyGridDiagram(slide: Slide): void {
    const columns: [string, (string | [string, string[]])[], "flat" | "grouped"][] = [
      [
        "Mandatory Selection",
        ["Look-Ahead (LA)", "Last-Minute (LM, CF70)", "Last-Minute (LM, CF90)", "Service-Level (SL1)", "Service-Level (SL2)"],
        "flat",
      ],
      [
        "Route Constructor",
        [
          ["Exact Methods", ["BPC", "SWC-TCF"]],
          ["Meta-Heuristics", ["HGS", "ALNS", "SANS", "PG-CLNS", "PSOMA"]],
          ["Hyper-Heuristics", ["ACO-HH"]],
        ],
        "grouped",
      ],
      ["Route Improver", ["Fast-TSP", "Local Search (CLS)"], "flat"],
    ];
    const top = 1.15;
    const bottom = 6.5;
    const gap = 0.3;
    const left0 = 0.6;
    const colW = (SLIDE_W - 1.2 - gap * (columns.length - 1)) / columns.length;
    const headerH = 0.55;
    columns.forEach(([header, items, kind], ci) => {
      const left = left0 + ci * (colW + gap);
      slide.addShape("roundRect", { x: left, y: top, w: colW, h: headerH, fill: { color: DARK }, rectRadius: 0.06 });
      slide.addText(header, { x: left, y: top, w: colW, h: headerH, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle" });
      const bodyTop = top + headerH + 0.15;
      const bodyH = bottom - bodyTop;
      if (kind === "flat") {
        const flat = items as string[];
        const itemGap = 0.12;
        const itemH = (bodyH - itemGap * (flat.length - 1)) / flat.length;
        flat.forEach((label, ii) => {
          const y = bodyTop + ii * (itemH + itemGap);
          slide.addShape("roundRect", { x: left, y, w: colW, h: itemH, fill: { color: ACCENT }, rectRadius: 0.06 });
          slide.addText(label, { x: left, y, w: colW, h: itemH, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle" });
        });
      } else {
        const grouped = items as [string, string[]][];
        const grpGap = 0.15;
        const grpH = (bodyH - grpGap * (grouped.length - 1)) / grouped.length;
        grouped.forEach(([grpLabel, subItems], gi) => {
          const y = bodyTop + gi * (grpH + grpGap);
          slide.addShape("rect", {
            x: left,
            y,
            w: colW,
            h: grpH,
            fill: { color: WHITE, transparency: 100 },
            line: { color: MUTED, width: 1, dashType: "dash" },
          });
          slide.addText(
            [
              { text: grpLabel, options: { fontSize: 12, bold: true, color: ACCENT, align: "center", breakLine: true } },
              { text: subItems.join("  ·  "), options: { fontSize: 11, color: DARK, align: "center" } },
            ],
            { x: left, y, w: colW, h: grpH, valign: "top" }
          );
        });
      }
    });
  }

  private doeTreeDiagram(slide: Slide): number {
    const horizons: [string, [string, string[]][]][] = [
      ["30 Days", [["RM-100", ["Empirical", "Gamma-3"]], ["RM-170", ["Empirical", "Gamma-3"]], ["FFZ-350", ["Empirical", "Gamma-3"]]]],
      ["90 Days\n(Pareto-front policies only)", [["RM-100", ["Empirical", "Gamma-3"]], ["RM-170", ["Empirical", "Gamma-3"]], ["FFZ-350", ["Empirical", "Gamma-3"]]]],
    ];
    const top = 1.25;
    const rootW = 2.2;
    const rootH = 0.5;
    slide.addShape("roundRect", { x: (SLIDE_W - rootW) / 2, y: top, w: rootW, h: rootH, fill: { color: DARK }, rectRadius: 0.08 });
    slide.addText("Simulation Runs", { x: (SLIDE_W - rootW) / 2, y: top, w: rootW, h: rootH, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle" });
    const nH = horizons.length;
    const gapH = 0.4;
    const hzTop = top + rootH + 0.35;
    const hzW = (SLIDE_W - 1.2 - gapH * (nH - 1)) / nH;
    const hzH = 0.6;
    const left0 = 0.6;
    let distBottom = hzTop;
    horizons.forEach(([hzLabel, scenarios], hi) => {
      const hzLeft = left0 + hi * (hzW + gapH);
      slide.addShape("roundRect", { x: hzLeft, y: hzTop, w: hzW, h: hzH, fill: { color: ACCENT }, rectRadius: 0.06 });
      slide.addText(hzLabel, { x: hzLeft, y: hzTop, w: hzW, h: hzH, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle" });
      const nS = scenarios.length;
      const gapS = 0.15;
      const scTop = hzTop + hzH + 0.3;
      const scW = (hzW - gapS * (nS - 1)) / nS;
      const scH = 0.5;
      scenarios.forEach(([scLabel, dists], si) => {
        const scLeft = hzLeft + si * (scW + gapS);
        slide.addShape("rect", { x: scLeft, y: scTop, w: scW, h: scH, fill: { color: "8A9BB0" } });
        slide.addText(scLabel, { x: scLeft, y: scTop, w: scW, h: scH, fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle" });
        const distTop = scTop + scH + 0.1;
        slide.addText(dists.join(" / "), { x: scLeft, y: distTop, w: scW, h: 0.4, fontSize: 10, color: DARK, align: "center", valign: "top" });
        distBottom = distTop + 0.4;
      });
    });
    return distBottom;
  }

  private algoTaxonomyDiagram(slide: Slide): void {
    const groups: [string, string, string[]][] = [
      ["Exact Methods", ACCENT, ["Branch-Price-and-Cut\n(BPC)", "Smart Waste Collection\nTwo-Commodity Flow (SWC-TCF)"]],
      ["Meta-Heuristics", GREEN, [
        "Hybrid Genetic Search (HGS)",
        "Adaptive Large Neighborhood Search (ALNS)",
        "Simulated Annealing Neighborhood Search (SANS)",
        "Policy-Gradient Cooperative LNS (PG-CLNS)",
        "Particle Swarm Optimisation Memetic Algorithm (PSOMA)",
      ]],
      ["Hyper-Heuristics", ORANGE, ["Ant Colony Optimisation Hyper-Heuristic (ACO-HH)"]],
    ];
    const top = 1.25;
    const bottom = 4.15;
    const gap = 0.35;
    const left0 = 0.6;
    const colW = (SLIDE_W - 1.2 - gap * (groups.length - 1)) / groups.length;
    const headerH = 0.6;
    groups.forEach(([header, color, items], ci) => {
      const left = left0 + ci * (colW + gap);
      slide.addShape("roundRect", { x: left, y: top, w: colW, h: headerH, fill: { color }, rectRadius: 0.06 });
      slide.addText(header, { x: left, y: top, w: colW, h: headerH, fontSize: 17, bold: true, color: WHITE, align: "center", valign: "middle" });
      const bodyTop = top + headerH + 0.15;
      const bodyH = bottom - bodyTop;
      const itemGap = 0.12;
      const itemH = (bodyH - itemGap * (items.length - 1)) / items.length;
      items.forEach((label, ii) => {
        const y = bodyTop + ii * (itemH + itemGap);
        slide.addShape("roundRect", { x: left, y, w: colW, h: itemH, fill: { color: LIGHT_FILL }, line: { color, width: 1.25 }, rectRadius: 0.06 });
        slide.addText(label, { x: left, y, w: colW, h: itemH, fontSize: 11, bold: true, color: DARK, align: "center", valign: "middle" });
      });
    });
  }

  // ── Multi-figure placement helpers ─────────────────────────────────────────

  private async figuresSideBySide(
    slide: Slide,
    names: string[],
    x: number,
    y: number,
    totalW: number,
    totalH: number,
    figuresDir?: string
  ): Promise<void> {
    const imgs = (await Promise.all(names.map((n) => this.figure(n, figuresDir)))).filter(
      (i): i is NonNullable<typeof i> => i !== null
    );
    if (!imgs.length) return;
    const wEach = totalW / imgs.length;
    imgs.forEach((img, i) => this.placeFit(slide, img, x + wEach * i, y, wEach - 0.15, totalH));
  }

  private async figuresGrid2x2(
    slide: Slide,
    names: string[],
    x: number,
    y: number,
    totalW: number,
    totalH: number
  ): Promise<void> {
    const imgs = (await Promise.all(names.map((n) => this.figure(n)))).filter(
      (i): i is NonNullable<typeof i> => i !== null
    );
    if (!imgs.length) return;
    const ncols = 2;
    const nrows = Math.ceil(imgs.length / ncols);
    const wEach = totalW / ncols;
    const hEach = totalH / nrows;
    imgs.forEach((img, i) => {
      this.placeFit(slide, img, x + (i % ncols) * wEach, y + Math.floor(i / ncols) * hEach, wEach - 0.15, hEach - 0.1);
    });
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

    // Institution logos (repo assets) + conference logo — each [x, y, w, h] is
    // a bounding box; the image is contain-fitted to its natural aspect ratio
    // so logos are never stretched.
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
        } catch {
          /* keep the box as-is if the image cannot be measured */
        }
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
    this.titleBar(slide, spec.title);
    const showBullets = spec.show_bullets ?? true;
    const figNames = spec.figures ?? (spec.figure ? [spec.figure] : []);
    const figs = (await Promise.all(figNames.map((n) => this.figure(n)))).filter(
      (f): f is NonNullable<typeof f> => f !== null
    );
    const hasFig = figs.length > 0;

    if (spec.equation) {
      const colW = hasFig ? SLIDE_W / 2 - 0.4 : SLIDE_W - 1.2;
      const bulletsTop = await this.equationFocus(slide, spec.equation, {
        x: 0.6,
        w: colW,
        sizePt: spec.eq_size_pt ?? (hasFig ? 15 : 20),
        lineH: spec.eq_line_h ?? 0.62,
      });
      if (showBullets && spec.bullets) {
        this.bullets(slide, spec.bullets, { x: 0.6, y: bulletsTop, w: colW, size: hasFig ? 13 : 14 });
      }
      if (hasFig) {
        await this.figuresSideBySide(slide, figNames, SLIDE_W / 2 + 0.2, 1.2, SLIDE_W / 2 - 0.8, SLIDE_H - 1.5);
      }
    } else if (spec.diagram === "pipeline") {
      const bottom = this.pipelineDiagram(slide);
      if (showBullets && spec.bullets) this.bullets(slide, spec.bullets, { y: bottom + 0.1, size: 14 });
    } else if (spec.diagram === "policy_grid") {
      this.policyGridDiagram(slide);
      if (showBullets && spec.bullets) this.bullets(slide, spec.bullets, { y: 6.6, size: 13 });
    } else if (spec.diagram === "metaheuristic_families") {
      const gridBottom = SLIDE_H - 0.3;
      if (figNames.length >= 3) {
        await this.figuresGrid2x2(slide, figNames, 0.4, 1.15, SLIDE_W - 0.8, gridBottom - 1.15);
      } else {
        await this.figuresSideBySide(slide, figNames, 0.4, 1.15, SLIDE_W - 0.8, gridBottom - 1.15);
      }
      if (showBullets && spec.bullets) this.bullets(slide, spec.bullets, { y: gridBottom, size: 12 });
    } else if (spec.diagram === "algo_taxonomy") {
      this.algoTaxonomyDiagram(slide);
      if (showBullets && spec.bullets) {
        this.bullets(slide, spec.bullets, { y: 4.35, w: hasFig ? SLIDE_W / 2 - 0.7 : undefined, size: 12 });
      }
      if (hasFig) {
        await this.figuresSideBySide(slide, figNames, SLIDE_W / 2 + 0.1, 1.2, SLIDE_W / 2 - 0.7, SLIDE_H - 1.5);
      }
    } else if (spec.diagram === "doe_tree") {
      const bottom = this.doeTreeDiagram(slide);
      const contentTop = bottom + 0.15;
      if (hasFig) {
        await this.figuresSideBySide(slide, figNames, 0.6, contentTop, SLIDE_W - 1.2, SLIDE_H - contentTop - 0.2);
      } else if (showBullets && spec.bullets) {
        this.bullets(slide, spec.bullets, { y: contentTop, size: 13 });
      }
    } else if (hasFig) {
      const figLeft = showBullets ? SLIDE_W / 2 : 0.5;
      const figW = showBullets ? SLIDE_W / 2 - 0.4 : SLIDE_W - 1.0;
      // equal-height row
      const hEach = SLIDE_H - 1.4;
      const wEach = figW / figs.length;
      figs.forEach((img, i) => this.placeFit(slide, img, figLeft + wEach * i, 1.2, wEach - 0.1, hEach));
      if (showBullets && spec.bullets) {
        this.bullets(slide, spec.bullets, { x: 0.5, y: 1.4, w: SLIDE_W / 2 - 0.7, size: 13 });
      }
    } else if (showBullets && spec.bullets) {
      this.bullets(slide, spec.bullets);
    }
    const scriptParts = [spec.caption ?? "", ...(spec.bullets ?? []), ...(spec.speaker_notes ?? [])];
    this.recordScript(spec.title, scriptParts);
    slide.addNotes(this.scripts[this.scripts.length - 1].script);
  }

  private objective(): void {
    const spec = this.content.slides.objective;
    const slide = this.pptx.addSlide();
    this.titleBar(slide, spec.title);
    slide.addShape("rect", { x: 4.126, y: 1.242, w: 5.634, h: 0.438, fill: { color: DARK } });
    slide.addText("One framework to compare them all", { x: 4.126, y: 1.242, w: 5.634, h: 0.438, fontSize: 15, italic: true, color: WHITE, align: "center", valign: "middle" });

    const rbox = (x: number, y: number, w: number, h: number, color: string, text: string, size = 14) => {
      slide.addShape("roundRect", { x, y, w, h, fill: { color }, rectRadius: 0.06 });
      slide.addText(text, { x, y, w, h, fontSize: size, bold: true, color: WHITE, align: "center", valign: "middle" });
    };
    rbox(2.059, 1.791, 1.426, 0.457, ACCENT, "Exact Methods", 13);
    rbox(2.059, 2.414, 1.426, 0.6, GREEN, "Meta-Heuristics", 13);
    rbox(2.059, 3.159, 1.426, 0.6, ORANGE, "Hyper-Heuristics", 13);
    slide.addShape("roundRect", { x: 5.049, y: 1.704, w: 2.426, h: 2.046, fill: { color: DARK }, rectRadius: 0.08 });
    slide.addText(
      [
        { text: "One Shared Simulator", options: { fontSize: 15, bold: true, color: WHITE, align: "center", breakLine: true } },
        { text: "Multi-day simulator:\nregion × graph size × demand", options: { fontSize: 11, color: LIGHT_TXT, align: "center", paraSpaceBefore: 6 } },
      ],
      { x: 5.049, y: 1.704, w: 2.426, h: 2.046, valign: "middle" }
    );
    rbox(8.612, 2.158, 1.564, 1.139, ACCENT, "Fair Benchmarks\n(KPIs)", 13);
    const connector = (x: number, y: number, w: number, h: number, color: string, flipV = false) =>
      slide.addShape("line", { x, y, w, h, flipV, line: { color, width: 2, endArrowType: "triangle" } });
    connector(3.485, 2.02, 1.564, 0.707, ACCENT);
    connector(3.485, 2.714, 1.564, 0.013, GREEN);
    // flipV: sources at the Hyper-Heuristics box centre (bottom-left), arrow
    // into the simulator (top-right)
    connector(3.485, 2.727, 1.564, 0.732, ORANGE, true);
    connector(7.475, 2.727, 1.136, 0.001, ACCENT);

    const taxHeader = (x: number, y: number, color: string, text: string) => {
      slide.addShape("roundRect", { x, y, w: 3.811, h: 0.6, fill: { color }, rectRadius: 0.06 });
      slide.addText(text, { x, y, w: 3.811, h: 0.6, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle" });
    };
    const taxItem = (x: number, y: number, h: number, text: string, lineColor: string, size = 13) => {
      slide.addShape("roundRect", { x, y, w: 3.811, h, fill: { color: LIGHT_FILL }, line: { color: lineColor, width: 1.25 }, rectRadius: 0.06 });
      slide.addText(text, { x, y, w: 3.811, h, fontSize: size, bold: true, color: DARK, align: "center", valign: "middle" });
    };
    taxHeader(0.64, 4.258, ACCENT, "Exact Methods");
    taxItem(0.64, 5.008, 1.015, "Branch-and-Price-and-Cut (BPC)", ACCENT);
    taxItem(0.64, 6.143, 1.015, "Smart Waste Collection —\nTwo-Commodity Flow (SWC-TCF)", ACCENT);
    taxHeader(4.801, 4.258, GREEN, "Meta-Heuristics");
    (
      [
        ["Hybrid Genetic Search (HGS)", 5.008],
        ["Adaptive Large Neighborhood Search (ALNS)", 5.462],
        ["Simulated Annealing Neighborhood Search (SANS)", 5.916],
        ["Pheromone-Guided Cooperative Large Neighborhood Search (PG-CLNS)", 6.37],
        ["Particle Swarm Optimization Memetic Algorithm (PSOMA)", 6.824],
      ] as [string, number][]
    ).forEach(([algo], i) => taxItem(4.801, 5.008 + i * 0.49, 0.45, algo, GREEN, 10));
    taxHeader(8.962, 4.258, ORANGE, "Hyper-Heuristics");
    taxItem(8.962, 5.008, 2.15, "Ant Colony Optimization\nHyper-Heuristic (ACO-HH)", ORANGE);

    this.recordScript(spec.title, spec.speaker_notes ?? []);
    slide.addNotes(this.scripts[this.scripts.length - 1].script);
  }

  private async simulator(): Promise<void> {
    const spec = this.content.slides.simulator;
    const slide = this.pptx.addSlide();
    this.titleBar(slide, spec.title);
    this.pipelineDiagram(slide);

    const binImg = await assetToDataUrl(GEN_IMAGES.waste_bin_icon);
    const truckImg = await assetToDataUrl(GEN_IMAGES.waste_truck_icon);
    const binAndLabel = (x: number, y: number, pct: string) => {
      slide.addImage({ data: binImg, x, y, w: 0.495, h: 0.703 });
      slide.addText(pct, { x: x + 0.023, y: y + 0.36, w: 0.449, h: 0.286, fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle" });
    };
    const leftBins: [number, number, string][] = [
      [1.063, 4.912, "87%"], [0.431, 5.399, "60%"], [0.708, 6.182, "42%"], [1.405, 5.6, "50%"],
      [1.538, 4.57, "30%"], [2.063, 5.113, "80%"], [2.412, 6.26, "20%"], [2.851, 5.335, "55%"],
    ];
    for (const [x, y, pct] of leftBins) binAndLabel(x, y, pct);
    slide.addShape("rect", { x: 0.99, y: 4.902, w: 0.599, h: 0.759, fill: { color: WHITE, transparency: 100 }, line: { color: "FF0000", width: 2.5 } });
    const rightBins: [number, number, string][] = [
      [7.424, 4.698, "87%"], [6.791, 5.185, "60%"], [7.068, 5.968, "42%"], [7.766, 5.386, "50%"],
      [7.899, 4.356, "30%"], [8.424, 4.899, "80%"], [8.773, 6.046, "20%"], [9.212, 5.121, "55%"],
    ];
    for (const [x, y, pct] of rightBins) binAndLabel(x, y, pct);
    slide.addImage({ data: truckImg, x: 7.7, y: 6.673, w: 1.231, h: 0.943 });

    // Collection-tour arrows (orange): truck → 50 → 60 → 87 → 30 → 80 → truck.
    // Bounding boxes + flips copied verbatim from the reference deck
    // (tmp/wsmart_route_results.pptx slide 5) — the flips encode the tour
    // direction, so the arrowhead lands on the tour target.
    const tourArrows: [number, number, number, number, boolean, boolean][] = [
      [8.005, 6.032, 0.217, 0.756, true, true], // truck → 50
      [7.245, 5.688, 0.528, 0.158, true, true], // 50 → 60
      [7.039, 5.049, 0.385, 0.136, false, true], // 60 → 87
      [7.706, 4.567, 0.226, 0.146, false, true], // 87 → 30
      [8.299, 4.661, 0.373, 0.238, false, false], // 30 → 80
      [8.462, 5.575, 0.14, 1.213, true, false], // 80 → truck
    ];
    for (const [x, y, w, h, flipH, flipV] of tourArrows) {
      slide.addShape("line", {
        x,
        y,
        w,
        h,
        flipH,
        flipV,
        line: { color: ARROW_ORANGE, width: 1.5, endArrowType: "triangle" },
      });
    }

    // Horizontal underbrace beneath the constructor→improver chevrons — a
    // rightBrace rotated 90° (geometry from the reference deck; the unrotated
    // bbox spans vertically and rotates about its centre).
    slide.addShape("rightBrace", {
      x: 7.977,
      y: -0.194,
      w: 0.323,
      h: 8.752,
      rotate: 90,
      fill: { color: WHITE, transparency: 100 },
      line: { color: ACCENT, width: 1.5 },
    });

    this.recordScript(spec.title, [...(spec.speaker_notes ?? []), ...(spec.bullets ?? [])]);
    slide.addNotes(this.scripts[this.scripts.length - 1].script);
  }

  private async figureSlide(spec: ContentSlide): Promise<void> {
    const slide = this.pptx.addSlide();
    this.titleBar(slide, spec.title);
    const names = spec.figures ?? (spec.figure ? [spec.figure] : []);
    const areaTop = 1.15;
    const caption = spec.caption ?? spec.note ?? "";
    const areaH = SLIDE_H - areaTop - (caption ? 0.65 : 0.2);
    await this.figuresSideBySide(slide, names, 0.3, areaTop, SLIDE_W - 0.6, areaH, spec.figures_dir);
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
      this.titleBar(slide, part.title);
      this.tabCount += 1;
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
      this.placeFit(slide, fig, SLIDE_W / 2 + 0.2, 0.6, SLIDE_W / 2 - 0.8, SLIDE_H - 1.2);
    }
    this.recordScript(spec.title, spec.bullets ?? []);
    slide.addNotes(this.scripts[this.scripts.length - 1].script);
  }

  // ── Build + export ─────────────────────────────────────────────────────────

  async build(): Promise<{ outputs: string[] }> {
    const outputs: string[] = [];
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

    this.progress("Writing PPTX …");
    const b64 = (await this.pptx.write({ outputType: "base64" })) as string;
    const outPath = joinPath(this.opts.projectRoot, this.opts.out);
    await writeBinaryFile(outPath, b64);
    outputs.push(this.opts.out);
    this.progress(`Written: ${this.opts.out} (${this.scripts.length} slides)`);

    if (this.opts.speakerScript) {
      const scriptOut =
        this.opts.speakerScriptOut?.trim() || this.opts.out.replace(/\.pptx$/i, ".docx");
      this.progress("Writing speaker script DOCX …");
      const blob = await buildSpeakerScript(this.content.title, this.author, this.scripts);
      const buf = new Uint8Array(await blob.arrayBuffer());
      let binary = "";
      for (let i = 0; i < buf.length; i += 0x8000) {
        binary += String.fromCharCode(...buf.subarray(i, i + 0x8000));
      }
      await writeBinaryFile(joinPath(this.opts.projectRoot, scriptOut), btoa(binary));
      outputs.push(scriptOut);
      this.progress(`Written: ${scriptOut}`);
    }

    if (this.opts.excel) {
      const mode = this.opts.resultsTable ?? "30d";
      const data = await loadResultsTable(this.opts.projectRoot, mode === "none" ? "30d" : mode);
      if (data) {
        this.progress("Writing XLSX results workbook …");
        const buffer = await buildResultsWorkbook(data);
        const arr = new Uint8Array(buffer);
        let binary = "";
        for (let i = 0; i < arr.length; i += 0x8000) {
          binary += String.fromCharCode(...arr.subarray(i, i + 0x8000));
        }
        const xlsxOut = this.opts.out.replace(/\.pptx$/i, ".xlsx");
        await writeBinaryFile(joinPath(this.opts.projectRoot, xlsxOut), btoa(binary));
        outputs.push(xlsxOut);
        this.progress(`Written: ${xlsxOut}`);
      } else {
        this.progress("[WARN] No data for Excel export");
      }
    }

    if (this.opts.html) {
      const htmlOut = await generateHtmlDeck(
        {
          projectRoot: this.opts.projectRoot,
          figuresDir: this.opts.figuresDir,
          out: this.opts.out.replace(/\.pptx$/i, ".html"),
          author: this.opts.author,
          coauthors: this.opts.coauthors,
          groups: this.opts.groups,
          resultsTable: this.opts.resultsTable,
          resultsTableSplit: this.opts.resultsTableSplit,
          imageMode: this.opts.imageMode,
        },
        this.progress
      );
      outputs.push(htmlOut);
    }

    if (this.opts.pdf) {
      const pdfOut = await generateDeckPdf(
        {
          projectRoot: this.opts.projectRoot,
          figuresDir: this.opts.figuresDir,
          out: this.opts.out,
          author: this.opts.author,
          coauthors: this.opts.coauthors,
          groups: this.opts.groups,
          resultsTable: this.opts.resultsTable,
          resultsTableSplit: this.opts.resultsTableSplit,
          imageMode: this.opts.imageMode,
        },
        this.progress
      );
      outputs.push(pdfOut);
    }
    return { outputs };
  }
}

export async function generatePresentation(
  opts: DeckOptions,
  progress: Progress = () => {}
): Promise<{ outputs: string[] }> {
  return await new NativeDeckBuilder(opts, progress).build();
}
