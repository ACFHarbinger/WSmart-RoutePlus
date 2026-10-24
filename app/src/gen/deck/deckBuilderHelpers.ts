/**
 * Helper utilities for PPTX presentation deck generation (§H.6).
 */
import type PptxGenJS from "pptxgenjs";
import { renderEquationPng } from "./equations";
import { joinPath, writeBinaryFile } from "../io";
import { buildSpeakerScript, type SlideScript } from "./speakerScript";
import { buildResultsWorkbook, loadResultsTable } from "./resultsTable";
import { generateHtmlDeck } from "./htmlDeck";
import { generateDeckPdf } from "./pdfExport";
import type { Progress } from "../report/simulationReport";
import type { DeckOptions } from "./deckBuilder";

export const SLIDE_W = 13.333;
export const SLIDE_H = 7.5;

export const ACCENT = "2E74B5";
export const DARK = "1F2D3D";
export const MUTED = "5A6A7A";
export const LIGHT_TXT = "C9D6E4";
export const WHITE = "FFFFFF";
export const GREEN = "3E8E41";
export const ORANGE = "B06A2E";
export const LIGHT_FILL = "F0F4FA";
export const ARROW_ORANGE = "ED7D31";

export const PIPELINE_STAGES: [string, string][] = [
  ["Mandatory bin\nselection strategy", "LA · LM · SL\nwhich bins today?"],
  ["Route\nconstructor", "exact · meta-h. · hyper-h.\nbuild the routes"],
  ["Acceptance criterion\n(optional)", "constructor-dependent\ne.g. BMC, OI"],
  ["Route improver\n(optional)", "CLS · FTSP\npost-optimisation"],
];

type Slide = ReturnType<PptxGenJS["addSlide"]>;

export function titleBar(slide: Slide, title: string): void {
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

export function bullets(
  slide: Slide,
  items: string[],
  box: { x?: number; y?: number; w?: number; h?: number; size?: number } = {}
): void {
  if (!items.length) return;
  const x = box.x ?? 0.6;
  const y = box.y ?? 1.3;
  const w = box.w ?? SLIDE_W - 1.2;
  slide.addText(
    items.map((text) => ({
      text,
      options: { bullet: { characterCode: "2022", indent: 14 }, fontSize: box.size ?? 16, color: DARK, paraSpaceAfter: 8 },
    })),
    { x, y, w, h: box.h ?? SLIDE_H - y - 0.4, valign: "top" }
  );
}

export function placeFit(
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

export function captionBox(slide: Slide, label: string, text: string, y?: number): void {
  slide.addText(
    [
      { text: `${label}: `, options: { bold: true, fontSize: 11, color: DARK } },
      { text, options: { italic: true, fontSize: 11, color: MUTED } },
    ],
    { x: 0.6, y: y ?? SLIDE_H - 0.55, w: SLIDE_W - 1.2, h: 0.5, valign: "top" }
  );
}

export async function equationFocus(
  slide: Slide,
  lines: string[],
  box: { x?: number; y?: number; w?: number; sizePt?: number; lineH?: number } = {},
  progress: Progress = () => {}
): Promise<number> {
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
      progress(`[WARN] Equation render failed (${String(err)}); using text fallback`);
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

export function figuresSideBySide(
  slide: Slide,
  imgs: { data: string; w: number; h: number }[],
  x: number,
  y: number,
  totalW: number,
  totalH: number
): void {
  if (!imgs.length) return;
  const wEach = totalW / imgs.length;
  imgs.forEach((img, i) => placeFit(slide, img, x + wEach * i, y, wEach - 0.15, totalH));
}

export function figuresGrid2x2(
  slide: Slide,
  imgs: { data: string; w: number; h: number }[],
  x: number,
  y: number,
  totalW: number,
  totalH: number
): void {
  if (!imgs.length) return;
  const ncols = 2;
  const nrows = Math.ceil(imgs.length / ncols);
  const wEach = totalW / ncols;
  const hEach = totalH / nrows;
  imgs.forEach((img, i) => {
    placeFit(slide, img, x + (i % ncols) * wEach, y + Math.floor(i / ncols) * hEach, wEach - 0.15, hEach - 0.1);
  });
}

export async function exportDeckArtifacts(
  opts: DeckOptions,
  pptx: PptxGenJS,
  title: string,
  author: string,
  scripts: SlideScript[],
  progress: Progress
): Promise<string[]> {
  const outputs: string[] = [];
  progress("Writing PPTX …");
  const b64 = (await pptx.write({ outputType: "base64" })) as string;
  const outPath = joinPath(opts.projectRoot, opts.out);
  await writeBinaryFile(outPath, b64);
  outputs.push(opts.out);
  progress(`Written: ${opts.out} (${scripts.length} slides)`);

  if (opts.speakerScript) {
    const scriptOut =
      opts.speakerScriptOut?.trim() || opts.out.replace(/\.pptx$/i, ".docx");
    progress("Writing speaker script DOCX …");
    const blob = await buildSpeakerScript(title, author, scripts);
    const buf = new Uint8Array(await blob.arrayBuffer());
    let binary = "";
    for (let i = 0; i < buf.length; i += 0x8000) {
      binary += String.fromCharCode(...buf.subarray(i, i + 0x8000));
    }
    await writeBinaryFile(joinPath(opts.projectRoot, scriptOut), btoa(binary));
    outputs.push(scriptOut);
    progress(`Written: ${scriptOut}`);
  }

  if (opts.excel) {
    const mode = opts.resultsTable ?? "30d";
    const data = await loadResultsTable(opts.projectRoot, mode === "none" ? "30d" : mode);
    if (data) {
      progress("Writing XLSX results workbook …");
      const buffer = await buildResultsWorkbook(data);
      const arr = new Uint8Array(buffer);
      let binary = "";
      for (let i = 0; i < arr.length; i += 0x8000) {
        binary += String.fromCharCode(...arr.subarray(i, i + 0x8000));
      }
      const xlsxOut = opts.out.replace(/\.pptx$/i, ".xlsx");
      await writeBinaryFile(joinPath(opts.projectRoot, xlsxOut), btoa(binary));
      outputs.push(xlsxOut);
      progress(`Written: ${xlsxOut}`);
    } else {
      progress("[WARN] No data for Excel export");
    }
  }

  if (opts.html) {
    const htmlOut = await generateHtmlDeck(
      {
        projectRoot: opts.projectRoot,
        figuresDir: opts.figuresDir,
        out: opts.out.replace(/\.pptx$/i, ".html"),
        author: opts.author,
        coauthors: opts.coauthors,
        groups: opts.groups,
        resultsTable: opts.resultsTable,
        resultsTableSplit: opts.resultsTableSplit,
        imageMode: opts.imageMode,
      },
      progress
    );
    outputs.push(htmlOut);
  }

  if (opts.pdf) {
    const pdfOut = await generateDeckPdf(
      {
        projectRoot: opts.projectRoot,
        figuresDir: opts.figuresDir,
        out: opts.out,
        author: opts.author,
        coauthors: opts.coauthors,
        groups: opts.groups,
        resultsTable: opts.resultsTable,
        resultsTableSplit: opts.resultsTableSplit,
        imageMode: opts.imageMode,
      },
      progress
    );
    outputs.push(pdfOut);
  }
  return outputs;
}
