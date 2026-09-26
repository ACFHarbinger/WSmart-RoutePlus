/**
 * Hierarchical results table (§H.6) — native port of `compute_global_best`,
 * `render_hier_table_image` (as a native PPTX table instead of a raster) and
 * `export_results_excel` (via exceljs) from `logic/gen/gen_presentation.py`.
 */
import ExcelJS from "exceljs";
import { SIM_CFG } from "../config";
import {
  aggregate,
  buildCtx,
  buildFullResultsMatrix,
  cellKey,
  filterData,
  groupSpans,
  loadHorizonCsv,
  type ColKey,
  type ResultsMatrix,
  type RowKey,
  type SimRow,
} from "../data/simulation";
import { joinPath, pathExists } from "../io";

export type ResultsTableMode = "30d" | "90d" | "all" | "none";
export type ResultsTableSplit = "none" | "strategy" | "constructor" | "improver";

const CELL_RE = /^(.*) ov<br>(.*) kg\/km$/;

function parseResultCell(text: string): { ov: number | null; kg: number | null } {
  const m = CELL_RE.exec(text);
  if (!m) return { ov: null, kg: null };
  const parse = (raw: string) => {
    const v = parseFloat(raw.split("±")[0]);
    return Number.isFinite(v) ? v : null;
  };
  return { ov: parse(m[1]), kg: parse(m[2]) };
}

export interface GlobalBest {
  ov: string | null;
  kg: string | null;
}

/** Best (lowest overflow / highest kg-km) column per row, over the FULL column set. */
export function computeGlobalBest(matrix: ResultsMatrix): Map<string, GlobalBest> {
  const best = new Map<string, GlobalBest>();
  for (const rk of matrix.rowKeys) {
    let ovKey: string | null = null;
    let ovVal = Infinity;
    let kgKey: string | null = null;
    let kgVal = -Infinity;
    for (const ck of matrix.colKeys) {
      const { ov, kg } = parseResultCell(matrix.cells.get(cellKey(rk, ck)) ?? "—");
      const key = ck.join("|");
      if (ov !== null && ov < ovVal) {
        ovVal = ov;
        ovKey = key;
      }
      if (kg !== null && kg > kgVal) {
        kgVal = kg;
        kgKey = key;
      }
    }
    best.set(rk.join("|"), { ov: ovKey, kg: kgKey });
  }
  return best;
}

// ── Data loading (ports _load_horizon / the results-table slide data flow) ───

export interface ResultsTableData {
  matrix: ResultsMatrix;
  title: string;
  multi: boolean;
  usedDays: number[];
}

async function loadClsHorizon(
  projectRoot: string,
  spec: { days: number; csv: string },
  horizonLabel?: string
): Promise<ResultsMatrix | null> {
  if (!spec.csv || !(await pathExists(joinPath(projectRoot, spec.csv)))) return null;
  const filter = { scenarios: SIM_CFG.scenarios, policies: SIM_CFG.policies };
  const rows = filterData(await loadHorizonCsv(projectRoot, spec.csv), filter);
  if (!rows.length) return null;
  let cls: SimRow[] = rows.filter((r) => r.improver.toUpperCase() === "CLS");
  if (!cls.length) cls = rows;
  const dfm = aggregate(cls);
  const ctx = buildCtx(cls, spec.days);
  return buildFullResultsMatrix(dfm, ctx, horizonLabel);
}

export async function loadResultsTable(
  projectRoot: string,
  mode: ResultsTableMode
): Promise<ResultsTableData | null> {
  if (mode === "none") return null;
  const specs = [...SIM_CFG.horizons].sort((a, b) => a.days - b.days);
  if (mode === "all") {
    const rowKeys: RowKey[] = [];
    const seen = new Set<string>();
    const colKeys: ColKey[] = [];
    const cells = new Map<string, string>();
    const usedDays: number[] = [];
    for (const spec of specs) {
      const m = await loadClsHorizon(projectRoot, spec, `${spec.days}d`);
      if (!m) continue;
      for (const rk of m.rowKeys) {
        const key = rk.join("|");
        if (!seen.has(key)) {
          seen.add(key);
          rowKeys.push(rk);
        }
      }
      colKeys.push(...m.colKeys);
      for (const [k, v] of m.cells) cells.set(k, v);
      usedDays.push(spec.days);
    }
    if (!rowKeys.length) return null;
    rowKeys.sort((a, b) => a[0].localeCompare(b[0]) || a[1] - b[1] || a[2].localeCompare(b[2]));
    return {
      matrix: { rowKeys, colKeys, cells },
      title: `Results — CLS Improver Table (All Horizons: ${usedDays.map((d) => `${d}d`).join(", ")})`,
      multi: true,
      usedDays,
    };
  }
  const days = parseInt(mode, 10);
  const spec = specs.find((s) => s.days === days);
  if (!spec) return null;
  const m = await loadClsHorizon(projectRoot, spec);
  if (!m) return null;
  return { matrix: m, title: `Results — CLS Improver (${days}-Day Horizon)`, multi: false, usedDays: [days] };
}

/** Partition the column set by a hierarchy level (ports --results-table-split). */
export function splitMatrix(data: ResultsTableData, split: ResultsTableSplit): ResultsTableData[] {
  if (split === "none") return [data];
  const levelNames = data.multi
    ? ["horizon", "strategy", "constructor", "improver"]
    : ["strategy", "constructor", "improver"];
  const level = levelNames.indexOf(split);
  if (level < 0) return [data];
  const groups = new Map<string, ColKey[]>();
  for (const ck of data.matrix.colKeys) {
    const key = ck[level];
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(ck);
  }
  return [...groups.entries()].map(([label, cols]) => ({
    ...data,
    title: `${data.title} — ${label}`,
    matrix: { ...data.matrix, colKeys: cols },
  }));
}

// ── Native PPTX table rows (pptxgenjs cell format) ───────────────────────────

export interface PptxTableCell {
  text: string;
  options: Record<string, unknown>;
}

const HDR_FILL = "1F2D3D";
const ALT_FILL = "EEF2F7";
const BEST_FILL = "C6EFCE";
const BEST_COLOR = "1F6B2C";

/** Build merged-header pptxgenjs rows for a results matrix (ports render_hier_table_image layout). */
export function buildPptxTableRows(
  matrix: ResultsMatrix,
  globalBest: Map<string, GlobalBest>,
  fontSize: number
): PptxTableCell[][] {
  const { rowKeys, colKeys, cells } = matrix;
  const nLevels = colKeys[0]?.length ?? 0;
  const rowHeaders = ["Region", "N", "Dist"];
  const rows: PptxTableCell[][] = [];
  const hdr = (text: string, extra: Record<string, unknown> = {}): PptxTableCell => ({
    text,
    options: {
      fill: { color: HDR_FILL },
      color: "FFFFFF",
      bold: true,
      align: "center",
      valign: "middle",
      fontSize,
      ...extra,
    },
  });

  // header rows: one per column-hierarchy level with merged spans
  for (let level = 0; level < nLevels; level++) {
    const row: PptxTableCell[] = [];
    if (level === 0) {
      rowHeaders.forEach((h) => row.push(hdr(h, { rowspan: nLevels })));
    }
    for (const [start, end, label] of groupSpans(colKeys, level)) {
      row.push(hdr(String(label), end - start > 1 ? { colspan: end - start } : {}));
    }
    rows.push(row);
  }

  // data rows with merged row-label spans and best-cell highlighting
  let prev: RowKey | null = null;
  rowKeys.forEach((rk, ri) => {
    const row: PptxTableCell[] = [];
    const altFill = ri % 2 === 1 ? { fill: { color: ALT_FILL } } : {};
    for (let lvl = 0; lvl < rk.length; lvl++) {
      const same = prev !== null && prev.slice(0, lvl + 1).join("|") === rk.slice(0, lvl + 1).join("|");
      row.push({
        text: same ? "" : String(rk[lvl]),
        options: { align: "center", valign: "middle", bold: true, fontSize, ...altFill },
      });
    }
    prev = rk;
    const best = globalBest.get(rk.join("|"));
    for (const ck of colKeys) {
      const raw = cells.get(cellKey(rk, ck)) ?? "—";
      const [ovPart, kgPart] = raw.includes("<br>") ? raw.split("<br>") : [raw, ""];
      const key = ck.join("|");
      const isBestOv = best?.ov === key;
      const isBestKg = best?.kg === key;
      row.push({
        text: kgPart ? `${ovPart.replace(" ov", "")}\n${kgPart.replace(" kg/km", "")}` : ovPart,
        options: {
          align: "center",
          valign: "middle",
          fontSize: fontSize - 1,
          ...(isBestOv || isBestKg
            ? { fill: { color: BEST_FILL }, color: BEST_COLOR, bold: true }
            : altFill),
        },
      });
    }
    rows.push(row);
  });
  return rows;
}

// ── Excel export (ports export_results_excel via exceljs) ───────────────────

export async function buildResultsWorkbook(data: ResultsTableData): Promise<ArrayBuffer> {
  const { matrix } = data;
  const wb = new ExcelJS.Workbook();
  const ws = wb.addWorksheet("Results");
  const nRowLabels = 3;

  const hdrFill: ExcelJS.Fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2D3D" } };
  const altFill: ExcelJS.Fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFEEF2F7" } };
  const bestOvFill: ExcelJS.Fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFC6EFCE" } };
  const bestKgFill: ExcelJS.Fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFBDD7EE" } };
  const center: Partial<ExcelJS.Alignment> = { horizontal: "center", vertical: "middle", wrapText: true };

  const headerRow = ws.getRow(1);
  ["Region", "N", "Distribution"].forEach((label, i) => {
    const cell = headerRow.getCell(i + 1);
    cell.value = label;
    cell.fill = hdrFill;
    cell.font = { color: { argb: "FFFFFFFF" }, bold: true };
    cell.alignment = center;
  });
  matrix.colKeys.forEach((ck, i) => {
    const cell = headerRow.getCell(nRowLabels + 1 + i);
    cell.value = ck.join(" / ");
    cell.fill = hdrFill;
    cell.font = { color: { argb: "FFFFFFFF" }, bold: true };
    cell.alignment = center;
  });
  headerRow.height = 40;

  const globalBest = computeGlobalBest(matrix);
  let prev: RowKey | null = null;
  matrix.rowKeys.forEach((rk, ri) => {
    const row = ws.getRow(ri + 2);
    const useAlt = (ri + 2) % 2 === 0;
    for (let li = 0; li < rk.length; li++) {
      const same = prev !== null && prev.slice(0, li + 1).join("|") === rk.slice(0, li + 1).join("|");
      const cell = row.getCell(li + 1);
      cell.value = same ? "" : rk[li];
      cell.alignment = center;
      if (useAlt) cell.fill = altFill;
    }
    prev = rk;
    const best = globalBest.get(rk.join("|"));
    matrix.colKeys.forEach((ck, ci) => {
      const cell = row.getCell(nRowLabels + 1 + ci);
      cell.value = (matrix.cells.get(cellKey(rk, ck)) ?? "—").replace("<br>", "\n");
      cell.alignment = center;
      const key = ck.join("|");
      if (best?.ov === key) cell.fill = bestOvFill;
      else if (best?.kg === key) cell.fill = bestKgFill;
      else if (useAlt) cell.fill = altFill;
    });
  });

  for (let ci = 1; ci <= nRowLabels; ci++) ws.getColumn(ci).width = 14;
  for (let ci = nRowLabels + 1; ci <= nRowLabels + matrix.colKeys.length; ci++) ws.getColumn(ci).width = 18;

  return await wb.xlsx.writeBuffer();
}
