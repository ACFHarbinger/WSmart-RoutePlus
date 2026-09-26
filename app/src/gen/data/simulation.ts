/**
 * Simulation results data engine (§H.1) — native port of the data layer of
 * `logic/gen/gen_simulation_analysis.py`.
 *
 * Covers: raw output-tree parsing (filename-encoded policy metadata),
 * horizon CSV loading, scenario detection, config filtering, CF/SL variant
 * aggregation, Pareto fronts, and every markdown table builder.
 */
import { META, KGKM_LABEL } from "../config";
import { joinPath, listFilesRecursive, loadCsv, readTextFile } from "../io";

export interface SimRow {
  city: string;
  N: number;
  dist: string;
  improver: string;
  strategy: string;
  cf: string;
  sl_var: string;
  acceptance: string;
  constructor: string;
  overflows: number;
  kg: number;
  ncol: number;
  kg_lost: number;
  km: number;
  kgkm: number;
  reward: number;
  profit: number;
  time: number;
  days: number;
  /** derived: LM (CF70) / SL (SL1) / LA … */
  variant: string;
}

export interface Scenario {
  city: string;
  N: number;
  dist: string;
}

export interface AnalysisFilter {
  scenarios?: Scenario[] | null;
  policies?: {
    strategies?: string[] | null;
    constructors?: string[] | null;
    improvers?: string[] | null;
    acceptance?: string[] | null;
  };
}

export type SimMetric = "overflows" | "kgkm" | "km" | "profit" | "kg" | "reward";

// ── Filename / directory parsing (ports _parse_area_dir / _parse_filename) ──

export function parseAreaDir(dirname: string): { city: string; N: number } | null {
  const m = /^([a-z]+?)(\d+)(?:_\w+)?$/.exec(dirname);
  if (!m) return null;
  const city = META.dir_city[m[1]];
  if (!city) return null;
  return { city, N: parseInt(m[2], 10) };
}

export interface ParsedFilename {
  strategy: string;
  cf: string | null;
  sl_var: string | null;
  improver: string;
  constructor: string;
  acceptance: string | null;
}

export function parseFilename(stem: string): ParsedFilename | null {
  if (!stem.startsWith("log_")) return null;
  let rest = stem.slice(4);

  let strategy: string | null = null;
  let cf: string | null = null;
  let slVar: string | null = null;
  for (const spec of META.strategy_prefixes) {
    if (rest.startsWith(spec.prefix)) {
      strategy = spec.strategy;
      cf = spec.cf;
      slVar = spec.sl_var;
      rest = rest.slice(spec.prefix.length);
      break;
    }
  }
  if (strategy === null) return null;

  const impRe = new RegExp(META.improver_pattern, "i");
  const m = impRe.exec(rest);
  if (!m) return null;
  const improver = m[1].toUpperCase();
  const middle = rest.slice(0, m.index);

  let constructor: string | null = null;
  let acceptance: string | null = null;
  for (const [token, label] of META.constructors) {
    if (middle.startsWith(token)) {
      constructor = label;
      acceptance = middle.slice(token.length).replace(/^_+|_+$/g, "") || null;
      break;
    }
  }
  if (constructor === null) return null;

  return { strategy, cf, sl_var: slVar, improver, constructor, acceptance };
}

/** Walk an `assets/output/<horizon>days/` tree into SimRows (ports parse_output_dir). */
export async function parseOutputDir(root: string): Promise<SimRow[]> {
  const files = await listFilesRecursive(root, { prefix: "log_", suffix: ".json" });
  const rows: SimRow[] = [];
  const rootNorm = root.replace(/[\\/]+$/, "");
  for (const file of files) {
    const rel = file.slice(rootNorm.length + 1).split(/[\\/]/);
    if (rel.length < 3) continue; // area/dist/.../log_*.json
    const area = parseAreaDir(rel[0]);
    if (!area) continue;
    const dist = META.dist_map[rel[1]];
    if (!dist) continue;
    const stem = rel[rel.length - 1].replace(/\.json$/i, "");
    const meta = parseFilename(stem);
    if (!meta) continue;
    let mean: Record<string, number>;
    try {
      const data = JSON.parse(await readTextFile(file)) as { mean?: Record<string, number> };
      if (!data.mean || Object.keys(data.mean).length === 0) continue;
      mean = data.mean;
    } catch {
      continue;
    }
    rows.push(
      withVariant({
        city: area.city,
        N: area.N,
        dist,
        improver: meta.improver,
        strategy: meta.strategy,
        cf: meta.cf ?? "",
        sl_var: meta.sl_var ?? "",
        acceptance: meta.acceptance ?? "",
        constructor: meta.constructor,
        overflows: mean.overflows ?? 0,
        kg: mean.kg ?? 0,
        ncol: mean.ncol ?? 0,
        kg_lost: mean.kg_lost ?? 0,
        km: mean.km ?? 0,
        kgkm: mean["kg/km"] ?? 0,
        reward: mean.reward ?? 0,
        profit: mean.profit ?? 0,
        time: mean.time ?? 0,
        days: mean.days ?? 0,
      })
    );
  }
  return rows;
}

/** Serialize SimRows to the summary CSV schema (ports the --parse-output mode). */
export function rowsToCsv(rows: SimRow[]): string {
  const cols = [
    "city", "N", "dist", "improver", "strategy", "cf", "sl_var", "acceptance",
    "constructor", "overflows", "kg", "ncol", "kg_lost", "km", "kgkm", "reward",
    "profit", "time", "days",
  ] as const;
  const esc = (v: unknown) => {
    const s = String(v ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [cols.join(",")];
  for (const r of rows) lines.push(cols.map((c) => esc(r[c])).join(","));
  return lines.join("\n") + "\n";
}

// ── Labels (ports region_label / scenario_label / variant_label) ────────────

export function regionLabel(city: string, N: number): string {
  const short = META.city_short[city] ?? city.slice(0, 3).toUpperCase();
  return `${short}-${N}`;
}

export function scenarioLabel(s: Scenario): string {
  return `${regionLabel(s.city, s.N)} / ${s.dist}`;
}

function variantLabel(row: { strategy: string; cf?: string; sl_var?: string }): string {
  if (row.strategy === "LM" && row.cf) return `LM (${row.cf})`;
  if (row.strategy === "SL" && row.sl_var) return `SL (${row.sl_var})`;
  return row.strategy;
}

function withVariant(row: Omit<SimRow, "variant">): SimRow {
  return { ...row, variant: variantLabel(row) };
}

export function variantColor(label: string, fallbackIdx = 0): string {
  const palette = META.variant_colors;
  if (palette[label]) return palette[label];
  const cycle = [...Object.values(palette), ...META.scenario_colors];
  return cycle[fallbackIdx % cycle.length];
}

// ── CSV loading (ports load_horizon_csv) ─────────────────────────────────────

function num(v: unknown): number {
  const n = typeof v === "number" ? v : parseFloat(String(v ?? ""));
  return Number.isFinite(n) ? n : 0;
}

function str(v: unknown): string {
  if (v == null) return "";
  const s = String(v);
  return s === "nan" || s === "None" ? "" : s;
}

export async function loadHorizonCsv(projectRoot: string, csvRel: string): Promise<SimRow[]> {
  const file = await loadCsv(joinPath(projectRoot, csvRel));
  return file.rows.map((r) =>
    withVariant({
      city: str(r.city),
      N: num(r.N),
      dist: str(r.dist),
      improver: str(r.improver),
      strategy: str(r.strategy),
      cf: str(r.cf),
      sl_var: str(r.sl_var),
      acceptance: str(r.acceptance),
      constructor: str(r.constructor),
      overflows: num(r.overflows),
      kg: num(r.kg),
      ncol: num(r.ncol),
      kg_lost: num(r.kg_lost),
      km: num(r.km),
      kgkm: num(r.kgkm ?? r["kg/km"]),
      reward: num(r.reward),
      profit: num(r.profit),
      time: num(r.time),
      days: num(r.days),
    })
  );
}

// ── Scenario detection + filtering (ports detect_scenarios / filter_data) ───

export function detectScenarios(rows: SimRow[]): Scenario[] {
  const seen = new Map<string, Scenario>();
  for (const r of rows) {
    const key = `${r.city}|${r.N}|${r.dist}`;
    if (!seen.has(key)) seen.set(key, { city: r.city, N: r.N, dist: r.dist });
  }
  return [...seen.values()].sort(
    (a, b) => a.N - b.N || a.city.localeCompare(b.city) || a.dist.localeCompare(b.dist)
  );
}

export function filterData(rows: SimRow[], config: AnalysisFilter): SimRow[] {
  let out = rows;
  if (config.scenarios?.length) {
    const keys = new Set(config.scenarios.map((s) => `${s.city}|${s.N}|${s.dist}`));
    out = out.filter((r) => keys.has(`${r.city}|${r.N}|${r.dist}`));
  }
  const pol = config.policies ?? {};
  const fields: [keyof NonNullable<AnalysisFilter["policies"]>, keyof SimRow][] = [
    ["strategies", "strategy"],
    ["constructors", "constructor"],
    ["improvers", "improver"],
    ["acceptance", "acceptance"],
  ];
  for (const [key, col] of fields) {
    const allowed = pol[key];
    if (allowed?.length) out = out.filter((r) => allowed.includes(String(r[col])));
  }
  return out;
}

export function scenSub<T extends Scenario>(rows: (SimRow & Partial<T>)[], s: Scenario): SimRow[] {
  return rows.filter((r) => r.city === s.city && r.N === s.N && r.dist === s.dist);
}

// ── Aggregation (ports aggregate) ────────────────────────────────────────────

/** Average metrics over CF/SL variants (per scenario × strategy × constructor × improver). */
export function aggregate(rows: SimRow[]): SimRow[] {
  const groups = new Map<string, SimRow[]>();
  for (const r of rows) {
    const key = [r.city, r.N, r.dist, r.improver, r.strategy, r.constructor].join("|");
    const arr = groups.get(key);
    if (arr) arr.push(r);
    else groups.set(key, [r]);
  }
  const metrics: SimMetric[] = ["overflows", "kgkm", "km", "profit", "kg", "reward"];
  const out: SimRow[] = [];
  for (const grp of groups.values()) {
    const base = { ...grp[0], cf: "", sl_var: "", acceptance: "" };
    for (const m of metrics) {
      base[m] = grp.reduce((acc, r) => acc + r[m], 0) / grp.length;
    }
    out.push(withVariant(base));
  }
  return out;
}

// ── Statistics helpers ───────────────────────────────────────────────────────

export function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
}

export function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
}

export function metricValues(rows: SimRow[], metric: SimMetric): number[] {
  return rows.map((r) => r[metric]);
}

/** Positional indices of non-dominated points (min x=overflows, max y=kgkm) — ports pareto_indices. */
export function paretoIndices(xs: number[], ys: number[]): Set<number> {
  const order = [...xs.keys()].sort((a, b) => xs[a] - xs[b] || ys[b] - ys[a]);
  const front = new Set<number>();
  let best = -Infinity;
  for (const i of order) {
    if (ys[i] > best) {
      front.add(i);
      best = ys[i];
    }
  }
  return front;
}

// ── Analysis context (ports the ctx dict built in main) ─────────────────────

export interface AnalysisCtx {
  nDays: number;
  scenarios: Scenario[];
  regions: [string, number][];
  dists: string[];
  strategies: string[];
  improvers: string[];
  constructors: string[];
  acceptance: string[];
  paretoPoints: "all" | "front";
}

export function buildCtx(rows: SimRow[], nDays: number, paretoPoints: "all" | "front" = "all"): AnalysisCtx {
  const scenarios = detectScenarios(rows);
  const regions: [string, number][] = [];
  const regionSeen = new Set<string>();
  for (const s of scenarios) {
    const key = `${s.city}|${s.N}`;
    if (!regionSeen.has(key)) {
      regionSeen.add(key);
      regions.push([s.city, s.N]);
    }
  }
  const uniqSorted = (vals: string[]) => [...new Set(vals)].sort();
  const constructorOrder = META.constructors.map(([, label]) => label);
  const constructors = [...new Set(rows.map((r) => r.constructor))].sort(
    (a, b) => constructorOrder.indexOf(a) - constructorOrder.indexOf(b)
  );
  return {
    nDays,
    scenarios,
    regions,
    dists: uniqSorted(rows.map((r) => r.dist)),
    strategies: uniqSorted(rows.map((r) => r.strategy)),
    improvers: uniqSorted(rows.map((r) => r.improver)),
    constructors,
    acceptance: uniqSorted(rows.map((r) => r.acceptance).filter(Boolean)),
    paretoPoints,
  };
}

// ── Markdown table builders (ports build_* / render_* helpers) ──────────────

export function buildParetoFrontTable(rows: SimRow[], ctx: AnalysisCtx): string {
  interface PfRow extends SimRow {
    _scenario: string;
  }
  const pareto: PfRow[] = [];
  for (const s of ctx.scenarios) {
    const sub = scenSub(rows, s);
    if (!sub.length) continue;
    const front = paretoIndices(metricValues(sub, "overflows"), metricValues(sub, "kgkm"));
    for (const idx of front) pareto.push({ ...sub[idx], _scenario: scenarioLabel(s) });
  }
  if (!pareto.length) return "_No Pareto-front data available._";

  const groups = new Map<string, PfRow[]>();
  for (const r of pareto) {
    const key = `${r.variant}|${r.constructor}|${r.improver}`;
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(r);
  }
  const tableRows = [...groups.values()].map((grp) => ({
    sel: grp[0].variant,
    con: grp[0].constructor,
    imp: grp[0].improver,
    ov: mean(grp.map((r) => r.overflows)),
    eff: mean(grp.map((r) => r.kgkm)),
    scenarios: [...new Set(grp.map((r) => r._scenario))].sort(),
  }));
  tableRows.sort(
    (a, b) =>
      b.scenarios.length - a.scenarios.length ||
      a.sel.localeCompare(b.sel) ||
      a.con.localeCompare(b.con) ||
      a.imp.localeCompare(b.imp)
  );
  const lines = [
    "| Selection | Constructor | Improver | Overflows | kg/km | Pareto-Front Scenarios |",
    "|-----------|-------------|----------|----------:|------:|------------------------|",
  ];
  for (const r of tableRows) {
    lines.push(
      `| ${r.sel} | ${r.con} | ${r.imp} | ${r.ov.toFixed(1)} | ${r.eff.toFixed(3)} | ${r.scenarios.join(", ")} |`
    );
  }
  return lines.join("\n");
}

export function buildKpiTable(
  dfm: SimRow[],
  ctx: AnalysisCtx,
  metric: SimMetric,
  decimals: number
): string {
  const { improvers } = ctx;
  const header = "| Config |" + improvers.map((i) => ` ${i} Min | ${i} Max | ${i} Mean |`).join("");
  const sep = "|--------|" + "-----|-----|------|".repeat(improvers.length);
  const lines = [header, sep];
  const fmt = (v: number) => v.toFixed(decimals);
  for (const s of ctx.scenarios) {
    for (const strat of ctx.strategies) {
      let anyData = false;
      const cells = improvers.map((imp) => {
        const vals = metricValues(
          scenSub(dfm.filter((r) => r.improver === imp && r.strategy === strat), s),
          metric
        );
        if (!vals.length) return " — | — | — |";
        anyData = true;
        return ` ${fmt(Math.min(...vals))} | ${fmt(Math.max(...vals))} | ${fmt(mean(vals))} |`;
      });
      if (anyData) lines.push(`| ${scenarioLabel(s)} / ${strat} |` + cells.join(""));
    }
  }
  return lines.join("\n");
}

export function buildStrategyBest(dfm: SimRow[], ctx: AnalysisCtx, placeholder: string): string {
  const lines: string[] = [];
  for (const strat of ctx.strategies) {
    lines.push(`#### ${strat} (${META.strategy_names[strat] ?? strat})`, "");
    for (const s of ctx.scenarios) {
      const sub = scenSub(dfm.filter((r) => r.strategy === strat), s);
      if (!sub.length) continue;
      const byCon = new Map<string, SimRow[]>();
      for (const r of sub) (byCon.get(r.constructor) ?? byCon.set(r.constructor, []).get(r.constructor)!).push(r);
      let bestOv = "";
      let bestOvVal = Infinity;
      let bestEff = "";
      let bestEffVal = -Infinity;
      for (const [con, grp] of byCon) {
        const ov = mean(grp.map((r) => r.overflows));
        const eff = mean(grp.map((r) => r.kgkm));
        if (ov < bestOvVal) {
          bestOvVal = ov;
          bestOv = con;
        }
        if (eff > bestEffVal) {
          bestEffVal = eff;
          bestEff = con;
        }
      }
      lines.push(
        `**${scenarioLabel(s)}:** best overflow: **${bestOv}** (${bestOvVal.toFixed(1)}); ` +
          `best efficiency: **${bestEff}** (${bestEffVal.toFixed(3)} kg/km).`,
        ""
      );
    }
    lines.push(placeholder, "");
  }
  return lines.join("\n");
}

// ── Full hierarchical results matrix (ports build_full_results_matrix etc.) ─

export type RowKey = [string, number, string];
export type ColKey = string[];

export interface ResultsMatrix {
  rowKeys: RowKey[];
  colKeys: ColKey[];
  cells: Map<string, string>;
}

export function cellKey(rk: RowKey, ck: ColKey): string {
  return `${rk.join("|")}::${ck.join("|")}`;
}

/** One matrix cell: mean±std overflows and kg/km — ports _fmt_result_cell. */
function fmtResultCell(sub: SimRow[]): string {
  if (!sub.length) return "—";
  const ov = sub.map((r) => r.overflows);
  const kg = sub.map((r) => r.kgkm);
  const ovS = mean(ov).toFixed(1) + (ov.length > 1 ? `±${std(ov).toFixed(1)}` : "");
  const kgS = mean(kg).toFixed(3) + (kg.length > 1 ? `±${std(kg).toFixed(3)}` : "");
  return `${ovS} ov<br>${kgS} kg/km`;
}

export function buildFullResultsMatrix(
  dfm: SimRow[],
  ctx: AnalysisCtx,
  horizonLabel?: string
): ResultsMatrix {
  const rowKeySet = new Map<string, RowKey>();
  for (const s of ctx.scenarios) {
    rowKeySet.set(`${s.city}|${s.N}|${s.dist}`, [s.city, s.N, s.dist]);
  }
  const rowKeys = [...rowKeySet.values()].sort(
    (a, b) => a[0].localeCompare(b[0]) || a[1] - b[1] || a[2].localeCompare(b[2])
  );
  const colKeys: ColKey[] = [];
  for (const strat of ctx.strategies) {
    for (const con of ctx.constructors) {
      for (const imp of ctx.improvers) {
        colKeys.push(horizonLabel ? [horizonLabel, strat, con, imp] : [strat, con, imp]);
      }
    }
  }
  const cells = new Map<string, string>();
  for (const rk of rowKeys) {
    const s: Scenario = { city: rk[0], N: rk[1], dist: rk[2] };
    for (const ck of colKeys) {
      const [strat, con, imp] = horizonLabel ? ck.slice(1) : ck;
      const sub = scenSub(
        dfm.filter((r) => r.strategy === strat && r.constructor === con && r.improver === imp),
        s
      );
      cells.set(cellKey(rk, ck), fmtResultCell(sub));
    }
  }
  return { rowKeys, colKeys, cells };
}

/** Render the hierarchical results matrix as a GFM pipe table — ports render_full_results_table_md. */
export function renderFullResultsTableMd(matrix: ResultsMatrix): string {
  const { rowKeys, colKeys, cells } = matrix;
  if (!rowKeys.length || !colKeys.length) return "_No data available._";
  const nRowLevels = rowKeys[0].length;
  const rowHeaders = ["Region", "N", "Distribution"].slice(0, nRowLevels);
  const colLabels = colKeys.map((ck) => ck.join("<br>"));
  const lines = [
    "| " + rowHeaders.join(" | ") + " | " + colLabels.join(" | ") + " |",
    "|" + "---|".repeat(nRowLevels + colKeys.length),
  ];
  let prev: RowKey | null = null;
  for (const rk of rowKeys) {
    const rowCells: string[] = [];
    for (let lvl = 0; lvl < nRowLevels; lvl++) {
      const same =
        prev !== null && prev.slice(0, lvl + 1).join("|") === rk.slice(0, lvl + 1).join("|");
      rowCells.push(same ? "" : String(rk[lvl]));
    }
    prev = rk;
    const dataCells = colKeys.map((ck) => cells.get(cellKey(rk, ck)) ?? "—");
    lines.push("| " + rowCells.join(" | ") + " | " + dataCells.join(" | ") + " |");
  }
  return lines.join("\n");
}

export interface HorizonData {
  days: number;
  rows: SimRow[];
  dfm: SimRow[];
  ctx: AnalysisCtx;
  nLogs: number;
}

export function buildFullResultsTableAllHorizons(horizons: HorizonData[]): string {
  const rowKeys: RowKey[] = [];
  const seen = new Set<string>();
  const colKeys: ColKey[] = [];
  const cells = new Map<string, string>();
  for (const h of horizons) {
    const m = buildFullResultsMatrix(h.dfm, h.ctx, `${h.days}d`);
    for (const rk of m.rowKeys) {
      const key = rk.join("|");
      if (!seen.has(key)) {
        seen.add(key);
        rowKeys.push(rk);
      }
    }
    colKeys.push(...m.colKeys);
    for (const [k, v] of m.cells) cells.set(k, v);
  }
  rowKeys.sort((a, b) => a[0].localeCompare(b[0]) || a[1] - b[1] || a[2].localeCompare(b[2]));
  return renderFullResultsTableMd({ rowKeys, colKeys, cells });
}

/** Consecutive runs of keys sharing a prefix up to `level` — ports _group_spans. */
export function groupSpans(keys: ColKey[], level: number): [number, number, string][] {
  const spans: [number, number, string][] = [];
  let start = 0;
  for (let i = 1; i <= keys.length; i++) {
    if (
      i === keys.length ||
      keys[i].slice(0, level + 1).join("|") !== keys[start].slice(0, level + 1).join("|")
    ) {
      spans.push([start, i, keys[start][level]]);
      start = i;
    }
  }
  return spans;
}

export { KGKM_LABEL };
