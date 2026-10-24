/**
 * Markdown and hierarchical matrix table builders for simulation results (§H.1).
 */
import { META } from "../config";
import type { SimRow, Scenario, SimMetric, AnalysisCtx } from "./simulation";
import { scenarioLabel, scenSub, metricValues, mean, std, paretoIndices } from "./simulation";

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
