/**
 * Procedural conceptual illustrations for the results deck (§H.3) — native
 * SVG ports of the `generate_*_image` builders and the fetch fallback in
 * `logic/gen/gen_presentation.py` (see config/referenceLinks.json for the
 * third-party references the native versions replace).
 */
import { GEN_IMAGES, assetToDataUrl } from "../assets";
import { REFERENCE_LINKS } from "../config";
import { SvgCanvas, svgToPngDataUrl, mulberry32, normal } from "./svg";

const ACCENT = "#2E74B5";
const DARK = "#1F2D3D";
const MUTED_HEX = "#8A9BB0";
const GREEN = "#3E8E41";
const ORANGE = "#B06A2E";
const RED = "#C0392B";
const FAINT = "#F0F4FA";

export type ImageMode = "native" | "fetch";

// ── QA route illustration (ports generate_qa_route_image; seeded) ────────────

export async function qaRouteIllustration(seed = 42, nNodes = 45): Promise<string> {
  const S = 900;
  const svg = new SvgCanvas(S, S);
  const rng = mulberry32(seed);
  const coords: [number, number][] = Array.from({ length: nNodes }, () => [rng(), rng()]);
  const depot: [number, number] = [0.5, 0.5];
  const px = (v: number) => 60 + v * (S - 120);
  const py = (v: number) => S - 110 - v * (S - 170);

  const angles = coords.map(([x, y]) => Math.atan2(y - depot[1], x - depot[0]));
  const dists = coords.map(([x, y]) => Math.hypot(x - depot[0], y - depot[1]));

  const routes: number[][] = [[], [], []];
  const unvisited: number[] = [];
  for (let i = 0; i < nNodes; i++) {
    if (i % 3 === 0 && dists[i] > 0.15) {
      unvisited.push(i);
      continue;
    }
    const a = angles[i];
    if (a < -Math.PI / 3) routes[0].push(i);
    else if (a < Math.PI / 3) routes[1].push(i);
    else routes[2].push(i);
  }
  for (const r of routes) r.sort((a, b) => angles[a] - angles[b]);

  // faint road-network edges
  for (let i = 0; i < nNodes; i++) {
    for (let j = i + 1; j < nNodes; j++) {
      if (Math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) < 0.22) {
        svg.line(px(coords[i][0]), py(coords[i][1]), px(coords[j][0]), py(coords[j][1]), {
          stroke: "#E2E8F0",
          width: 1,
        });
      }
    }
  }
  const routeColors = ["#10B981", "#6366F1", "#F59E0B"];
  routes.forEach((r, ri) => {
    if (!r.length) return;
    const tour: [number, number][] = [depot, ...r.map((i) => coords[i]), depot];
    svg.polyline(tour.map(([x, y]) => [px(x), py(y)] as [number, number]), {
      stroke: routeColors[ri],
      width: 4,
    });
  });
  for (const i of unvisited) {
    svg.circle(px(coords[i][0]), py(coords[i][1]), 9, { fill: "#CBD5E1", stroke: "#64748B" });
  }
  routes.forEach((r, ri) => {
    for (const i of r) {
      svg.circle(px(coords[i][0]), py(coords[i][1]), 10, { fill: routeColors[ri], stroke: "#000000" });
    }
  });
  svg.rect(px(depot[0]) - 13, py(depot[1]) - 13, 26, 26, { fill: "#EF4444", stroke: "#000", strokeWidth: 2 });

  // legend band
  const legendY = S - 55;
  const entries: [string, string, "circle" | "rect"][] = [
    ["Vehicle Route 1", routeColors[0], "circle"],
    ["Vehicle Route 2", routeColors[1], "circle"],
    ["Vehicle Route 3", routeColors[2], "circle"],
    ["Unvisited Bins (No Profit)", "#CBD5E1", "circle"],
    ["Depot / Warehouse", "#EF4444", "rect"],
  ];
  svg.roundRect(40, legendY - 26, S - 80, 62, { fill: "#ffffff", stroke: "#E2E8F0" });
  entries.forEach(([label, color, shape], i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 80 + col * 280;
    const y = legendY - 8 + row * 28;
    if (shape === "circle") svg.circle(x, y, 8, { fill: color, stroke: "#000" });
    else svg.rect(x - 8, y - 8, 16, 16, { fill: color, stroke: "#000" });
    svg.text(x + 16, y, label, { size: 15, anchor: "start", color: DARK });
  });
  return svgToPngDataUrl(svg.toString(), S, S);
}

// ── B&B tree (ports generate_bb_tree_image) ──────────────────────────────────

export async function bbTreeIllustration(): Promise<string> {
  const W = 1220;
  const H = 900;
  const svg = new SvgCanvas(W, H);
  const X = (v: number) => (v / 11.6) * W;
  const Y = (v: number) => H - (v / 10) * H;
  const nodes: Record<string, [number, number, string, string]> = {
    root: [5.6, 9, "LP Relaxation\n(+ cuts)", DARK],
    l1: [2.3, 6.2, "Branch: x_ij = 0", ACCENT],
    r1: [8.6, 6.2, "Branch: x_ij = 1", ACCENT],
    l2: [2.3, 3.2, "Pruned\n(bound ≤ incumbent)", MUTED_HEX],
    r2a: [6.9, 3.2, "Branch again", ACCENT],
    r2b: [10.2, 3.2, "Integer feasible\nnew incumbent", GREEN],
    leaf1: [5.3, 0.9, "Pruned\n(infeasible)", MUTED_HEX],
    leaf2: [8.7, 0.9, "Optimal route set", GREEN],
  };
  const edges: [string, string][] = [
    ["root", "l1"], ["root", "r1"], ["l1", "l2"], ["r1", "r2a"], ["r1", "r2b"],
    ["r2a", "leaf1"], ["r2a", "leaf2"],
  ];
  for (const [a, b] of edges) {
    svg.line(X(nodes[a][0]), Y(nodes[a][1]) + 32, X(nodes[b][0]), Y(nodes[b][1]) - 32, {
      stroke: MUTED_HEX,
      width: 1.6,
    });
  }
  for (const [x, y, label, color] of Object.values(nodes)) {
    svg.roundRect(X(x) - 135, Y(y) - 40, 270, 80, { fill: color, stroke: "#fff", rx: 12 });
    svg.text(X(x), Y(y), label, { size: 17, bold: true, color: "#fff" });
  }
  svg.text(X(5.6), 24, "Branch-and-Bound (BPC / SWC-TCF)", { size: 24, bold: true, color: DARK });
  return svgToPngDataUrl(svg.toString(), W, H);
}

// ── Local-search operators (ports generate_ls_operators_image) ───────────────

export async function lsOperatorsIllustration(): Promise<string> {
  const PW = 430;
  const PH = 330;
  const W = PW * 3;
  const H = PH * 2 + 60;
  const svg = new SvgCanvas(W, H);
  svg.text(W / 2, 26, "Classical Local Search Neighbourhood Moves", { size: 24, bold: true, color: DARK });

  const panel = (col: number, row: number) => {
    const ox = col * PW + 45;
    const oy = 60 + row * PH + 20;
    const sw = PW - 90;
    const sh = PH - 80;
    return {
      X: (v: number) => ox + v * sw,
      Y: (v: number) => oy + sh - v * sh,
      titleXY: [col * PW + PW / 2, 60 + row * PH + 4] as [number, number],
    };
  };
  const drawPts = (p: ReturnType<typeof panel>, pts: [number, number][], labels: string[], colors?: string[]) => {
    pts.forEach(([x, y], i) => {
      svg.circle(p.X(x), p.Y(y), 11, { fill: colors?.[i] ?? DARK, stroke: "#fff", strokeWidth: 2 });
      svg.text(p.X(x), p.Y(y) - 24, labels[i], { size: 15, bold: true, color: DARK });
    });
  };
  const seg = (p: ReturnType<typeof panel>, a: [number, number], b: [number, number], color: string, dash = false) =>
    svg.line(p.X(a[0]), p.Y(a[1]), p.X(b[0]), p.Y(b[1]), {
      stroke: color,
      width: dash ? 2 : 4,
      dash: dash ? "6,5" : undefined,
    });

  // 2-opt
  const pts2: [number, number][] = [[0.1, 0.15], [0.85, 0.75], [0.75, 0.1], [0.15, 0.85]];
  let p = panel(0, 0);
  svg.text(...p.titleXY, "2-opt — before", { size: 16, bold: true, color: RED });
  seg(p, pts2[0], pts2[1], RED);
  seg(p, pts2[2], pts2[3], RED);
  seg(p, pts2[1], pts2[2], MUTED_HEX, true);
  seg(p, pts2[3], pts2[0], MUTED_HEX, true);
  drawPts(p, pts2, ["A", "B", "C", "D"]);
  p = panel(0, 1);
  svg.text(...p.titleXY, "2-opt — after: A–C, B–D", { size: 16, bold: true, color: GREEN });
  seg(p, pts2[0], pts2[2], GREEN);
  seg(p, pts2[1], pts2[3], GREEN);
  seg(p, pts2[1], pts2[2], MUTED_HEX, true);
  seg(p, pts2[3], pts2[0], MUTED_HEX, true);
  drawPts(p, pts2, ["A", "B", "C", "D"]);

  // swap
  const r1: [number, number][] = [[0.05, 0.85], [0.4, 0.85], [0.75, 0.85]];
  const r2: [number, number][] = [[0.05, 0.15], [0.4, 0.15], [0.75, 0.15]];
  p = panel(1, 0);
  svg.text(...p.titleXY, "Swap — before", { size: 16, bold: true, color: RED });
  svg.polyline(r1.map(([x, y]) => [p.X(x), p.Y(y)] as [number, number]), { stroke: ACCENT, width: 4 });
  svg.polyline(r2.map(([x, y]) => [p.X(x), p.Y(y)] as [number, number]), { stroke: ORANGE, width: 4 });
  drawPts(p, [...r1, ...r2], ["A", "B", "C", "D", "E", "F"], [ACCENT, ACCENT, ACCENT, ORANGE, ORANGE, ORANGE]);
  p = panel(1, 1);
  svg.text(...p.titleXY, "Swap — after: B ↔ E", { size: 16, bold: true, color: GREEN });
  const r1s: [number, number][] = [r1[0], r2[1], r1[2]];
  const r2s: [number, number][] = [r2[0], r1[1], r2[2]];
  svg.polyline(r1s.map(([x, y]) => [p.X(x), p.Y(y)] as [number, number]), { stroke: ACCENT, width: 4 });
  svg.polyline(r2s.map(([x, y]) => [p.X(x), p.Y(y)] as [number, number]), { stroke: ORANGE, width: 4 });
  drawPts(p, [...r1s, ...r2s], ["A", "E", "C", "D", "B", "F"], [ACCENT, ACCENT, ACCENT, ORANGE, ORANGE, ORANGE]);

  // relocate
  const before: [number, number][] = [[0.05, 0.5], [0.35, 0.85], [0.65, 0.15], [0.95, 0.5]];
  p = panel(2, 0);
  svg.text(...p.titleXY, "Relocate — before", { size: 16, bold: true, color: RED });
  svg.polyline(before.map(([x, y]) => [p.X(x), p.Y(y)] as [number, number]), { stroke: RED, width: 4 });
  drawPts(p, before, ["A", "B", "C", "D"]);
  p = panel(2, 1);
  svg.text(...p.titleXY, "Relocate — after: B moved", { size: 16, bold: true, color: GREEN });
  const after: [number, number][] = [[0.05, 0.5], [0.65, 0.15], [0.35, 0.85], [0.95, 0.5]];
  svg.polyline(after.map(([x, y]) => [p.X(x), p.Y(y)] as [number, number]), { stroke: GREEN, width: 4 });
  drawPts(p, after, ["A", "C", "B", "D"]);

  return svgToPngDataUrl(svg.toString(), W, H);
}

// ── Knapsack (ports generate_knapsack_image) ─────────────────────────────────

export async function knapsackIllustration(): Promise<string> {
  const W = 1110;
  const H = 990;
  const svg = new SvgCanvas(W, H);
  const X = (v: number) => (v / 11.6) * W;
  const Y = (v: number) => H - (v / 10) * H;
  svg.text(W / 2, 28, "Mandatory Selection ≈ a Knapsack Problem", { size: 24, bold: true, color: DARK });

  svg.roundRect(X(0.6), Y(6.8), X(4.6) - X(0.6), Y(1.6) - Y(6.8), {
    fill: FAINT,
    stroke: DARK,
    strokeWidth: 3,
    rx: 26,
  });
  svg.text(X(2.6), Y(7.15), "Today's Route\n(capacity Q)", { size: 18, bold: true, color: DARK });

  const itemsIn: [string, string][] = [
    ["Bin 1 — 90% full", GREEN],
    ["Bin 2 — 85% full", GREEN],
    ["Bin 3 — 70% full", GREEN],
  ];
  itemsIn.forEach(([label, color], i) => {
    const y = 2.3 + i * 1.5;
    svg.roundRect(X(1.0), Y(y + 1.1), X(4.2) - X(1.0), Y(y) - Y(y + 1.1), { fill: color, stroke: "#fff", rx: 12 });
    svg.text(X(2.6), Y(y + 0.55), label, { size: 16, bold: true, color: "#fff" });
  });
  const itemsOut = ["Bin 4 — 30% full", "Bin 5 — 20% full", "Bin 6 — 15% full"];
  itemsOut.forEach((label, i) => {
    const x = 6.4 + (i % 2) * 2.5;
    const y = 7.4 - Math.floor(i / 2) * 1.6;
    svg.roundRect(X(x), Y(y + 1.1), X(x + 2.1) - X(x), Y(y) - Y(y + 1.1), { fill: "#CBD5E1", stroke: "#5A6A7A", rx: 12 });
    svg.text(X(x + 1.05), Y(y + 0.55), label, { size: 14, bold: true, color: "#333333" });
  });
  svg.text(X(5.8), Y(0.9), "maximise Σ value (waste kg)  subject to  Σ weight (fill/urgency) ≤ today's budget", {
    size: 16,
    italic: true,
    color: "#5A6A7A",
  });
  return svgToPngDataUrl(svg.toString(), W, H);
}

// ── Framework objective (ports generate_framework_objective_image) ───────────

export async function frameworkObjectiveIllustration(): Promise<string> {
  const W = 930;
  const H = 990;
  const svg = new SvgCanvas(W, H);
  const X = (v: number) => (v / 10) * W;
  const Y = (v: number) => H - (v / 10) * H;
  svg.text(W / 2, 28, "Objective: One Framework to Compare Them All", { size: 22, bold: true, color: DARK });

  const algos: [string, string][] = [
    ["Exact\nMethods", ACCENT],
    ["Meta-\nHeuristics", GREEN],
    ["Hyper-\nHeuristics", ORANGE],
  ];
  algos.forEach(([label, color], i) => {
    const y = 8.4 - i * 1.7;
    svg.roundRect(X(0.4), Y(y + 1.15), X(3.0) - X(0.4), Y(y) - Y(y + 1.15), { fill: color, stroke: "#fff", rx: 14 });
    svg.text(X(1.7), Y(y + 0.575), label, { size: 17, bold: true, color: "#fff" });
    svg.line(X(3.0), Y(y + 0.575), X(4.35), Y(5.2), { stroke: MUTED_HEX, width: 2.4, arrow: true });
  });
  svg.roundRect(X(4.4), Y(6.2), X(7.4) - X(4.4), Y(4.2) - Y(6.2), { fill: DARK, stroke: "#fff", rx: 18 });
  svg.text(X(5.9), Y(5.2), "One Shared\nSimulator", { size: 19, bold: true, color: "#fff" });
  svg.line(X(7.4), Y(5.2), X(7.87), Y(5.2), { stroke: MUTED_HEX, width: 3, arrow: true });
  svg.roundRect(X(7.9), Y(6.2), X(9.6) - X(7.9), Y(4.2) - Y(6.2), { fill: "#5A6A7A", stroke: "#fff", rx: 18 });
  svg.text(X(8.75), Y(5.2), "Fair\nBenchmark", { size: 16, bold: true, color: "#fff" });
  svg.text(X(5.9), Y(1.6), "Same scenarios, same KPIs (overflows, KG / KM)\nfor every exact / meta- / hyper-heuristic method", {
    size: 16,
    italic: true,
    color: "#5A6A7A",
  });
  return svgToPngDataUrl(svg.toString(), W, H);
}

// ── Meta-heuristic landscape (ports generate_metaheuristic_overview_image) ───

export async function metaheuristicOverviewIllustration(): Promise<string> {
  const W = 1290;
  const H = 600;
  const svg = new SvgCanvas(W, H);
  svg.text(W / 2, 26, "Meta-Heuristics: Explore the Landscape, Exploit the Best Region", {
    size: 21,
    bold: true,
    color: DARK,
  });
  const n = 400;
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = (10 * i) / (n - 1);
    xs.push(x);
    ys.push(3 + 1.4 * Math.sin(x * 1.3) + 0.6 * Math.sin(x * 3.7 + 1) - 0.12 * (x - 5) ** 2 * 0.15);
  }
  const ymin = Math.min(...ys) - 0.5;
  const ymax = Math.max(...ys) + 1.6;
  const X = (v: number) => 40 + (v / 10) * (W - 80);
  const Y = (v: number) => 60 + (1 - (v - ymin) / (ymax - ymin)) * (H - 130);
  const linePts = xs.map((x, i) => [X(x), Y(ys[i])] as [number, number]);
  svg.polygon([[X(0), Y(ymin)], ...linePts, [X(10), Y(ymin)]], { fill: FAINT });
  svg.polyline(linePts, { stroke: DARK, width: 3 });

  const [iS, iM, iE] = [40, 180, 305];
  svg.circle(X(xs[iS]), Y(ys[iS]), 11, { fill: ORANGE, stroke: "#fff", strokeWidth: 2 });
  svg.text(X(xs[iS]), Y(ys[iS]) - 26, "start", { size: 15, bold: true, color: ORANGE });
  svg.curve(X(xs[iS]), Y(ys[iS] + 0.15), X(xs[iM]), Y(ys[iM] + 0.15), 0.12, { stroke: RED, width: 2.6, arrow: true });
  svg.text(X((xs[iS] + xs[iM]) / 2), Y(ys[iM] + 1.35), "explore\n(escape local optima)", {
    size: 14,
    bold: true,
    color: RED,
  });
  svg.circle(X(xs[iE]), Y(ys[iE]), 13, { fill: GREEN, stroke: "#fff", strokeWidth: 2 });
  svg.curve(X(xs[iM]), Y(ys[iM] + 0.1), X(xs[iE]), Y(ys[iE] + 0.1), -0.08, { stroke: GREEN, width: 2.6, arrow: true });
  svg.text(X((xs[iM] + xs[iE]) / 2), Y(ys[iE] - 1.0), "exploit\n(intensify the best region)", {
    size: 14,
    bold: true,
    color: GREEN,
  });
  return svgToPngDataUrl(svg.toString(), W, H);
}

// ── Hyper-heuristic controller (ports generate_hyperheuristic_overview_image) ─

export async function hyperheuristicOverviewIllustration(): Promise<string> {
  const W = 1290;
  const H = 600;
  const svg = new SvgCanvas(W, H);
  const X = (v: number) => (v / 10) * W;
  const Y = (v: number) => H - (v / 8) * H;
  svg.text(W / 2, 26, "Hyper-Heuristics: Choosing Among Heuristics", { size: 21, bold: true, color: DARK });

  svg.roundRect(X(3.5), Y(7.0), X(6.5) - X(3.5), Y(5.6) - Y(7.0), { fill: ORANGE, stroke: "#fff", rx: 16 });
  svg.text(X(5.0), Y(6.3), "Hyper-Heuristic\nController (ACO-HH)", { size: 17, bold: true, color: "#fff" });

  const heuristics = ["H1: 2-opt", "H2: relocate", "H3: swap", "H4: or-opt"];
  heuristics.forEach((h, i) => {
    const x = 0.6 + i * 2.35;
    svg.roundRect(X(x), Y(3.7), X(x + 2.0) - X(x), Y(2.6) - Y(3.7), { fill: ACCENT, stroke: "#fff", rx: 12 });
    svg.text(X(x + 1.0), Y(3.15), h, { size: 15, bold: true, color: "#fff" });
    svg.line(X(5.0), Y(5.6), X(x + 1.0), Y(3.72), { stroke: MUTED_HEX, width: 2.2, arrow: true });
  });
  svg.curve(X(6.8), Y(1.9), X(6.8), Y(5.55), -0.35, { stroke: GREEN, width: 2.6, arrow: true });
  svg.text(X(8.9), Y(3.8), "pheromone /\nreward feedback\non each heuristic", { size: 13.5, bold: true, color: GREEN });
  svg.text(X(5.0), Y(0.9), "Picks which low-level heuristic to apply next — searches\nthe space of heuristics, not the space of solutions", {
    size: 15,
    italic: true,
    color: "#5A6A7A",
  });
  return svgToPngDataUrl(svg.toString(), W, H);
}

// ── Trajectory search (ports generate_trajectory_overview_image; seeded) ─────

export async function trajectoryOverviewIllustration(seed = 7): Promise<string> {
  const W = 1290;
  const H = 600;
  const svg = new SvgCanvas(W, H);
  svg.text(W / 2, 26, "Trajectory-Based: SANS, PG-CLNS", { size: 21, bold: true, color: DARK });
  const rng = normal(mulberry32(seed));
  const steps = 9;
  const xs = Array.from({ length: steps }, (_, i) => 0.5 + (9 * i) / (steps - 1));
  const ys: number[] = [];
  let acc = 4;
  for (let i = 0; i < steps; i++) {
    acc += rng() * 0.5;
    ys.push(acc);
  }
  ys[steps - 1] = Math.min(...ys.slice(0, -1)) - 0.4;
  const ymin = Math.min(...ys) - 1.8;
  const ymax = Math.max(...ys) + 1;
  const X = (v: number) => 40 + (v / 10) * (W - 80);
  const Y = (v: number) => 60 + (1 - (v - ymin) / (ymax - ymin)) * (H - 130);
  svg.polyline(xs.map((x, i) => [X(x), Y(ys[i])] as [number, number]), { stroke: MUTED_HEX, width: 2 });
  for (let i = 0; i < steps - 2; i++) {
    if (ys[i + 1] > ys[i]) {
      svg.line(X(xs[i]), Y(ys[i]), X(xs[i + 1]), Y(ys[i + 1]), { stroke: RED, width: 1.8, arrow: true, opacity: 0.7 });
    }
  }
  for (let i = 0; i < steps - 1; i++) {
    svg.circle(X(xs[i]), Y(ys[i]), 9, { fill: ORANGE, stroke: "#fff", strokeWidth: 1.6 });
  }
  svg.circle(X(xs[steps - 1]), Y(ys[steps - 1]), 12, { fill: GREEN, stroke: "#fff", strokeWidth: 2 });
  svg.text(X(xs[0]) - 8, Y(ys[0]) - 24, "incumbent", { size: 14, bold: true, color: ORANGE, anchor: "start" });
  svg.text(X(xs[steps - 1]), Y(ys[steps - 1]) - 26, "best found", { size: 14, bold: true, color: GREEN });
  svg.text(X(5), Y(ymin + 0.5), "one solution moves step by step; occasional uphill\nmoves escape local optima (e.g. simulated annealing)", {
    size: 15,
    italic: true,
    color: "#5A6A7A",
  });
  return svgToPngDataUrl(svg.toString(), W, H);
}

// ── Population search (ports generate_population_overview_image; seeded) ─────

export async function populationOverviewIllustration(seed = 3): Promise<string> {
  const W = 1290;
  const H = 600;
  const svg = new SvgCanvas(W, H);
  svg.text(W / 2, 26, "Population-Based: HGS, PSOMA", { size: 21, bold: true, color: DARK });
  const rng = normal(mulberry32(seed));
  const X = (v: number) => (v / 10) * W;
  const Y = (v: number) => H - (v / 8) * H;
  const gens = [1.5, 5.0, 8.5];
  const genLabels = ["Generation t", "Generation t+1", "Generation t+2"];
  const spread = [1.6, 1.0, 0.55];
  const colors = [ORANGE, ACCENT, GREEN];
  gens.forEach((gx, gi) => {
    for (let i = 0; i < 8; i++) {
      const y = 4 + rng() * spread[gi];
      const x = gx + rng() * 0.15;
      svg.circle(X(x), Y(y), 8, { fill: colors[gi], stroke: "#fff", strokeWidth: 1.5, opacity: 0.85 });
    }
    svg.text(X(gx), Y(7.2), genLabels[gi], { size: 15, bold: true, color: colors[gi] });
    if (gi < gens.length - 1) {
      svg.line(X(gx + 0.6), Y(4), X(gens[gi + 1] - 0.6), Y(4), { stroke: MUTED_HEX, width: 2.6, arrow: true });
    }
  });
  svg.text(X(5), Y(0.6), "selection + crossover + mutation narrow the population\ntoward better solutions each generation", {
    size: 15,
    italic: true,
    color: "#5A6A7A",
  });
  return svgToPngDataUrl(svg.toString(), W, H);
}

// ── VRPP illustration (bundled source image; fetch mode uses the link) ───────

export async function vrppIllustration(): Promise<string> {
  return assetToDataUrl(GEN_IMAGES.vrpp_illustration_source);
}

// ── Registry (ports NATIVE_DIAGRAM_BUILDERS + ensure_reference_images) ───────

export const ILLUSTRATION_BUILDERS: Record<string, () => Promise<string>> = {
  "qa_route_illustration.png": () => qaRouteIllustration(),
  "bb_tree.png": bbTreeIllustration,
  "ls_operators.png": lsOperatorsIllustration,
  "knapsack_illustration.png": knapsackIllustration,
  "framework_objective.png": frameworkObjectiveIllustration,
  "metaheuristic_overview.png": metaheuristicOverviewIllustration,
  "hyperheuristic_overview.png": hyperheuristicOverviewIllustration,
  "trajectory_overview.png": trajectoryOverviewIllustration,
  "population_overview.png": populationOverviewIllustration,
  "vrpp_illustration.png": vrppIllustration,
};

const FETCH_KEYS: Record<string, string> = {
  "bb_tree.png": "bb_tree",
  "ls_operators.png": "ls_operators",
  "population_overview.png": "population_overview",
  "vrpp_illustration.png": "vrpp_illustration",
};

/** Resolve a conceptual illustration as a PNG data URL (native or fetched). */
export async function resolveIllustration(name: string, mode: ImageMode): Promise<string | null> {
  if (mode === "fetch" && FETCH_KEYS[name]) {
    const link = REFERENCE_LINKS[FETCH_KEYS[name]];
    if (link) {
      try {
        const blob = await (await fetch(link.url)).blob();
        return await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result));
          reader.onerror = () => reject(reader.error);
          reader.readAsDataURL(blob);
        });
      } catch {
        // fall through to the native builder
      }
    }
  }
  const builder = ILLUSTRATION_BUILDERS[name];
  return builder ? await builder() : null;
}
