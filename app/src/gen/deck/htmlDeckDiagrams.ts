/**
 * HTML deck stylesheets and native HTML/CSS diagram generators (§H.6).
 */
import { GEN_IMAGES, assetToDataUrl } from "../assets";

export const ACCENT = "#2E74B5";
export const DARK = "#1F2D3D";
export const MUTED = "#5A6A7A";
export const LIGHT_TXT = "#C9D6E4";
export const GREEN = "#3E8E41";
export const ORANGE = "#B06A2E";
export const LIGHT_FILL = "#F0F4FA";

export const esc = (s: string) =>
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

export function pipelineHtml(): string {
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

export function policyGridHtml(): string {
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

export function taxonomyHtml(): string {
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

export function doeTreeHtml(): string {
  const horizon = (label: string) =>
    `<div class="doe-h"><div class="doe-hbox">${esc(label)}</div><div class="doe-scen">${["RM-100", "RM-170", "FFZ-350"]
      .map((sc) => `<div class="doe-s"><div class="doe-sbox">${sc}</div><div class="doe-d">Empirical / Gamma-3</div></div>`)
      .join("")}</div></div>`;
  return `<div class="doe"><div class="doe-root">Simulation Runs</div><div class="doe-row">${horizon("30 Days")}${horizon("90 Days (Pareto-front policies only)")}</div></div>`;
}

export async function simulatorSceneHtml(): Promise<string> {
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
