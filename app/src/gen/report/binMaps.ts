/**
 * Bin-location map figures (§H.2) — native port of `gen_bin_location_maps` /
 * `gen_selected_bin_maps` from `logic/gen/gen_simulation_analysis.py`.
 *
 * Renders lat/lon scatter maps of all/selected bins per city. The Python
 * "street" mode fetched an OSMnx basemap; natively the interactive deck.gl
 * Digital Twin covers street-level context, and report figures use the
 * scatter mode (no network access, deterministic output).
 */
import type { ChartSpec } from "../charts/common";
import { axisStyle } from "../charts/common";
import type { GenTheme } from "../config";
import { joinPath, loadCsv, pathExists } from "../io";

const COORD_DIR = "data/simulator/coordinates";

/** Recover a decimal point dropped during export (ports _fix_stripped_decimal). */
export function fixStrippedDecimal(val: number, lo: number, hi: number): number {
  const [alo, ahi] = [Math.abs(lo), Math.abs(hi)].sort((a, b) => a - b);
  const s = Math.abs(val);
  if (s >= alo && s <= ahi) return val;
  for (let k = 1; k < 18; k++) {
    const cand = s / 10 ** k;
    if (cand >= alo && cand <= ahi) return Math.sign(val) * cand;
  }
  return val;
}

export interface BinCoords {
  lat: number[];
  lon: number[];
}

const num = (v: unknown) => {
  const n = typeof v === "number" ? v : parseFloat(String(v ?? ""));
  return Number.isFinite(n) ? n : NaN;
};

/** Every known bin's (lat, lon) for a city (ports _load_bin_coords). */
export async function loadBinCoords(projectRoot: string, city: string): Promise<BinCoords | null> {
  let file: string;
  let fix = false;
  if (city === "Rio Maior") {
    file = joinPath(projectRoot, `${COORD_DIR}/out_info[riomaior].csv`);
  } else if (city === "Figueira da Foz") {
    file = joinPath(projectRoot, `${COORD_DIR}/out_info[figdafoz].csv`);
    fix = true;
  } else {
    return null;
  }
  if (!(await pathExists(file))) return null;
  const csv = await loadCsv(file);
  const seen = new Set<string>();
  const lat: number[] = [];
  const lon: number[] = [];
  for (const row of csv.rows) {
    const id = String(row.ID ?? row.id ?? `${row.Latitude}|${row.Longitude}`);
    if (seen.has(id)) continue;
    seen.add(id);
    let la = num(row.Latitude ?? row.lat);
    let lo = num(row.Longitude ?? row.lon);
    if (fix) {
      la = fixStrippedDecimal(la, 36, 43);
      lo = fixStrippedDecimal(lo, -10, -6);
    }
    if (Number.isFinite(la) && Number.isFinite(lo)) {
      lat.push(la);
      lon.push(lo);
    }
  }
  return lat.length ? { lat, lon } : null;
}

/** Selected-bin coordinates for a scenario file, when exported (ports gen_selected_bin_maps). */
export async function loadSelectedBinCoords(
  projectRoot: string,
  csvName: string
): Promise<BinCoords | null> {
  const file = joinPath(projectRoot, `${COORD_DIR}/${csvName}`);
  if (!(await pathExists(file))) return null;
  const csv = await loadCsv(file);
  const lat: number[] = [];
  const lon: number[] = [];
  for (const row of csv.rows) {
    const lower: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(row)) lower[k.toLowerCase()] = v;
    const la = num(lower.lat ?? lower.latitude);
    const lo = num(lower.lon ?? lower.longitude);
    if (Number.isFinite(la) && Number.isFinite(lo)) {
      lat.push(la);
      lon.push(lo);
    }
  }
  return lat.length ? { lat, lon } : null;
}

export function buildBinMapChart(coords: BinCoords, title: string, theme: GenTheme): ChartSpec {
  return {
    width: 1150,
    height: 800,
    background: "#ffffff",
    option: {
      backgroundColor: "#ffffff",
      title: {
        text: title,
        left: "center",
        top: 10,
        textStyle: { color: "#222222", fontSize: 15, fontWeight: "bold" },
      },
      grid: { left: 90, right: 40, top: 60, bottom: 70 },
      xAxis: {
        type: "value",
        scale: true,
        ...axisStyle({ ...theme, axisLabelColor: "#222222", gridColor: "#c8c8c8" }, { name: "Longitude" }),
      },
      yAxis: {
        type: "value",
        scale: true,
        inverse: true, // north-up orientation parity with the OSMnx plots
        ...axisStyle({ ...theme, axisLabelColor: "#222222", gridColor: "#c8c8c8" }, { name: "Latitude" }),
        nameGap: 52,
      },
      series: [
        {
          type: "scatter",
          data: coords.lon.map((lo, i) => [lo, coords.lat[i]]),
          symbolSize: 7,
          silent: true,
          itemStyle: { color: "#C0392B", opacity: 0.85, borderColor: "#ffffff", borderWidth: 0.4 },
          z: 5,
        },
      ],
    },
  };
}

export const SELECTED_MAP_SPECS: {
  city: string;
  nBins: number;
  csvName: string;
  outName: string;
  fallbackName: string;
}[] = [
  { city: "Rio Maior", nBins: 100, csvName: "out_selected[riomaior100].csv", outName: "riomaior100_selected_map.png", fallbackName: "riomaior_map.png" },
  { city: "Rio Maior", nBins: 170, csvName: "out_selected[riomaior170].csv", outName: "riomaior170_selected_map.png", fallbackName: "riomaior_map.png" },
  { city: "Figueira da Foz", nBins: 350, csvName: "out_selected[figdafoz350].csv", outName: "figueiradafoz350_selected_map.png", fallbackName: "figueiradafoz_map.png" },
];
