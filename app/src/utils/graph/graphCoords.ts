/**
 * Load geographic bin coordinates from graph JSON + area CSV files (§G.3.1).
 * Mirrors Python `load_focus_coords` / `process_data` row-selection semantics.
 */
import { invoke } from "@tauri-apps/api/core";
import type { BinCoord, DayLogEntry, SimDayData } from "../../types";

export interface GraphPreset {
  id: string;
  label: string;
  graphFile: string;
  coordFile: string;
  area: "riomaior" | "figueiradafoz";
  wasteType?: string;
  depotSigla: string;
}

export const GRAPH_PRESETS: GraphPreset[] = [
  {
    id: "rm-100",
    label: "RM-100 (Rio Maior)",
    graphFile: "data/simulator/bins_selection/graphs_100V_1N_plastic.json",
    coordFile: "data/simulator/coordinates/old_out_info[riomaior].csv",
    area: "riomaior",
    wasteType: "Mistura de embalagens",
    depotSigla: "CTEASO",
  },
  {
    id: "rm-170",
    label: "RM-170 (Rio Maior)",
    graphFile: "data/simulator/bins_selection/graphs_170V_1N_plastic.json",
    coordFile: "data/simulator/coordinates/old_out_info[riomaior].csv",
    area: "riomaior",
    wasteType: "Mistura de embalagens",
    depotSigla: "CTEASO",
  },
  {
    id: "ffz-350",
    label: "FFZ-350 (Figueira da Foz)",
    graphFile: "data/simulator/bins_selection/graphs_350V_1N_plastic.json",
    coordFile: "data/simulator/coordinates/out_info[figdafoz].csv",
    area: "figueiradafoz",
    wasteType: "Mistura de embalagens",
    depotSigla: "CITVRSU",
  },
];

interface CoordRow {
  lat: number;
  lng: number;
}

function joinPath(root: string, rel: string): string {
  const base = root.replace(/[/\\]+$/, "");
  const parts = rel.split("/");
  return [base, ...parts].join("/");
}

function parseGraphIndices(text: string, sampleIndex = 0): number[] {
  const parsed = JSON.parse(text) as number[] | number[][];
  if (Array.isArray(parsed[0])) return (parsed as number[][])[sampleIndex] ?? (parsed as number[][])[0];
  return parsed as number[];
}

function rowLatLng(row: Record<string, unknown>): CoordRow | null {
  const latRaw = row.Lat ?? row.lat ?? row.Latitude ?? row.latitude;
  const lngRaw = row.Lng ?? row.Lon ?? row.lon ?? row.Longitude ?? row.longitude;
  const lat = typeof latRaw === "number" ? latRaw : Number(latRaw);
  const lng = typeof lngRaw === "number" ? lngRaw : Number(lngRaw);
  if (Number.isNaN(lat) || Number.isNaN(lng)) return null;
  return { lat, lng };
}

function filterWasteRows(
  rows: Record<string, unknown>[],
  wasteType?: string
): Record<string, unknown>[] {
  if (!wasteType) return rows;
  const key = rows[0]?.["Tipo de Residuos"] != null ? "Tipo de Residuos" : "Tipo de resíduos";
  if (rows[0]?.[key] == null) return rows;
  return rows.filter((r) => String(r[key]) === wasteType);
}

async function loadCsvRows(path: string): Promise<Record<string, unknown>[]> {
  const file = await invoke<{ rows: Record<string, unknown>[] }>("load_csv_file", { path });
  return file.rows;
}

async function loadDepotCoords(
  projectRoot: string,
  sigla: string
): Promise<CoordRow | null> {
  const path = joinPath(projectRoot, "data/simulator/coordinates/Facilities.csv");
  const rows = await loadCsvRows(path);
  const facility = rows.find((r) => String(r.Sigla) === sigla);
  return facility ? rowLatLng(facility) : null;
}

/** Build processed coordinate table: index 0 = depot, 1..N = bins sorted by ID. */
export async function buildGraphCoordTable(
  projectRoot: string,
  preset: GraphPreset,
  sampleIndex = 0
): Promise<CoordRow[]> {
  const graphPath = joinPath(projectRoot, preset.graphFile);
  const coordPath = joinPath(projectRoot, preset.coordFile);

  const [graphText, coordRows, depot] = await Promise.all([
    invoke<string>("read_text_file", { path: graphPath }),
    loadCsvRows(coordPath),
    loadDepotCoords(projectRoot, preset.depotSigla),
  ]);

  const indices = parseGraphIndices(graphText, sampleIndex);
  const filtered = filterWasteRows(coordRows, preset.wasteType);

  const withIds = indices
    .map((i) => filtered[i])
    .filter(Boolean)
    .map((row) => ({
      id: Number(row.ID ?? row.id ?? 0),
      coord: rowLatLng(row)!,
    }))
    .filter((x) => x.coord != null)
    .sort((a, b) => a.id - b.id);

  const table: CoordRow[] = [];
  if (depot) table.push(depot);
  for (const { coord } of withIds) table.push(coord);
  if (table.length <= 1) throw new Error("Could not resolve graph coordinates");
  return table;
}

function guiIdToNodeIdx(id: number): number {
  return id === -1 ? 0 : id + 1;
}

export function enrichSimDayData(data: SimDayData, coordTable: CoordRow[]): SimDayData {
  if (!data.all_bin_coords?.length || !coordTable.length) return data;

  const enrichedBins: BinCoord[] = data.all_bin_coords.map((bin) => {
    if (bin.lat != null && bin.lng != null) return bin;
    const nodeIdx = guiIdToNodeIdx(bin.id);
    const coord = coordTable[nodeIdx];
    if (!coord) return bin;
    return { ...bin, lat: coord.lat, lng: coord.lng };
  });

  return { ...data, all_bin_coords: enrichedBins };
}

export function enrichEntriesWithGraphCoords(
  entries: DayLogEntry[],
  coordTable: CoordRow[]
): DayLogEntry[] {
  return entries.map((e) => ({
    ...e,
    data: enrichSimDayData(e.data, coordTable),
  }));
}

export async function loadGraphCoordinates(
  projectRoot: string,
  presetId: string,
  sampleIndex = 0
): Promise<CoordRow[]> {
  const preset = GRAPH_PRESETS.find((p) => p.id === presetId);
  if (!preset) throw new Error(`Unknown graph preset: ${presetId}`);
  return buildGraphCoordTable(projectRoot, preset, sampleIndex);
}

/** Infer graph preset from log path segments or bin count in day-1 coords. */
export function guessGraphPreset(
  logPath?: string | null,
  entries?: DayLogEntry[]
): string | null {
  const path = (logPath ?? "").toLowerCase();
  if (path.includes("figueiradafoz") || path.includes("figdafoz") || path.includes("ffz")) {
    if (path.includes("350") || !path.match(/\d{2,3}/)) return "ffz-350";
  }
  if (path.includes("riomaior") || path.includes("rio_maior") || path.includes("rm")) {
    if (path.includes("170")) return "rm-170";
    if (path.includes("100")) return "rm-100";
  }
  if (path.includes("170")) return "rm-170";
  if (path.includes("350")) return "ffz-350";
  if (path.includes("100")) return "rm-100";

  if (entries?.length) {
    const day1 = entries.find((e) => e.day === 1 && e.data.all_bin_coords?.length);
    const n = day1?.data.all_bin_coords?.filter((b) => b.id >= 0).length ?? 0;
    if (n >= 320) return "ffz-350";
    if (n >= 150) return "rm-170";
    if (n >= 80) return "rm-100";
  }
  return null;
}
