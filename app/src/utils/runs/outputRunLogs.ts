/**
 * Discover simulation JSONL logs in the simulation output folders (§G.1.4 portfolio).
 */

import { invoke } from "@tauri-apps/api/core";
import type { DayLogEntry, DirEntry, OutputDir } from "../../types";

const MAX_JSONL_BYTES = 20 * 1024 * 1024;

/** Find the first .jsonl log in a run directory (top-level or hydra/). */
export async function findRunJsonl(runPath: string): Promise<string | null> {
  const top = await invoke<DirEntry[]>("list_dir", { path: runPath });
  const topJsonl = top.find(
    (f) => !f.is_dir && f.extension === "jsonl" && f.size_bytes < MAX_JSONL_BYTES
  );
  if (topJsonl) return topJsonl.path;

  const hydra = top.find((f) => f.is_dir && f.name === "hydra");
  if (hydra) {
    const sub = await invoke<DirEntry[]>("list_dir", { path: hydra.path });
    const nested = sub.find(
      (f) => !f.is_dir && f.extension === "jsonl" && f.size_bytes < MAX_JSONL_BYTES
    );
    if (nested) return nested.path;
  }
  return null;
}

export interface OutputLogRef {
  path: string;
  label: string;
}

export const PORTFOLIO_SCAN_DEFAULT = 480;
export const PORTFOLIO_SCAN_LEGACY = 48;

/** Collect JSONL paths from the run folders of one or more output roots (capped for Wasm/UI budget). */
export async function scanOutputPortfolio(
  outputPaths: string | string[],
  limit = PORTFOLIO_SCAN_DEFAULT
): Promise<OutputLogRef[]> {
  const refs: OutputLogRef[] = [];

  for (const outputPath of Array.isArray(outputPaths) ? outputPaths : [outputPaths]) {
    const dirs = await invoke<OutputDir[]>("list_output_dirs", { outputPath });
    for (const dir of dirs) {
      if (refs.length >= limit) return refs;
      const jsonl = await findRunJsonl(dir.path);
      if (jsonl) refs.push({ path: jsonl, label: dir.name });
    }
  }

  return refs;
}

export interface PortfolioRun extends OutputLogRef {
  entries: DayLogEntry[];
}

/** Load simulation logs in batches with optional progress callback (§G.1.4). */
export async function loadPortfolioLogs(
  refs: OutputLogRef[],
  opts: {
    batchSize?: number;
    onProgress?: (loaded: number, total: number) => void;
  } = {}
): Promise<PortfolioRun[]> {
  const { batchSize = 24, onProgress } = opts;
  const loaded: PortfolioRun[] = [];

  for (let i = 0; i < refs.length; i += batchSize) {
    const batch = refs.slice(i, i + batchSize);
    const results = await Promise.all(
      batch.map(async (ref) => {
        const entries = await invoke<DayLogEntry[]>("load_simulation_log", { path: ref.path });
        return { ...ref, entries };
      })
    );
    loaded.push(...results);
    onProgress?.(loaded.length, refs.length);
  }

  return loaded;
}
