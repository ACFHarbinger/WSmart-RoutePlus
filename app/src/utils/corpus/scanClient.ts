import { invoke } from "@tauri-apps/api/core";

import type { CorpusScanResult } from "../../types/corpus";

/** Walk one or more parent directories off the WebView (#176). */
export function scanSimulationCorpus(roots: string[]): Promise<CorpusScanResult> {
  return invoke<CorpusScanResult>("scan_simulation_corpus", { roots });
}
