/**
 * Where simulation runs live under the project root.
 *
 * The simulator writes runs to ``sim.output_dir`` relative to the project root, by default
 * ``outputs/simulations/<n>days/<area><n>_<waste>/<distribution>/<run_name>/``. Before 2026-10-13
 * they went to ``assets/output``; that folder is still scanned so older runs stay visible.
 */

export const SIM_OUTPUT_DIR = "outputs/simulations";
export const LEGACY_SIM_OUTPUT_DIR = "assets/output";

function joinRoot(projectRoot: string, rel: string): string {
  return `${projectRoot.replace(/[\\/]+$/, "")}/${rel}`;
}

/** The current simulation output folder under ``projectRoot``. */
export function simOutputRoot(projectRoot: string): string {
  return joinRoot(projectRoot, SIM_OUTPUT_DIR);
}

/** Every folder that may hold simulation runs, the current one first. */
export function simOutputRoots(projectRoot: string): string[] {
  return [SIM_OUTPUT_DIR, LEGACY_SIM_OUTPUT_DIR].map((rel) => joinRoot(projectRoot, rel));
}
