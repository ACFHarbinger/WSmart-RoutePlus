/**
 * Derive simulation run directories from process stdout (§G.14 / §G.9 / §G.15).
 */
import type { ProcessEntry } from "../../types";
import { extractJsonlPathFromLogLines } from "../benchmark/policyTelemetryTrends";
import { isTrainOrHpoProcess } from "../training/trainingProcess";
import { trainingRunPathFromLogLines } from "../training/trainingRunPath";

export type ProcessLogPathKind = "sim" | "train" | "auto";

// The simulator prints "->" (it printed "→" before 2026-10-24, which the Windows console garbles)
const PRUNED_CONFIG_RE = /Pruned config saved (?:→|->) (.+)/;
const HYDRA_SNAPSHOT_RE = /Hydra config snapshot saved (?:→|->) (.+)/;
const ASSETS_OUTPUT_RE = /([^\s'"]*(?:outputs\/simulations|assets\/output)\/[^\s'"]+)/;

/** Resolve the Hydra run root from a simulation ``.jsonl`` log path. */
export function outputRunPathFromJsonl(jsonlPath: string): string {
  const normalized = jsonlPath.replace(/\\/g, "/");
  const hydraIdx = normalized.lastIndexOf("/hydra/");
  if (hydraIdx !== -1) {
    return normalized.slice(0, hydraIdx);
  }
  const lastSlash = normalized.lastIndexOf("/");
  if (lastSlash === -1) {
    return normalized;
  }
  return normalized.slice(0, lastSlash);
}

/** Derive the run root from a Hydra snapshot or pruned-config artefact path. */
export function outputRunPathFromHydraArtifact(artifactPath: string): string {
  const normalized = artifactPath.replace(/\\/g, "/");
  const hydraIdx = normalized.lastIndexOf("/hydra/");
  if (hydraIdx !== -1) {
    return normalized.slice(0, hydraIdx);
  }
  if (normalized.endsWith("/hydra")) {
    return normalized.slice(0, -"/hydra".length);
  }
  const lastSlash = normalized.lastIndexOf("/");
  if (lastSlash === -1) {
    return normalized;
  }
  return normalized.slice(0, lastSlash);
}

/** Scan stdout for Hydra snapshot / pruned-config / simulation output paths (newest line wins). */
export function extractOutputRunPathFromLogLines(lines: string[]): string | null {
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i]!;
    const pruned = line.match(PRUNED_CONFIG_RE);
    if (pruned?.[1]) {
      return outputRunPathFromHydraArtifact(pruned[1].trim());
    }
    const hydra = line.match(HYDRA_SNAPSHOT_RE);
    if (hydra?.[1]) {
      return outputRunPathFromHydraArtifact(hydra[1].trim());
    }
    const assets = line.match(ASSETS_OUTPUT_RE);
    if (assets?.[1]) {
      const path = assets[1].replace(/\\/g, "/");
      if (path.endsWith(".jsonl")) continue;
      if (path.includes("/hydra")) {
        return outputRunPathFromHydraArtifact(path);
      }
      return path;
    }
  }
  return null;
}

/** Scan process stdout for the newest simulation run directory. */
export function outputRunPathFromLogLines(lines: string[]): string | null {
  const jsonl = extractJsonlPathFromLogLines(lines);
  if (jsonl) return outputRunPathFromJsonl(jsonl);
  return extractOutputRunPathFromLogLines(lines);
}

/** Resolve the best log/run path for ``PathRunLabelChip`` brush from process stdout (§G.9–§G.18 / §D.7). */
export function brushLogPathFromProcessLines(
  lines: string[],
  kind: ProcessLogPathKind = "auto"
): string | null {
  const jsonl = extractJsonlPathFromLogLines(lines);
  if (jsonl) return jsonl;
  if (kind === "train" || kind === "auto") {
    const train = trainingRunPathFromLogLines(lines);
    if (train) return train;
  }
  if (kind === "sim" || kind === "auto") {
    return outputRunPathFromLogLines(lines);
  }
  return null;
}

/** Infer stdout log-path resolution kind from process id + command (§G.15 / §D.7). */
export function processLogPathKind(id: string, command: string): ProcessLogPathKind {
  return isTrainOrHpoProcess(id, command) ? "train" : "sim";
}

/** Convert a ``file://`` URI to a local filesystem path (§G.18 / §D.7). */
export function localPathFromUri(uri: string): string | null {
  if (!uri) return null;
  if (uri.startsWith("file://")) {
    const stripped = uri.slice("file://".length);
    try {
      const decoded = decodeURIComponent(stripped);
      if (/^\/[A-Za-z]:/.test(decoded)) {
        return decoded.slice(1);
      }
      return decoded;
    } catch {
      return stripped;
    }
  }
  if (!uri.startsWith("http://") && !uri.startsWith("https://")) {
    return uri;
  }
  return null;
}

/** MLflow run directory from ``artifact_uri`` (strips trailing ``/artifacts``) (§G.18 / §D.7). */
export function mlflowRunDirFromArtifactUri(artifactUri: string): string | null {
  const local = localPathFromUri(artifactUri);
  if (!local) return null;
  const normalized = local.replace(/\\/g, "/");
  if (normalized.endsWith("/artifacts")) {
    return normalized.slice(0, -"/artifacts".length);
  }
  return local;
}

/** Resolve a local SQLite file path from an Optuna ``sqlite:///`` storage URL (§G.18 / §D.7). */
export function sqlitePathFromStorageUrl(storageUrl: string): string | null {
  if (!storageUrl.startsWith("sqlite:///")) return null;
  const path = storageUrl.slice("sqlite:///".length);
  return path.trim() || null;
}

/** Resolve a tracking URI or relative path against ``projectRoot`` for path-chip brush (§G.18 / §G.19 / §D.7). */
export function resolveLocalProjectPath(
  uriOrPath: string,
  projectRoot: string | null | undefined
): string | null {
  const trimmed = uriOrPath.trim();
  if (!trimmed) return null;
  const fromUri = localPathFromUri(trimmed);
  if (!fromUri) return null;
  if (fromUri.startsWith("/") || /^[A-Za-z]:[\\/]/.test(fromUri)) {
    return fromUri;
  }
  if (!projectRoot) return fromUri;
  const root = projectRoot.replace(/\\/g, "/").replace(/\/$/, "");
  const rel = fromUri.replace(/^\.\//, "");
  return `${root}/${rel}`;
}

/** Resolve Optuna ``sqlite:///`` storage URL to an absolute local path for path-chip brush (§G.18 / §G.19 / §D.7). */
export function sqliteStoragePathFromUrl(
  storageUrl: string,
  projectRoot: string | null | undefined
): string | null {
  const raw = sqlitePathFromStorageUrl(storageUrl);
  if (!raw) return null;
  return resolveLocalProjectPath(raw, projectRoot);
}

/** Derive log/run path per process id for row path-chip brush parity (§G.15 / §D.7). */
export function brushLogPathMapFromProcesses(
  processes: Record<string, Pick<ProcessEntry, "logLines" | "command">>,
  ids: string[]
): Record<string, string | null> {
  const map: Record<string, string | null> = {};
  for (const id of ids) {
    const proc = processes[id];
    if (!proc) continue;
    map[id] = brushLogPathFromProcessLines(
      proc.logLines,
      processLogPathKind(id, proc.command)
    );
  }
  return map;
}
