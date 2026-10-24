import type { HoneycombCellData } from "../../ui/orion/OrionHoneycombMatrix";
import type { OptunaTrial } from "../../../types";

export interface HpoHoneycombConfig {
  xKey: string;
  yKey: string;
  cells: HoneycombCellData[];
}

function rankedAxisIndex(labels: string[]): Map<string, number> {
  const unique = [...new Set(labels)];
  const numeric = unique.every((label) => Number.isFinite(Number(label)));
  unique.sort((a, b) => (numeric ? Number(a) - Number(b) : a.localeCompare(b)));
  return new Map(unique.map((label, index) => [label, index]));
}

/**
 * Place completed trials on the two most important hyperparameters.
 * Row and column are the ranked parameter values. A repeated pair keeps
 * the trial with the lower objective.
 */
export function buildHpoHoneycomb(
  trials: readonly OptunaTrial[],
  importances: Record<string, number>
): HpoHoneycombConfig | null {
  const completed = trials.filter((t) => t.state === "COMPLETE" && t.value != null);
  if (completed.length < 2) return null;
  const importanceSorted = Object.entries(importances)
    .sort((a, b) => b[1] - a[1])
    .map(([key]) => key);
  const allParamKeys = [
    ...new Set(completed.flatMap((trial) => Object.keys(trial.params))),
  ];
  const top2 =
    importanceSorted.length >= 2 ? importanceSorted.slice(0, 2) : allParamKeys.slice(0, 2);
  if (top2.length < 2) return null;
  const xKey = top2[0]!;
  const yKey = top2[1]!;
  const xIndex = rankedAxisIndex(completed.map((t) => String(t.params[xKey] ?? "—")));
  const yIndex = rankedAxisIndex(completed.map((t) => String(t.params[yKey] ?? "—")));
  const bestByCell = new Map<string, HoneycombCellData>();
  for (const trial of completed) {
    const xLabel = String(trial.params[xKey] ?? "—");
    const yLabel = String(trial.params[yKey] ?? "—");
    const key = `${yIndex.get(yLabel) ?? 0}:${xIndex.get(xLabel) ?? 0}`;
    const value = Number(trial.value ?? 0);
    const cell: HoneycombCellData = {
      id: `trial-${trial.number}`,
      row: yIndex.get(yLabel) ?? 0,
      col: xIndex.get(xLabel) ?? 0,
      label: `#${trial.number}`,
      xParam: `${xKey}=${xLabel}`,
      yParam: `${yKey}=${yLabel}`,
      value,
      secondaryLabel: trial.state,
    };
    const previous = bestByCell.get(key);
    if (!previous || value < previous.value) bestByCell.set(key, cell);
  }
  return { xKey, yKey, cells: [...bestByCell.values()] };
}
