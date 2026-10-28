/**
 * Orion cohort rows for Algorithm Comparison.
 * A metric with no finite values is omitted. It is not stored as zero.
 */
import type { RadarPolicyCohortEntry } from "../../components/ui/orion/OrionRadarChart";
import type { SimKpiKey } from "../../components/ui/telemetry/metrics";
import {
  orbitalBubblesFromPolicies,
  type PolicyMetricRow,
} from "../orion/liveScenes";

export interface OrionLogEntry {
  policy: string;
  data: Partial<Record<SimKpiKey, number | null | undefined>>;
}

const ORION_METRIC_KEYS: readonly SimKpiKey[] = [
  "profit",
  "km",
  "kg",
  "kg/km",
  "overflows",
  "cost",
  "kg_lost",
  "ncol",
];

function finiteMetricMean(
  values: readonly (number | null | undefined)[]
): number | undefined {
  let sum = 0;
  let count = 0;
  for (const value of values) {
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    sum += value;
    count += 1;
  }
  if (count === 0) return undefined;
  return sum / count;
}

function groupedByPolicy(
  entries: readonly OrionLogEntry[]
): Map<string, OrionLogEntry[]> {
  const groups = new Map<string, OrionLogEntry[]>();
  for (const entry of entries) {
    if (entry.policy.trim().length === 0) continue;
    const rows = groups.get(entry.policy);
    if (rows) rows.push(entry);
    else groups.set(entry.policy, [entry]);
  }
  return groups;
}

/** One cohort row per policy that has at least one finite metric mean. */
export function radarCohortFromLog(
  entries: readonly OrionLogEntry[]
): RadarPolicyCohortEntry[] {
  const cohort: RadarPolicyCohortEntry[] = [];
  for (const [policy, rows] of groupedByPolicy(entries)) {
    const means: Partial<Record<SimKpiKey, number>> = {};
    for (const key of ORION_METRIC_KEYS) {
      const mean = finiteMetricMean(rows.map((row) => row.data[key]));
      if (mean === undefined) continue;
      means[key] = mean;
    }
    if (Object.keys(means).length === 0) continue;
    cohort.push({ policy, means });
  }
  return cohort;
}

export function orionPolicySubtitle(policies: readonly string[]): string {
  const names = policies.map((policy) => policy.trim()).filter((policy) => policy.length > 0);
  return names.length > 0 ? names.join(", ") : "Policies in this log";
}

/** One row per policy whose finite profits average to a number. Null profit is left out. */
export function profitBubbleRows(entries: readonly OrionLogEntry[]): PolicyMetricRow[] {
  const rows: PolicyMetricRow[] = [];
  for (const [policy, group] of groupedByPolicy(entries)) {
    const profit = finiteMetricMean(group.map((row) => row.data.profit));
    if (profit === undefined) continue;
    rows.push({ policy, metrics: { profit } });
  }
  return rows;
}

export function policyProfitBubbles(entries: readonly OrionLogEntry[]) {
  return orbitalBubblesFromPolicies(profitBubbleRows(entries), "profit");
}
