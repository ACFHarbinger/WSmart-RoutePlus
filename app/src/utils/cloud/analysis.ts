/**
 * Cloud Job Analysis utility — extracts execution metrics, cost accruals (CR-10),
 * spot savings, runtime predictions (CR-11), and CSV export for CR-16 (§J.1).
 */

import type { CloudJobAnalysis, CloudJobStatus, CloudTaskAnalysis, CloudTaskSummary } from "../../types/cloud";

export interface AnalysisComputeOptions {
  provider?: string;
  compute?: string;
  spot?: boolean;
  machineType?: string;
  region?: string;
  hourlyRateUsd?: number;
  onDemandHourlyRateUsd?: number;
  estimatedCostUsd?: number | null;
}

/** Status messages and IDs are not execution telemetry. */
export function extractTaskDuration(_task: CloudTaskSummary): null {
  return null;
}

/**
 * Extract policy name from cell_id (e.g. "30d/riomaior/100/emp/lm_cls/cf70/bpc" -> "bpc").
 */
export function extractPolicyFromCellId(cellId: string): string {
  const parts = cellId.split("/").map((p) => p.trim()).filter(Boolean);
  if (parts.length > 0) {
    return parts[parts.length - 1];
  }
  return "default";
}

/**
 * Computes full analysis metrics for a given CloudJobStatus.
 */
export function computeJobAnalysis(
  job: CloudJobStatus,
  options?: AnalysisComputeOptions
): CloudJobAnalysis {
  // The status transport currently carries no timestamps, prices or predictions.
  // Preserve missing data until the Python -> Rust -> TypeScript contract exposes it.
  const tasks: CloudTaskAnalysis[] = job.tasks.map((task) => ({
    ...task,
    policy: null, // cell_id is opaque (normally a hash), not a policy path.
    duration_seconds: null,
    expected_wall_seconds: null,
    cost_usd: null,
  }));
  const count = (state: string) => tasks.filter((task) => task.state === state).length;
  return {
    job_id: job.job_id,
    name: job.name,
    state: job.state,
    provider: options?.provider ?? "Unknown",
    compute: options?.compute ?? "Unknown",
    spot: options?.spot ?? null,
    machine_type: options?.machineType ?? null,
    region: options?.region ?? null,
    total_tasks: tasks.length,
    succeeded_tasks: count("succeeded"),
    failed_tasks: count("failed"),
    cancelled_tasks: count("cancelled"),
    preempted_tasks: count("preempted"),
    total_wall_seconds: null,
    total_compute_seconds: null,
    accrued_cost_usd: null,
    estimated_cost_usd: options?.estimatedCostUsd ?? null,
    on_demand_baseline_cost_usd: null,
    spot_savings_usd: null,
    spot_savings_pct: null,
    tasks,
  };
}

/**
 * Triggers a CSV file download from task analysis records in the browser / WebView.
 */
export function exportTasksToCsv(tasks: CloudTaskAnalysis[], filename = "cloud_tasks_analysis.csv"): void {
  const headers = [
    "Task ID",
    "Cell ID",
    "Policy",
    "State",
    "Attempt",
    "Duration (s)",
    "Expected (s)",
    "Cost (USD)",
    "Message",
  ];

  const escapeCsv = (val: string | number | null | undefined): string => {
    if (val == null) return "";
    const str = String(val);
    if (str.includes(",") || str.includes('"') || str.includes("\n")) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };

  const rows = tasks.map((t) => [
    escapeCsv(t.task_id),
    escapeCsv(t.cell_id),
    escapeCsv(t.policy || ""),
    escapeCsv(t.state),
    escapeCsv(t.attempt),
    escapeCsv(t.duration_seconds?.toFixed(2)),
    escapeCsv(t.expected_wall_seconds != null ? t.expected_wall_seconds.toFixed(2) : ""),
    escapeCsv(t.cost_usd?.toFixed(5)),
    escapeCsv(t.message || ""),
  ]);

  const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function formatObserved(value: number | null | undefined, digits = 1): string {
  return value == null ? "Unavailable" : value.toFixed(digits);
}
