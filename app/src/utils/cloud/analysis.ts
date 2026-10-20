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

/** Status messages and IDs are not execution telemetry. Returns real duration if present on task. */
export function extractTaskDuration(task: CloudTaskSummary): number | null {
  if (task.duration != null && !isNaN(task.duration)) {
    return task.duration;
  }
  if (task.started_at != null && task.finished_at != null && task.finished_at >= task.started_at) {
    return task.finished_at - task.started_at;
  }
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
  const tasks: CloudTaskAnalysis[] = job.tasks.map((task) => {
    const duration = task.duration ?? (task.started_at != null && task.finished_at != null ? task.finished_at - task.started_at : null);
    const expected = task.expected_wall_seconds ?? null;
    const policy = task.policy ?? null;
    const cost = task.cost ?? null;
    return {
      task_id: task.task_id,
      cell_id: task.cell_id,
      policy,
      state: task.state,
      attempt: task.attempt,
      duration_seconds: duration,
      expected_wall_seconds: expected,
      cost_usd: cost,
      message: task.message ?? null,
    };
  });

  const count = (state: string) => tasks.filter((task) => task.state === state).length;

  const validDurations = tasks
    .map((t) => t.duration_seconds)
    .filter((d): d is number => d != null && !isNaN(d));

  const total_compute_seconds = validDurations.length > 0
    ? validDurations.reduce((acc, d) => acc + d, 0)
    : null;

  const startedTimes = job.tasks
    .map((t) => t.started_at)
    .filter((t): t is number => t != null && !isNaN(t));

  const finishedTimes = job.tasks
    .map((t) => t.finished_at ?? (t.started_at != null && t.duration != null ? t.started_at + t.duration : null))
    .filter((t): t is number => t != null && !isNaN(t));

  let total_wall_seconds: number | null = null;
  if (startedTimes.length > 0 && finishedTimes.length > 0) {
    const minStart = Math.min(...startedTimes);
    const maxEnd = Math.max(...finishedTimes);
    if (maxEnd >= minStart) {
      total_wall_seconds = maxEnd - minStart;
    }
  }

  const validCosts = tasks
    .map((t) => t.cost_usd)
    .filter((c): c is number => c != null && !isNaN(c));

  const accrued_cost_usd = validCosts.length > 0
    ? validCosts.reduce((acc, c) => acc + c, 0)
    : null;

  const spot = job.spot ?? options?.spot ?? null;
  const onDemandRate = job.on_demand_price_per_hour ?? options?.onDemandHourlyRateUsd ?? null;
  const hourlyRate = job.price_per_hour ?? options?.hourlyRateUsd ?? null;

  let on_demand_baseline_cost_usd: number | null = null;
  let spot_savings_usd: number | null = null;
  let spot_savings_pct: number | null = null;

  if (spot && accrued_cost_usd != null && total_compute_seconds != null && onDemandRate != null) {
    on_demand_baseline_cost_usd = (total_compute_seconds / 3600.0) * onDemandRate;
    if (on_demand_baseline_cost_usd > accrued_cost_usd) {
      spot_savings_usd = on_demand_baseline_cost_usd - accrued_cost_usd;
      spot_savings_pct = (spot_savings_usd / on_demand_baseline_cost_usd) * 100;
    } else {
      spot_savings_usd = 0;
      spot_savings_pct = 0;
    }
  }

  let estimated_cost_usd = options?.estimatedCostUsd ?? null;
  if (estimated_cost_usd == null && hourlyRate != null) {
    const validExpected = tasks
      .map((t) => t.expected_wall_seconds)
      .filter((e): e is number => e != null && !isNaN(e));
    if (validExpected.length > 0) {
      const sumExpected = validExpected.reduce((acc, e) => acc + e, 0);
      estimated_cost_usd = (sumExpected / 3600.0) * hourlyRate;
    }
  }

  return {
    job_id: job.job_id,
    name: job.name,
    state: job.state,
    provider: job.provider ?? options?.provider ?? "Unknown",
    compute: job.compute ?? options?.compute ?? "Unknown",
    spot,
    machine_type: job.machine_type ?? options?.machineType ?? null,
    region: job.region ?? options?.region ?? null,
    total_tasks: tasks.length,
    succeeded_tasks: count("succeeded"),
    failed_tasks: count("failed"),
    cancelled_tasks: count("cancelled"),
    preempted_tasks: count("preempted"),
    total_wall_seconds,
    total_compute_seconds,
    accrued_cost_usd,
    estimated_cost_usd,
    on_demand_baseline_cost_usd,
    spot_savings_usd,
    spot_savings_pct,
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
