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

/** Only finite, nonnegative transported observations are usable. */
function observed(value: number | null | undefined): number | null {
  return value != null && Number.isFinite(value) && value >= 0 ? value : null;
}

export function extractTaskDuration(task: CloudTaskSummary): number | null {
  if (task.duration != null) return observed(task.duration);
  // A current attempt's timestamps cannot recover the duration of prior retries.
  if (task.attempt > 1) return null;
  const start = observed(task.started_at);
  const end = observed(task.finished_at);
  return start != null && end != null && end >= start ? end - start : null;
}

/** Missing observations must not silently become partial job totals. */
function completeSum(values: (number | null | undefined)[]): number | null {
  const valid = values.map(observed);
  return valid.length > 0 && valid.every((value) => value != null)
    ? valid.reduce<number>((sum, value) => sum + value!, 0) : null;
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
    const duration = extractTaskDuration(task);
    const expected = observed(task.expected_wall_seconds);
    const policy = task.policy ?? null;
    const cost = observed(task.cost);
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

  const total_compute_seconds = completeSum(tasks.map((task) => task.duration_seconds));
  // Status timestamps span the first attempt through the last observed end.
  // Never synthesize an end by adding summed retry durations to a latest start.
  const starts = job.tasks.map((task) => observed(task.started_at));
  const ends = job.tasks.map((task) => observed(task.finished_at));
  const completeTiming = starts.length > 0 && starts.every((time) => time != null)
    && ends.every((time, index) => time != null && time >= starts[index]!);
  const total_wall_seconds = completeTiming
    ? Math.max(0, Math.max(...ends as number[]) - Math.min(...starts as number[])) : null;
  const accrued_cost_usd = completeSum(tasks.map((task) => task.cost_usd));

  const spot = job.spot ?? options?.spot ?? null;
  const onDemandRate = observed(job.on_demand_price_per_hour ?? options?.onDemandHourlyRateUsd);
  const hourlyRate = observed(job.price_per_hour ?? options?.hourlyRateUsd);

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

  let estimated_cost_usd = observed(options?.estimatedCostUsd);
  if (estimated_cost_usd == null && hourlyRate != null) {
    const sumExpected = completeSum(tasks.map((task) => task.expected_wall_seconds));
    if (sumExpected != null) estimated_cost_usd = sumExpected / 3600.0 * hourlyRate;
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
  return observed(value) == null ? "Unavailable" : value!.toFixed(digits);
}
