import { describe, expect, it } from "vitest";
import type { CloudJobStatus } from "../../src/types/cloud";
import {
  computeJobAnalysis,
  extractPolicyFromCellId,
  extractTaskDuration,
} from "../../src/utils/cloud/analysis";

describe("Cloud Analysis Utility (CR-16)", () => {
  it("extracts policy from various cell_id formats", () => {
    expect(extractPolicyFromCellId("30d/riomaior/100/emp/lm_cls/cf70/bpc")).toBe("bpc");
    expect(extractPolicyFromCellId("tests/vrpp/alns")).toBe("alns");
    expect(extractPolicyFromCellId("simple")).toBe("simple");
    expect(extractPolicyFromCellId("")).toBe("default");
  });

  it("extracts task duration from message when present", () => {
    expect(
      extractTaskDuration({
        task_id: "task-01",
        cell_id: "c1",
        state: "succeeded",
        attempt: 1,
        message: "simulation completed in 42.5s",
      })
    ).toBe(42.5);

    expect(
      extractTaskDuration({
        task_id: "task-02",
        cell_id: "c2",
        state: "succeeded",
        attempt: 1,
        message: "finished in 120 seconds",
      })
    ).toBe(120);
  });

  it("generates deterministic fallback duration based on task state", () => {
    const d1 = extractTaskDuration({
      task_id: "task-01",
      cell_id: "cell-alpha",
      state: "succeeded",
      attempt: 1,
    });
    const d2 = extractTaskDuration({
      task_id: "task-01",
      cell_id: "cell-alpha",
      state: "succeeded",
      attempt: 1,
    });
    expect(d1).toBeGreaterThan(0);
    expect(d1).toBe(d2); // deterministic
  });

  it("computes full job analysis metrics, spot savings and costs", () => {
    const status: CloudJobStatus = {
      job_id: "job-full-test",
      name: "30-day Rio Maior Benchmark",
      state: "succeeded",
      counts: { succeeded: 2 },
      tasks: [
        {
          task_id: "task-001",
          cell_id: "30d/riomaior/100/emp/bpc",
          state: "succeeded",
          attempt: 1,
          message: "completed in 360s",
        },
        {
          task_id: "task-002",
          cell_id: "30d/riomaior/100/emp/alns",
          state: "succeeded",
          attempt: 1,
          message: "completed in 360s",
        },
      ],
    };

    const analysis = computeJobAnalysis(status, {
      provider: "aws",
      compute: "batch",
      spot: true,
      hourlyRateUsd: 0.04,
      onDemandHourlyRateUsd: 0.12,
      estimatedCostUsd: 0.02,
    });

    expect(analysis.job_id).toBe("job-full-test");
    expect(analysis.total_tasks).toBe(2);
    expect(analysis.succeeded_tasks).toBe(2);
    expect(analysis.total_compute_seconds).toBe(720); // 360s + 360s

    // Cost: (720 / 3600) * 0.04 = 0.2 * 0.04 = 0.008
    expect(analysis.accrued_cost_usd).toBeCloseTo(0.008, 4);

    // Baseline: (720 / 3600) * 0.12 = 0.2 * 0.12 = 0.024
    expect(analysis.on_demand_baseline_cost_usd).toBeCloseTo(0.024, 4);

    // Spot savings: 0.024 - 0.008 = 0.016 (66.7%)
    expect(analysis.spot_savings_usd).toBeCloseTo(0.016, 4);
    expect(analysis.spot_savings_pct).toBeCloseTo(66.7, 1);
  });
});
