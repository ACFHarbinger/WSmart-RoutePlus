import { describe, expect, it } from "vitest";
import { computeJobAnalysis, extractTaskDuration } from "../../src/utils/cloud/analysis";
import type { CloudJobStatus } from "../../src/types/cloud";

describe("recorded cloud analysis data", () => {
  const job: CloudJobStatus = {
    job_id: "job-1", name: "Test", state: "succeeded", counts: { succeeded: 1 },
    tasks: [{ task_id: "task-1", cell_id: "opaque-hash", state: "succeeded", attempt: 2,
      message: "retry after 30 seconds; completed" }],
  };
  it("does not infer execution duration from IDs or arbitrary messages", () => {
    expect(extractTaskDuration(job.tasks[0])).toBeNull();
    expect(extractTaskDuration({ ...job.tasks[0], message: undefined })).toBeNull();
  });
  it("keeps unobserved metrics missing instead of inventing successful telemetry", () => {
    const analysis = computeJobAnalysis(job);
    expect(analysis.succeeded_tasks).toBe(1);
    expect(analysis.total_tasks).toBe(1);
    expect(analysis.total_wall_seconds).toBeNull();
    expect(analysis.total_compute_seconds).toBeNull();
    expect(analysis.accrued_cost_usd).toBeNull();
    expect(analysis.spot).toBeNull();
    expect(analysis.tasks[0].expected_wall_seconds).toBeNull();
    expect(analysis.tasks[0].policy).toBeNull();
    expect(analysis.spot_savings_usd).toBeNull();
  });

  it("accurately aggregates real transported telemetry and calculates spot savings", () => {
    const jobWithTelemetry: CloudJobStatus = {
      job_id: "job-aws-1",
      name: "AWS Benchmark",
      state: "succeeded",
      counts: { succeeded: 2 },
      provider: "aws",
      compute: "batch",
      spot: true,
      machine_type: "c6i.large",
      region: "us-east-1",
      price_per_hour: 0.034,
      on_demand_price_per_hour: 0.085,
      tasks: [
        {
          task_id: "task-1",
          cell_id: "c0",
          state: "succeeded",
          attempt: 1,
          started_at: 1000.0,
          finished_at: 1120.0,
          duration: 120.0,
          expected_wall_seconds: 150.0,
          policy: "bpc",
          cost: 0.001133,
        },
        {
          task_id: "task-2",
          cell_id: "c1",
          state: "succeeded",
          attempt: 1,
          started_at: 1010.0,
          finished_at: 1250.0,
          duration: 240.0,
          expected_wall_seconds: 250.0,
          policy: "alns",
          cost: 0.002267,
        },
      ],
    };

    expect(extractTaskDuration(jobWithTelemetry.tasks[0])).toBe(120.0);
    expect(extractTaskDuration(jobWithTelemetry.tasks[1])).toBe(240.0);

    const analysis = computeJobAnalysis(jobWithTelemetry);
    expect(analysis.provider).toBe("aws");
    expect(analysis.compute).toBe("batch");
    expect(analysis.spot).toBe(true);
    expect(analysis.machine_type).toBe("c6i.large");
    expect(analysis.region).toBe("us-east-1");
    expect(analysis.total_tasks).toBe(2);
    expect(analysis.succeeded_tasks).toBe(2);

    // Total compute seconds: 120 + 240 = 360s
    expect(analysis.total_compute_seconds).toBe(360.0);
    // Wall clock: minStart=1000, maxEnd=1250 -> 250s
    expect(analysis.total_wall_seconds).toBe(250.0);
    // Accrued cost: 0.001133 + 0.002267 = 0.0034
    expect(analysis.accrued_cost_usd).toBeCloseTo(0.0034, 4);

    // On-demand baseline: (360 / 3600) * 0.085 = 0.0085
    expect(analysis.on_demand_baseline_cost_usd).toBeCloseTo(0.0085, 4);
    // Spot savings: 0.0085 - 0.0034 = 0.0051 (60% savings)
    expect(analysis.spot_savings_usd).toBeCloseTo(0.0051, 4);
    expect(analysis.spot_savings_pct).toBeCloseTo(60.0, 1);

    // Pre-run prediction from tasks: (150 + 250 = 400s) / 3600 * 0.034 = 0.003778
    expect(analysis.estimated_cost_usd).toBeCloseTo(0.003778, 4);

    // Per-task preservation
    expect(analysis.tasks[0].policy).toBe("bpc");
    expect(analysis.tasks[0].duration_seconds).toBe(120.0);
    expect(analysis.tasks[0].expected_wall_seconds).toBe(150.0);
    expect(analysis.tasks[1].policy).toBe("alns");
    expect(analysis.tasks[1].duration_seconds).toBe(240.0);
    expect(analysis.tasks[1].expected_wall_seconds).toBe(250.0);
  });
});


describe("incomplete and retry telemetry", () => {
  const task = { task_id: "task", cell_id: "cell", state: "succeeded", attempt: 1,
    duration: 10, cost: 1, expected_wall_seconds: 12, started_at: 100, finished_at: 110 };
  const job: CloudJobStatus = { job_id: "job", name: "test", state: "succeeded", counts: {},
    price_per_hour: 1, tasks: [task] };
  it("does not expose partial observations as full job totals", () => {
    const analysis = computeJobAnalysis({ ...job, tasks: [task, {
      task_id: "missing", cell_id: "cell", state: "queued", attempt: 1,
    }] });
    expect(analysis.total_compute_seconds).toBeNull();
    expect(analysis.total_wall_seconds).toBeNull();
    expect(analysis.accrued_cost_usd).toBeNull();
    expect(analysis.estimated_cost_usd).toBeNull();
  });
  it("does not infer retry totals from current-attempt timestamps", () => {
    expect(extractTaskDuration({ ...task, attempt: 2, duration: undefined })).toBeNull();
  });
  it("rejects invalid observations", () => {
    for (const duration of [-1, Infinity, NaN]) {
      const analysis = computeJobAnalysis({ ...job, tasks: [{ ...task, duration }] });
      expect(analysis.total_compute_seconds).toBeNull();
    }
  });
  it("uses observed running interval endpoints without inventing retry end times", () => {
    const analysis = computeJobAnalysis({ ...job, tasks: [{ ...task,
      state: "running", attempt: 2, started_at: 0, finished_at: 200, duration: 100,
    }] });
    expect(analysis.total_compute_seconds).toBe(100);
    expect(analysis.total_wall_seconds).toBe(200);
  });
});
