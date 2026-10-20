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
});
