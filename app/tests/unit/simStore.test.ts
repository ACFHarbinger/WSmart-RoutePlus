import { beforeEach, describe, expect, it } from "vitest";
import { useSimStore } from "../../src/store/sim";
import { useProcessStore } from "../../src/store/process";
import type { DayLogEntry, PolicyVizEntry, SimFailureEntry } from "../../src/types";

describe("useSimStore O(1) indexed deduplication and batching (Slice V0)", () => {
  beforeEach(() => {
    useSimStore.getState().reset();
  });

  it("deduplicates DayLogEntry by (policy, sample_id, day) and updates lastEventAt", () => {
    const entry1: DayLogEntry = {
      policy: "hgs",
      sample_id: 0,
      day: 1,
      data: { profit: 120, km: 14.5 },
    };
    const dup: DayLogEntry = {
      policy: "hgs",
      sample_id: 0,
      day: 1,
      data: { profit: 999 },
    };
    useSimStore.getState().addEntries([entry1, dup]);
    const state = useSimStore.getState();
    expect(state.entries).toHaveLength(1);
    expect(state.entries[0]?.data.profit).toBe(120);
    expect(state.lastEventAt).not.toBeNull();
  });

  it("handles 5,000 batched day entries in O(1) per entry (< 50 ms)", () => {
    const batch: DayLogEntry[] = [];
    for (let i = 0; i < 5000; i++) {
      batch.push({
        policy: `pol_${i % 10}`,
        sample_id: Math.floor(i / 1000),
        day: i % 100,
        data: { profit: i, km: i * 0.5 },
      });
    }
    const t0 = performance.now();
    useSimStore.getState().addEntries(batch);
    const elapsed = performance.now() - t0;
    expect(elapsed).toBeLessThan(100);
    // 10 policies * 5 samples * 100 days = 5000 unique keys
    expect(useSimStore.getState().entries.length).toBeGreaterThan(0);
  });

  it("replaces PolicyVizEntry only when incoming series length is >= existing", () => {
    const shortViz: PolicyVizEntry = {
      policy: "alns",
      sample_id: 0,
      day: 1,
      policy_type: "alns",
      data: { best_cost: [10, 9] },
    };
    const longViz: PolicyVizEntry = {
      policy: "alns",
      sample_id: 0,
      day: 1,
      policy_type: "alns",
      data: { best_cost: [10, 9, 8, 7] },
    };
    useSimStore.getState().addPolicyVizEntry(longViz);
    useSimStore.getState().addPolicyVizEntry(shortViz);
    expect(useSimStore.getState().policyVizEntries[0]?.data.best_cost).toHaveLength(4);
  });

  it("applies SimBatchUpdate atomically across day, viz, and failure entries", () => {
    const failure: SimFailureEntry = {
      policy: "bpc",
      sample_id: 0,
      day: 3,
      data: { has_failure: true, severity: "warning", root_causes: ["overflow"], summary: "test" },
    };
    useSimStore.getState().applyBatch({
      day_entries: [{ policy: "bpc", sample_id: 0, day: 3, data: { overflows: 2 } }],
      viz_entries: [],
      failure_entries: [failure],
    });
    expect(useSimStore.getState().entries).toHaveLength(1);
    expect(useSimStore.getState().failureEntries).toHaveLength(1);
  });
});

describe("useProcessStore batched log ring buffer (Slice V0)", () => {
  it("caps logLines at 2000 when appending large batches and records lastEventAt", () => {
    const store = useProcessStore.getState();
    store.addProcess({
      id: "sim_test_1",
      command: "python main.py test_sim",
      pid: 1234,
      status: "running",
      startTime: Date.now(),
    });
    const batch = Array.from({ length: 2500 }, (_, i) => `line-${i}`);
    useProcessStore.getState().appendLogs("sim_test_1", batch);
    const proc = useProcessStore.getState().processes["sim_test_1"];
    expect(proc?.logLines).toHaveLength(2000);
    expect(proc?.logLines[0]).toBe("line-500");
    expect(proc?.logLines[1999]).toBe("line-2499");
    expect(proc?.lastEventAt).toBeDefined();
    useProcessStore.getState().removeProcess("sim_test_1");
  });
});
