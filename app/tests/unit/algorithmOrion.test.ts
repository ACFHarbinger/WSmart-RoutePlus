import { describe, expect, it } from "vitest";
import {
  orionPolicySubtitle,
  profitBubbleRows,
  radarCohortFromLog,
} from "../../src/utils/analysis/algorithmOrion";
import { orbitalBubblesFromPolicies } from "../../src/utils/orion/liveScenes";

describe("algorithm Orion log rows", () => {
  it("leaves a null profit out of the bubble input and keeps finite profits", () => {
    const rows = profitBubbleRows([
      { policy: "hgs", data: { profit: null } },
      { policy: "alns", data: { profit: 30 } },
      { policy: "bpc", data: { profit: 10 } },
    ]);
    expect(rows.map((row) => row.policy)).toEqual(["alns", "bpc"]);
    expect(rows.map((row) => row.metrics.profit)).toEqual([30, 10]);
    const bubbles = orbitalBubblesFromPolicies(rows, "profit");
    expect(bubbles.map((bubble) => bubble.label)).toEqual(["alns", "bpc"]);
    expect(bubbles.some((bubble) => bubble.value === 0)).toBe(false);
  });

  it("averages a metric from finite values and does not turn a null into zero", () => {
    const cohort = radarCohortFromLog([
      { policy: "alns", data: { profit: null, km: 4 } },
      { policy: "alns", data: { profit: 10, km: null } },
      { policy: "hgs", data: { profit: null, km: null } },
    ]);
    expect(cohort).toEqual([{ policy: "alns", means: { profit: 10, km: 4 } }]);
    expect(cohort[0]?.means.profit).toBe(10);
    expect(cohort[0]?.means.km).not.toBe(0);
    expect(orionPolicySubtitle(cohort.map((entry) => entry.policy))).toBe("alns");
    expect(orionPolicySubtitle([])).toBe("Policies in this log");
  });
});
