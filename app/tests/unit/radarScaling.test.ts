import { describe, it, expect } from "vitest";
import {
  SIM_METRICS,
  normalizeCohortMetric,
} from "../../src/components/ui/telemetry/metrics";

describe("Radar Direction-Aware Cohort Scaling (Owner Ruling #4)", () => {
  it("maps best higher-is-better value (profit, kg, kg/km) to 1.0 and worst to 0.15", () => {
    const profits = [900, 1200, 1500];
    const scaled = normalizeCohortMetric(
      profits,
      SIM_METRICS.profit.higherIsBetter
    );
    expect(scaled[0]).toBeCloseTo(0.15);
    expect(scaled[1]).toBeCloseTo(0.575);
    expect(scaled[2]).toBeCloseTo(1.0);
  });

  it("inverts lower-is-better metrics (km, overflows, cost, kg_lost) so lowest value maps to 1.0", () => {
    const overflows = [0, 5, 10];
    const scaled = normalizeCohortMetric(
      overflows,
      SIM_METRICS.overflows.higherIsBetter
    );
    // 0 overflows is best -> 1.0; 10 overflows is worst -> 0.15
    expect(scaled[0]).toBeCloseTo(1.0);
    expect(scaled[1]).toBeCloseTo(0.575);
    expect(scaled[2]).toBeCloseTo(0.15);

    const kms = [120, 80, 100];
    const kmScaled = normalizeCohortMetric(kms, SIM_METRICS.km.higherIsBetter);
    expect(kmScaled[1]).toBeCloseTo(1.0); // 80 km is best
    expect(kmScaled[0]).toBeCloseTo(0.15); // 120 km is worst
  });

  it("returns 1.0 for all policies when cohort values are identical", () => {
    expect(normalizeCohortMetric([42, 42, 42], true)).toEqual([1.0, 1.0, 1.0]);
    expect(normalizeCohortMetric([0, 0], false)).toEqual([1.0, 1.0]);
  });
});
