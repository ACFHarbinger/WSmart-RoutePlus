import { describe, expect, it } from "vitest";
import {
  honeycombFromBins,
  honeycombFromCities,
  honeycombSubtitle,
  orbitalBubblesFromPolicies,
} from "../../src/utils/orion/liveScenes";

describe("live Orion scenes", () => {
  it("places coordinated bins on a grid and leaves bins without coordinates off that grid", () => {
    const scene = honeycombFromBins([
      { id: "a", lat: 41.2, lon: -8.6, fill: 0.2 },
      { id: "b", lat: null, lon: null, fill: 0.4 },
      { id: "c", lat: 41.0, lon: -8.4, fill: null },
    ]);
    expect(scene).not.toBeNull();
    expect(scene?.generated).toBe(false);
    expect(scene?.omitted).toBe(1);
    expect(scene?.trajectories).toEqual([]);
    expect(scene?.zones.map((zone) => zone.id)).toEqual(["a", "c"]);
    expect(scene?.zones[0]?.row).toBe(0);
    expect(scene?.zones[1]?.col).not.toBe(scene?.zones[0]?.col);
    expect(scene?.zones[1]?.status).toBe("idle");
    expect(scene?.zones[1]?.metrics?.fill).toBe("gap");
    expect(honeycombSubtitle(scene!)).not.toContain("Generated layout");
  });

  it("labels a square grid as a generated layout when no bin has coordinates", () => {
    const scene = honeycombFromBins([
      { id: "a", lat: null, lon: null, fill: null },
      { id: "b", lat: null, lon: null, fill: 1.2 },
    ]);
    expect(scene?.generated).toBe(true);
    expect(scene?.zones).toHaveLength(2);
    expect(scene?.zones[0]?.row).toBe(0);
    expect(scene?.zones[1]?.row).toBe(0);
    expect(scene?.zones[0]?.col).toBe(0);
    expect(scene?.zones[1]?.col).toBe(1);
    expect(scene?.zones[1]?.status).toBe("crit");
    expect(honeycombSubtitle(scene!)).toContain("Generated layout");
  });

  it("does not draw a city overflow count as a fill percentage", () => {
    const scene = honeycombFromCities([
      { city: "north", lat: 41.2, lon: -8.6, fill: null, overflows: 2 },
      { city: "south", lat: null, lon: null, fill: 0.1, overflows: null },
    ]);
    expect(scene?.generated).toBe(false);
    expect(scene?.omitted).toBe(1);
    expect(scene?.zones).toHaveLength(1);
    expect(scene?.zones[0]?.metrics?.fill).toBe("gap");
    expect(scene?.zones[0]?.metrics?.overflows).toBe(2);
    expect(scene?.zones[0]?.fillRatio).toBe(0);
    expect(scene?.zones[0]?.status).toBe("crit");
  });

  it("omits a policy with no finite profit instead of drawing zero", () => {
    const bubbles = orbitalBubblesFromPolicies(
      [
        { policy: "alns", metrics: { profit: 30 } },
        { policy: "hgs", metrics: { profit: null } },
        { policy: "bpc", metrics: { profit: 10 } },
      ],
      "profit"
    );
    expect(bubbles.map((bubble) => bubble.label)).toEqual(["alns", "bpc"]);
    expect(bubbles.find((bubble) => bubble.label === "alns")?.radius).toBeGreaterThan(
      bubbles.find((bubble) => bubble.label === "bpc")?.radius ?? 0
    );
    expect(bubbles.find((bubble) => bubble.label === "alns")?.orbitIdx).toBe(0);
    expect(bubbles.find((bubble) => bubble.label === "bpc")?.orbitIdx).toBe(2);
  });
});
