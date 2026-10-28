import { describe, expect, it } from "vitest";
import { cityHoneycombInputs } from "../../src/utils/benchmark/cityHoneycomb";
import type { CityRunSlice } from "../../src/utils/benchmark/cityComparison";
import { honeycombFromCities, honeycombSubtitle } from "../../src/utils/orion/liveScenes";
import type { DayLogEntry, SimDayData } from "../../src/types";

function entry(data: SimDayData): DayLogEntry {
  return { policy: "alns", sample_id: 0, day: 1, data };
}

function run(path: string, entries: DayLogEntry[]): CityRunSlice {
  return { path, label: path, entries };
}

describe("cityHoneycombInputs", () => {
  it("keeps a coordinated city with overflows and omits the city that has no coordinates", () => {
    const inputs = cityHoneycombInputs([
      [
        "north",
        [
          run("/logs/north-a.jsonl", [
            entry({
              all_bin_coords: [
                { id: 0, lat: 41.0, lng: -8.8 },
                { id: 1, lat: Number.NaN, lng: Number.POSITIVE_INFINITY },
              ],
              overflows: 1,
              profit: 80,
              kg: 12,
            }),
            entry({
              all_bin_coords: [{ id: 2, lat: 41.4, lng: -8.4 }],
              overflows: 3,
            }),
          ]),
        ],
      ],
      [
        "south",
        [
          run("/logs/south.jsonl", [
            entry({
              bin_state_c: [0.2, 0.4],
              profit: 10,
              kg: 4,
            }),
          ]),
        ],
      ],
    ]);

    expect(inputs[0]?.city).toBe("north");
    expect(inputs[0]?.lat).toBeCloseTo(41.2);
    expect(inputs[0]?.lon).toBeCloseTo(-8.6);
    expect(inputs[0]?.fill).toBeNull();
    expect(inputs[0]?.overflows).toBe(2);
    expect(inputs[1]?.city).toBe("south");
    expect(inputs[1]?.lat).toBeNull();
    expect(inputs[1]?.lon).toBeNull();
    expect(inputs[1]?.fill).toBeCloseTo(0.3);
    expect(inputs[1]?.overflows).toBeNull();

    const scene = honeycombFromCities(inputs);
    expect(scene?.generated).toBe(false);
    expect(scene?.omitted).toBe(1);
    expect(scene?.trajectories).toEqual([]);
    expect(scene?.zones.map((zone) => zone.id)).toEqual(["north"]);
    expect(scene?.zones[0]?.metrics?.fill).toBe("gap");
    expect(scene?.zones[0]?.metrics?.overflows).toBe(2);
    expect(scene?.zones[0]?.fillRatio).toBe(0);
    expect(scene?.zones[0]?.status).toBe("crit");
    expect(honeycombSubtitle(scene!)).not.toContain("Generated layout");
  });

  it("leaves overflows null when the key is absent and marks a coordinate-free set as generated", () => {
    const inputs = cityHoneycombInputs([
      ["beta", [run("/logs/beta.jsonl", [entry({ bin_state_c: [1.2] })])]],
      ["alpha", [run("/logs/alpha.jsonl", [entry({})])]],
    ]);

    expect(inputs.find((row) => row.city === "alpha")?.overflows).toBeNull();
    expect(inputs.find((row) => row.city === "alpha")?.fill).toBeNull();
    expect(inputs.every((row) => row.lat === null && row.lon === null)).toBe(true);

    const scene = honeycombFromCities(inputs);
    expect(scene?.generated).toBe(true);
    expect(scene?.omitted).toBe(0);
    expect(scene?.trajectories).toEqual([]);
    expect(scene?.zones.map((zone) => zone.id)).toEqual(["alpha", "beta"]);
    expect(honeycombSubtitle(scene!)).toContain("Generated layout");
  });
});
