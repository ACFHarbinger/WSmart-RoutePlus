import { describe, expect, it } from "vitest";

import type { CorpusSeriesPoint, CorpusView } from "../../src/types/corpus";
import { corpusMapFrame } from "../../src/utils/corpus/routeFrame";

function point(patch: Partial<CorpusSeriesPoint>): CorpusSeriesPoint {
  return {
    runId: "dir:/data/alns",
    runLabel: "alns",
    scenarioId: "/data",
    scenarioLabel: "data",
    partial: false,
    day: 1,
    policy: "alns",
    city: "Porto",
    value: 10,
    metrics: { profit: 10 },
    kgLost: null,
    lat: 41.1,
    lon: -8.6,
    bins: [],
    tour: [0, 1, 0],
    tourIndices: [0],
    routeBins: [{ id: 0, lat: 41.1, lon: -8.6 }],
    hasCoordinates: true,
    ...patch,
  };
}

function view(points: CorpusSeriesPoint[]): CorpusView {
  return {
    runs: [],
    points,
    baseline: null,
    baselineLabel: "Typed baseline",
    kept: points.length,
    skipped: 0,
    failed: [],
    dayLevelOnly: true,
  };
}

describe("corpusMapFrame", () => {
  it("builds a route when the day has a tour and coordinates", () => {
    const frame = corpusMapFrame(view([point({})]), 1);
    expect(frame.reason).toBeNull();
    expect(frame.routes).toHaveLength(1);
    expect(frame.routes[0]?.data.tour).toEqual([0, 1, 0]);
    expect(frame.routes[0]?.data.all_bin_coords?.[0]).toMatchObject({ id: 0, lat: 41.1, lng: -8.6 });
  });

  it("does not invent a route for a partial file without geometry", () => {
    const frame = corpusMapFrame(
      view([point({ partial: true, tour: [], tourIndices: [], routeBins: [] })]),
      1
    );
    expect(frame.routes).toEqual([]);
    expect(frame.reason).toBe("This partial run has no route geometry.");
  });
});
