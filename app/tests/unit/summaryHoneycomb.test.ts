import { describe, expect, it } from "vitest";
import type { DayLogEntry } from "../../src/types";
import {
  binsFromCorpusPoints,
  binsFromLogEntries,
} from "../../src/components/analysis/summary/summaryHoneycomb";

describe("summary honeycomb bins", () => {
  it("matches one coordinated bin to its route coordinates", () => {
    const bins = binsFromCorpusPoints([
      {
        bins: [{ binId: "2", fill: 0.35 }],
        routeBins: [{ id: 2, lat: 41.15, lon: -8.61 }],
      },
    ]);
    expect(bins).toEqual([
      { id: "2", lat: 41.15, lon: -8.61, fill: 0.35, depot: false },
    ]);
  });

  it("keeps a missing fill as a gap and does not invent coordinates", () => {
    const bins = binsFromCorpusPoints([
      {
        bins: [{ binId: "12", fill: null }],
        routeBins: [{ id: 12, lat: null, lon: null }],
      },
    ]);
    expect(bins).toEqual([{ id: "12", lat: null, lon: null, fill: null, depot: false }]);
    expect(bins[0]?.fill).not.toBe(0);
    const kept = binsFromCorpusPoints([
      {
        bins: [{ binId: "12", fill: 0.4 }],
        routeBins: [{ id: 12, lat: 41, lon: -8 }],
      },
      {
        bins: [{ binId: "12", fill: null }],
        routeBins: [{ id: 12, lat: null, lon: null }],
      },
    ]);
    expect(kept).toEqual([{ id: "12", lat: 41, lon: -8, fill: 0.4, depot: false }]);
  });

  it("reads one log entry from all_bin_coords and aligns fill by index", () => {
    const entry: DayLogEntry = {
      policy: "alns",
      sample_id: 0,
      day: 1,
      data: {
        all_bin_coords: [
          { id: -1, lat: 38.72, lng: -9.14 },
          { id: 0, lat: 38.73, lng: -9.13 },
          { id: 6 },
        ],
        bin_state_c: [0, 0.25, 0.8],
      },
    };
    expect(binsFromLogEntries([entry])).toEqual([
      { id: "-1", lat: 38.72, lon: -9.14, fill: 0, depot: true },
      { id: "0", lat: 38.73, lon: -9.13, fill: 0.25, depot: true },
      { id: "6", lat: null, lon: null, fill: 0.8, depot: false },
    ]);

    const mismatched: DayLogEntry = {
      policy: "hgs",
      sample_id: 1,
      day: 2,
      data: {
        all_bin_coords: [{ id: 8, lat: 1, lng: 2 }],
        bin_state_c: [0.4, 0.5],
      },
    };
    expect(binsFromLogEntries([mismatched])).toEqual([
      { id: "8", lat: 1, lon: 2, fill: null, depot: false },
    ]);
  });

  it("returns no bins for an empty input", () => {
    expect(binsFromCorpusPoints([])).toEqual([]);
    expect(binsFromLogEntries([])).toEqual([]);
    const fillsOnly: DayLogEntry = {
      policy: "alns",
      sample_id: 0,
      day: 1,
      data: { bin_state_c: [0.5] },
    };
    expect(binsFromLogEntries([fillsOnly])).toEqual([]);
  });
});
