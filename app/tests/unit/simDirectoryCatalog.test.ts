import { describe, expect, it } from "vitest";
import {
  enrichSimDayData,
  guessGraphPreset,
} from "../../src/utils/graph/graphCoords";
import type { SimDayData } from "../../src/types";

describe("Simulation directory and log adapter utilities", () => {
  it("synthesizes all_bin_coords from coordTable when data has no coords", () => {
    const rawData: SimDayData = {
      bin_state_c: [0.1, 0.4, 0.8],
      tour: [0, 1, 2, 0],
      tour_indices: [0, 1],
      mandatory: [1],
      overflows: 0,
      kg: 150,
    };

    const coordTable = [
      { lat: 39.336, lng: -8.936 }, // Depot index 0
      { lat: 39.340, lng: -8.930 }, // Bin 0 index 1
      { lat: 39.345, lng: -8.925 }, // Bin 1 index 2
      { lat: 39.350, lng: -8.920 }, // Bin 2 index 3
    ];

    const enriched = enrichSimDayData(rawData, coordTable);
    expect(enriched.all_bin_coords).toBeDefined();
    expect(enriched.all_bin_coords).toHaveLength(4);

    // Depot has id -1
    expect(enriched.all_bin_coords![0]).toEqual({
      id: -1,
      lat: 39.336,
      lng: -8.936,
    });

    // Bins have id 0, 1, 2
    expect(enriched.all_bin_coords![1]).toEqual({
      id: 0,
      lat: 39.340,
      lng: -8.930,
    });
    expect(enriched.all_bin_coords![2]).toEqual({
      id: 1,
      lat: 39.345,
      lng: -8.925,
    });
    expect(enriched.all_bin_coords![3]).toEqual({
      id: 2,
      lat: 39.350,
      lng: -8.920,
    });
  });

  it("enriches existing bin coords where lat or lng is missing", () => {
    const rawData: SimDayData = {
      all_bin_coords: [
        { id: -1 },
        { id: 0, lat: 10, lng: 20 },
        { id: 1 },
      ],
    };

    const coordTable = [
      { lat: 39.336, lng: -8.936 }, // Depot nodeIdx 0
      { lat: 39.340, lng: -8.930 }, // Bin 0 nodeIdx 1
      { lat: 39.345, lng: -8.925 }, // Bin 1 nodeIdx 2
    ];

    const enriched = enrichSimDayData(rawData, coordTable);
    expect(enriched.all_bin_coords![0]).toEqual({
      id: -1,
      lat: 39.336,
      lng: -8.936,
    });
    // Preserves existing coords if present
    expect(enriched.all_bin_coords![1]).toEqual({
      id: 0,
      lat: 10,
      lng: 20,
    });
    // Enriches missing coords
    expect(enriched.all_bin_coords![2]).toEqual({
      id: 1,
      lat: 39.345,
      lng: -8.925,
    });
  });

  it("correctly guesses graph presets from standard simulation paths", () => {
    // Rio Maior 100 with waste type
    expect(
      guessGraphPreset("outputs/simulations/90days/riomaior100_Mistura de embalagens/alns/log_alns_10N.json")
    ).toBe("rm-100");

    // Rio Maior 170 with distribution and waste type
    expect(
      guessGraphPreset("outputs/simulations/30days/riomaior170_all/gaussian/run_1/log_gurobi_10N.json")
    ).toBe("rm-170");

    // Figueira da Foz 350
    expect(
      guessGraphPreset("outputs/simulations/90days/figueiradafoz350_all/cmptp/log_regular_10N.json")
    ).toBe("ffz-350");

    // Inferred from bin count fallback
    const entries = [
      {
        policy: "alns",
        sample_id: 0,
        day: 1,
        data: {
          bin_state_c: new Array(100).fill(0),
        },
      },
    ];
    expect(guessGraphPreset(null, entries)).toBe("rm-100");
  });
});
