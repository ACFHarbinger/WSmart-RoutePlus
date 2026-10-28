import { describe, expect, it } from "vitest";

import type { CorpusFilters, CorpusScanResult } from "../../src/types/corpus";
import { buildCorpusView, corpusTableTsv, generatedLayout, groupCorpusRuns } from "../../src/utils/corpus/select";

function filters(patch: Partial<CorpusFilters> = {}): CorpusFilters {
  return {
    selectedRunIds: [],
    hiddenRunIds: [],
    metric: "profit",
    dayMin: null,
    dayMax: null,
    policies: [],
    cities: [],
    playbackDay: null,
    baselineMode: "value",
    pinnedRunId: null,
    typedBaseline: null,
    ...patch,
  };
}

const scan: CorpusScanResult = {
  roots: ["/data/scenario"],
  kept: 2,
  skipped: 1,
  failed: [{ path: "/data/scenario/bad.json", reason: "invalid json" }],
  files: [
    {
      path: "/data/scenario/alns/day.json",
      partial: false,
      days: [
        {
          day: 1,
          policy: "alns",
          city: "Porto",
          metrics: { profit: 10, kg: 4 },
          kgLost: 1,
          lat: 41.15,
          lon: -8.61,
          bins: [{ binId: "a", fill: 0.4 }],
          tour: [],
          tourIndices: [],
          routeBins: [],
        },
        {
          day: 2,
          policy: "alns",
          city: "Porto",
          metrics: { kg: 6 },
          kgLost: null,
          lat: null,
          lon: null,
          bins: [],
          tour: [],
          tourIndices: [],
          routeBins: [],
        },
      ],
    },
    {
      path: "/data/scenario/alns/extra.json",
      partial: false,
      days: [
        {
          day: 1,
          policy: "hgs",
          city: "Lisbon",
          metrics: { profit: 30 },
          kgLost: null,
          lat: null,
          lon: null,
          bins: [],
          tour: [],
          tourIndices: [],
          routeBins: [],
        },
      ],
    },
    {
      path: "/data/other/one.json",
      partial: true,
      days: [
        {
          day: 3,
          policy: "bpc",
          city: null,
          metrics: { profit: 5 },
          kgLost: null,
          lat: null,
          lon: null,
          bins: [],
          tour: [],
          tourIndices: [],
          routeBins: [],
        },
      ],
    },
  ],
};

describe("groupCorpusRuns", () => {
  it("merges JSON files under one run directory and keeps a single file partial", () => {
    const runs = groupCorpusRuns(scan.files);
    const full = runs.find((run) => run.label === "alns");
    const partial = runs.find((run) => run.partial);
    expect(full?.files).toHaveLength(2);
    expect(full?.scenarioLabel).toBe("scenario");
    expect(full?.hasCoordinates).toBe(true);
    expect(partial?.files).toEqual(["/data/other/one.json"]);
    expect(partial?.hasCoordinates).toBe(false);
  });
});

describe("buildCorpusView", () => {
  it("leaves a missing metric as a gap and applies the typed baseline", () => {
    const view = buildCorpusView(scan, filters({ typedBaseline: 12, dayMax: 1 }));
    const gap = view.points.find((point) => point.policy === "hgs");
    const alns = view.points.find((point) => point.policy === "alns" && point.day === 1);
    expect(gap?.value).toBe(30);
    expect(alns?.value).toBe(10);
    expect(view.points.some((point) => point.day === 2)).toBe(false);
    expect(view.baseline).toBe(12);
    expect(view.baselineLabel).toBe("Typed baseline");
    expect(view.dayLevelOnly).toBe(false);
  });

  it("uses the mean of the pinned run and skips hidden runs", () => {
    const runs = groupCorpusRuns(scan.files);
    const pinned = runs.find((run) => run.label === "alns");
    const hidden = runs.find((run) => run.partial);
    const view = buildCorpusView(
      scan,
      filters({
        baselineMode: "run",
        pinnedRunId: pinned?.id ?? null,
        hiddenRunIds: hidden ? [hidden.id] : [],
        metric: "profit",
      })
    );
    expect(view.runs.some((run) => run.partial)).toBe(false);
    expect(view.baseline).toBe(20);
    expect(view.points.find((point) => point.day === 2)?.value).toBeNull();
  });
});

describe("generatedLayout", () => {
  it("places nodes on a plane and the table keeps blank gaps", () => {
    const nodes = generatedLayout([
      { id: "a", label: "A" },
      { id: "b", label: "B" },
    ]);
    expect(nodes[0]).toMatchObject({ x: -0.5, y: 0 });
    expect(nodes[1]).toMatchObject({ x: 0.5, y: 0 });
    const view = buildCorpusView(scan, filters({ metric: "km" }));
    const table = corpusTableTsv(view, "km");
    expect(table.startsWith("run\tscenario\tday\tpolicy\tcity\tkm\tbaseline")).toBe(true);
    expect(table).toContain("\t\t");
  });
});
