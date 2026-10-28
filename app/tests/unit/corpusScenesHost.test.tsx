import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { CorpusScenes } from "../../src/components/analysis/summary/corpus/CorpusScenes";
import { useSimulationCorpusStore } from "../../src/store/simulationCorpus";
import type { CorpusScanResult } from "../../src/types/corpus";

const scan: CorpusScanResult = {
  roots: ["/data/scenario"],
  kept: 1,
  skipped: 0,
  failed: [],
  files: [
    {
      path: "/data/scenario/alns/day.json",
      partial: false,
      days: [
        {
          day: 1,
          policy: "alns",
          city: "Porto",
          metrics: { profit: 10 },
          kgLost: 1,
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

describe("CorpusScenes", () => {
  afterEach(() => {
    cleanup();
    useSimulationCorpusStore.setState({ scan: null, selectedRunIds: [], metric: "profit" });
  });

  it("stays empty until a scan exists", () => {
    useSimulationCorpusStore.setState({ scan: null });
    render(<CorpusScenes />);
    expect(screen.queryByTestId("corpus-scenes")).toBeNull();
  });

  it("mounts the summary scenes for a run with no coordinates", () => {
    useSimulationCorpusStore.setState({ scan, selectedRunIds: [], metric: "profit" });
    render(<CorpusScenes />);
    expect(screen.getByText("Policy orbit")).toBeTruthy();
    expect(screen.getByText("Against baseline")).toBeTruthy();
    expect(screen.getByText("Metric flow")).toBeTruthy();
    expect(screen.getByText("Day by bin")).toBeTruthy();
    expect(screen.getByText("No coordinates in this corpus.")).toBeTruthy();
    expect(screen.getByText("Generated layout")).toBeTruthy();
    expect(screen.getByText("Day-level log: one dot per day.")).toBeTruthy();
  });
});
