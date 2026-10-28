import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { DayBinFieldScene } from "../../src/components/analysis/summary/corpus/scenes/DayBinFieldScene";
import { DivergingBarsScene } from "../../src/components/analysis/summary/corpus/scenes/DivergingBarsScene";
import { MetricFlowScene } from "../../src/components/analysis/summary/corpus/scenes/MetricFlowScene";
import { PolicyOrbitScene } from "../../src/components/analysis/summary/corpus/scenes/PolicyOrbitScene";
import type { CorpusRun, CorpusSeriesPoint, CorpusView } from "../../src/types/corpus";
import type { CorpusSceneProps } from "../../src/components/analysis/summary/corpus/sceneProps";

afterEach(() => {
  cleanup();
});

function point(
  overrides: Partial<CorpusSeriesPoint> & Pick<CorpusSeriesPoint, "runId" | "runLabel">
): CorpusSeriesPoint {
  return {
    scenarioId: "scenario",
    scenarioLabel: "Scenario",
    partial: false,
    day: 1,
    policy: "gurobi",
    city: "Porto",
    value: null,
    metrics: {},
    kgLost: null,
    lat: null,
    lon: null,
    bins: [],
    hasCoordinates: false,
    ...overrides,
  };
}

function run(id: string, label: string): CorpusRun {
  return {
    id,
    label,
    scenarioId: "scenario",
    scenarioLabel: "Scenario",
    directory: "",
    partial: false,
    files: [],
    hasCoordinates: false,
  };
}

function view(points: CorpusSeriesPoint[], extra: Partial<CorpusView> = {}): CorpusView {
  const runs = new Map<string, CorpusRun>();
  for (const item of points) {
    if (!runs.has(item.runId)) runs.set(item.runId, run(item.runId, item.runLabel));
  }
  return {
    runs: [...runs.values()],
    points,
    baseline: null,
    baselineLabel: "",
    kept: points.length,
    skipped: 0,
    failed: [],
    dayLevelOnly: points.every((item) => item.bins.length === 0),
    ...extra,
  };
}

const idle: Pick<CorpusSceneProps, "highlight" | "onHighlight"> = {
  highlight: null,
  onHighlight: () => {},
};

describe("corpus 2D scenes", () => {
  it("does not draw a null profit as 0", () => {
    const data = view([
      point({
        runId: "a",
        runLabel: "Alpha",
        value: null,
        day: null,
        policy: null,
        city: null,
      }),
    ]);
    const { container } = render(
      <>
        <PolicyOrbitScene view={data} metric="profit" {...idle} />
        <DivergingBarsScene view={data} metric="profit" {...idle} />
        <MetricFlowScene view={data} metric="profit" {...idle} />
        <DayBinFieldScene view={data} metric="profit" {...idle} />
      </>
    );
    expect(container.textContent ?? "").not.toMatch(/(?:^|\D)0(?:\D|$)/);
    expect(screen.getAllByText(/gap/i).length).toBeGreaterThan(0);
  });

  it("shows a diverging delta of 5 against the baseline", () => {
    const data = view([point({ runId: "a", runLabel: "Alpha", value: 15 })], {
      baseline: 10,
      baselineLabel: "Pinned Alpha",
    });
    render(<DivergingBarsScene view={data} metric="profit" {...idle} />);
    expect(screen.getByRole("button", { name: /delta 5/i })).toBeInTheDocument();
    expect(screen.getByText(/Pinned Alpha/)).toBeInTheDocument();
  });

  it("says there is no baseline instead of subtracting zero", () => {
    const data = view([point({ runId: "a", runLabel: "Alpha", value: 15 })]);
    render(<DivergingBarsScene view={data} metric="profit" {...idle} />);
    expect(screen.getByText("No baseline")).toBeInTheDocument();
  });

  it("says when the field is one dot per day", () => {
    const data = view([point({ runId: "a", runLabel: "Alpha", value: 3, bins: [] })], {
      dayLevelOnly: true,
    });
    render(<DayBinFieldScene view={data} metric="profit" {...idle} />);
    expect(screen.getByText("Day-level log: one dot per day.")).toBeInTheDocument();
  });

  it("splits served and lost kilograms", () => {
    const data = view([
      point({
        runId: "a",
        runLabel: "Alpha",
        value: 4,
        metrics: { kg: 4 },
        kgLost: 1,
      }),
    ]);
    render(<MetricFlowScene view={data} metric="kg" {...idle} />);
    expect(screen.getByText(/served/i)).toBeInTheDocument();
    expect(screen.getByText(/lost/i)).toBeInTheDocument();
  });

  it("titles the orbit scene", () => {
    const data = view([point({ runId: "a", runLabel: "Alpha", value: 12 })]);
    render(<PolicyOrbitScene view={data} metric="profit" {...idle} />);
    expect(screen.getByText("Policy orbit")).toBeInTheDocument();
    expect(screen.queryByText(/Trial #/)).not.toBeInTheDocument();
  });
});
