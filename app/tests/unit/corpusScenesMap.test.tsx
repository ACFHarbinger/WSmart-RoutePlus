import React from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CorpusSeriesPoint, CorpusView } from "../../src/types/corpus";
import { CorpusGlobeScene } from "../../src/components/analysis/summary/corpus/scenes/CorpusGlobeScene";
import { CorpusHexScene } from "../../src/components/analysis/summary/corpus/scenes/CorpusHexScene";

const deckState = vi.hoisted(() => ({ allow: false }));

vi.mock("@deck.gl/react", () => ({
  default: ({ children }: { children?: React.ReactNode }) => {
    if (!deckState.allow) throw new Error("DeckGL should not mount");
    return React.createElement("div", { "data-testid": "deck-canvas" }, children);
  },
}));

vi.mock("react-map-gl/maplibre", () => ({
  default: () => {
    if (!deckState.allow) throw new Error("MapGL should not mount");
    return React.createElement("div", { "data-testid": "mapgl" });
  },
}));

function point(
  partial: Partial<CorpusSeriesPoint> & Pick<CorpusSeriesPoint, "runId" | "runLabel">
): CorpusSeriesPoint {
  return {
    scenarioId: "scenario",
    scenarioLabel: "Scenario",
    partial: false,
    day: 1,
    policy: "alns",
    city: "Lisbon",
    value: 8,
    metrics: { profit: 8 },
    kgLost: null,
    lat: null,
    lon: null,
    bins: [],
    tour: [],
    tourIndices: [],
    routeBins: [],
    hasCoordinates: false,
    ...partial,
  };
}

function viewOf(points: CorpusSeriesPoint[], extra?: Partial<CorpusView>): CorpusView {
  const runs = new Map<string, { id: string; label: string; hasCoordinates: boolean }>();
  for (const item of points) {
    const existing = runs.get(item.runId);
    if (existing) {
      existing.hasCoordinates = existing.hasCoordinates || item.hasCoordinates;
      continue;
    }
    runs.set(item.runId, {
      id: item.runId,
      label: item.runLabel,
      hasCoordinates: item.hasCoordinates,
    });
  }
  return {
    runs: [...runs.values()].map((run) => ({
      id: run.id,
      label: run.label,
      scenarioId: "scenario",
      scenarioLabel: "Scenario",
      directory: "",
      partial: false,
      files: [],
      hasCoordinates: run.hasCoordinates,
    })),
    points,
    baseline: null,
    baselineLabel: "Pinned run",
    kept: points.length,
    skipped: 0,
    failed: [],
    dayLevelOnly: true,
    ...extra,
  };
}

afterEach(() => {
  deckState.allow = false;
  cleanup();
});

describe("corpus globe and hex scenes", () => {
  it("keeps an uncoordinated corpus off the deck canvas", () => {
    const view = viewOf([
      point({ runId: "a", runLabel: "Alpha", hasCoordinates: false, value: 3 }),
      point({ runId: "b", runLabel: "Beta", hasCoordinates: false, value: 4 }),
    ]);
    const props = { view, metric: "profit" as const, highlight: null, onHighlight: () => undefined };
    render(
      <>
        <CorpusHexScene {...props} />
        <CorpusGlobeScene {...props} />
      </>
    );
    expect(screen.getByText(/Generated layout/)).toBeInTheDocument();
    expect(screen.getByText("No coordinates in this corpus.")).toBeInTheDocument();
    expect(screen.queryByTestId("deck-canvas")).not.toBeInTheDocument();
    expect(document.querySelector("canvas")).toBeNull();
  });

  it("lists a run that has no coordinates under the city globe", () => {
    deckState.allow = true;
    const view = viewOf([
      point({
        runId: "geo",
        runLabel: "PortoRun",
        hasCoordinates: true,
        lat: 41.15,
        lon: -8.61,
        city: "Porto",
        value: 12,
      }),
      point({
        runId: "off",
        runLabel: "InlandRun",
        hasCoordinates: false,
        value: 9,
      }),
    ]);
    render(
      <CorpusGlobeScene view={view} metric="kg" highlight={null} onHighlight={() => undefined} />
    );
    expect(screen.getByText("City globe")).toBeInTheDocument();
    expect(screen.getByText("kg")).toBeInTheDocument();
    expect(screen.getByText("Not on the globe: InlandRun")).toBeInTheDocument();
  });

  it("does not render a null metric as the digit 0 for GapOnlyLisbon", () => {
    const view = viewOf([
      point({
        runId: "gap",
        runLabel: "GapOnlyLisbon",
        hasCoordinates: false,
        value: null,
        metrics: {},
        city: "Lisbon",
      }),
      point({
        runId: "ok",
        runLabel: "ScoredPorto",
        hasCoordinates: false,
        value: 8,
        city: "Porto",
      }),
    ]);
    const onHighlight = vi.fn();
    render(
      <CorpusHexScene view={view} metric="profit" highlight={null} onHighlight={onHighlight} />
    );
    const gap = screen.getByRole("button", { name: /GapOnlyLisbon/ });
    expect(gap).toHaveTextContent("gap");
    expect(gap).toHaveAttribute("data-value", "gap");
    expect(gap.textContent ?? "").not.toMatch(/\b0\b/);
    expect(gap.getAttribute("data-value")).not.toBe("0");
    fireEvent.click(screen.getByRole("button", { name: /ScoredPorto/ }));
    expect(onHighlight).toHaveBeenCalledWith({ runId: "ok" });
  });
});
