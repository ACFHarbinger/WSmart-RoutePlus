import React from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CorpusPlayback } from "../../src/components/analysis/summary/corpus/CorpusPlayback";
import { CorpusRouteMap } from "../../src/components/analysis/summary/corpus/CorpusRouteMap";
import { useSimulationCorpusStore } from "../../src/store/simulationCorpus";
import type { CorpusDay, CorpusScanResult } from "../../src/types/corpus";

const mapProps = vi.hoisted(() => ({
  animate: [] as Array<boolean | undefined>,
}));

vi.mock("../../src/components/maps/DeckRouteMap", () => ({
  default: (props: { animate?: boolean }) => {
    mapProps.animate.push(props.animate);
    return React.createElement("div", {
      "data-testid": "route-map",
      "data-animate": String(props.animate),
    });
  },
}));

function day(patch: Partial<CorpusDay>): CorpusDay {
  return {
    day: 1,
    policy: "alns",
    city: "Porto",
    metrics: { profit: 10 },
    kgLost: null,
    lat: 41.1,
    lon: -8.6,
    bins: [],
    tour: [],
    tourIndices: [],
    routeBins: [],
    ...patch,
  };
}

function scanOf(partial: boolean, row: CorpusDay): CorpusScanResult {
  return {
    roots: ["/data"],
    kept: 1,
    skipped: 0,
    failed: [],
    files: [{ path: "/data/run.json", partial, days: [row] }],
  };
}

function resetStore(): void {
  useSimulationCorpusStore.setState({
    roots: [],
    scan: null,
    loading: false,
    error: null,
    selectedRunIds: [],
    hiddenRunIds: [],
    metric: "profit",
    dayMin: null,
    dayMax: null,
    policies: [],
    cities: [],
    playbackDay: null,
    playbackPlaying: false,
    baselineMode: "run",
    pinnedRunId: null,
    typedBaseline: null,
    highlight: null,
  });
}

function renderPlayback(): void {
  render(
    <>
      <CorpusPlayback />
      <CorpusRouteMap />
    </>
  );
}

afterEach(() => {
  cleanup();
  mapProps.animate.length = 0;
  resetStore();
});

describe("corpus playback", () => {
  it("says why a partial run with no route geometry does not draw the map", () => {
    useSimulationCorpusStore.setState({
      scan: scanOf(
        true,
        day({
          tour: [],
          tourIndices: [],
          routeBins: [],
        })
      ),
      selectedRunIds: [],
    });
    renderPlayback();
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    expect(screen.getByText("This partial run has no route geometry.")).toBeInTheDocument();
    expect(screen.queryByTestId("route-map")).not.toBeInTheDocument();
  });

  it("draws the route map after the playback day is set", async () => {
    useSimulationCorpusStore.setState({
      scan: scanOf(
        false,
        day({
          tour: [0, 1, 0],
          tourIndices: [0],
          routeBins: [{ id: 0, lat: 41.1, lon: -8.6 }],
        })
      ),
      selectedRunIds: [],
    });
    renderPlayback();
    expect(screen.queryByTestId("route-map")).not.toBeInTheDocument();
    useSimulationCorpusStore.getState().setPlaybackDay(1);
    expect(await screen.findByTestId("route-map")).toBeInTheDocument();
  });

  it("keeps route animation off when reduced motion is requested", async () => {
    const previous = window.matchMedia;
    window.matchMedia = vi.fn((query: string) => ({
      matches: query === "(prefers-reduced-motion: reduce)",
      media: query,
      onchange: null,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
    })) as unknown as typeof window.matchMedia;

    try {
      useSimulationCorpusStore.setState({
        scan: scanOf(
          false,
          day({
            tour: [0, 1, 0],
            tourIndices: [0],
            routeBins: [{ id: 0, lat: 41.1, lon: -8.6 }],
          })
        ),
        selectedRunIds: [],
        playbackDay: 1,
        playbackPlaying: true,
      });
      renderPlayback();
      const map = await screen.findByTestId("route-map");
      expect(map).toHaveAttribute("data-animate", "false");
      expect(mapProps.animate.at(-1)).toBe(false);
    } finally {
      window.matchMedia = previous;
    }
  });
});
