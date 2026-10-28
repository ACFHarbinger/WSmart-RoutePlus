import { cleanup, fireEvent, render, renderHook, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CorpusStrip } from "../../src/components/analysis/summary/corpus/CorpusStrip";
import {
  useCorpusSceneState,
  useSimulationCorpusStore,
} from "../../src/store/simulationCorpus";
import type { CorpusScanResult } from "../../src/types/corpus";

const { scanSimulationCorpus } = vi.hoisted(() => ({
  scanSimulationCorpus: vi.fn(),
}));

vi.mock("../../src/utils/corpus/scanClient", () => ({
  scanSimulationCorpus,
}));

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
        },
      ],
    },
  ],
};

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
    baselineMode: "run",
    pinnedRunId: null,
    typedBaseline: null,
    highlight: null,
  });
}

afterEach(() => {
  cleanup();
});

beforeEach(() => {
  scanSimulationCorpus.mockReset();
  resetStore();
});

describe("CorpusStrip", () => {
  it("renders before a simulation log is open", () => {
    render(<CorpusStrip />);
    expect(screen.getByRole("button", { name: "Add result folders" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add JSON file" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reload" })).toBeInTheDocument();
    expect(screen.getByTestId("corpus-baseline")).toHaveTextContent("No baseline");
    const scene = renderHook(() => useCorpusSceneState());
    expect(scene.result.current.view).toBeNull();
  });

  it("lists scanned runs and does not show a missing profit as zero", async () => {
    scanSimulationCorpus.mockResolvedValue(scan);
    await useSimulationCorpusStore.getState().addRoots(["/data/scenario"]);
    render(<CorpusStrip />);

    expect(scanSimulationCorpus).toHaveBeenCalledWith(["/data/scenario"]);
    expect(screen.getByRole("checkbox", { name: "alns" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "one" })).toBeChecked();
    expect(screen.getByText("partial")).toBeInTheDocument();
    expect(screen.getByTestId("corpus-counts")).toHaveTextContent("Kept 2, skipped 1, failed 1.");
    expect(screen.getByText("/data/scenario/bad.json: invalid json")).toBeInTheDocument();

    const baseline = screen.getByTestId("corpus-baseline");
    expect(baseline).toHaveTextContent("Baseline: No baseline");
    expect(baseline).not.toHaveTextContent(/\b0\b/);

    const scene = renderHook(() => useCorpusSceneState());
    const gap = scene.result.current.view?.points.find((point) => point.day === 2);
    expect(gap?.value).toBeNull();
    expect([...screen.getByLabelText("Cities").querySelectorAll("option")].map((option) => option.value)).toEqual([
      "Lisbon",
      "Porto",
    ]);
    expect(screen.getByRole("option", { name: "kg/km" })).toHaveValue("kg_per_km");

    fireEvent.click(screen.getByRole("button", { name: "alns" }));
    expect(useSimulationCorpusStore.getState().highlight).toEqual({ runId: "dir:/data/scenario/alns" });
    fireEvent.click(screen.getByRole("button", { name: "alns" }));
    expect(useSimulationCorpusStore.getState().highlight).toBeNull();
  });

  it("pins a run label and then uses a typed baseline", async () => {
    scanSimulationCorpus.mockResolvedValue(scan);
    await useSimulationCorpusStore.getState().addRoots(["/data/scenario"]);
    render(<CorpusStrip />);

    const pinned = screen.getByLabelText("Pinned run");
    const alns = [...pinned.querySelectorAll("option")].find((option) => option.textContent === "alns");
    expect(alns).toBeTruthy();
    fireEvent.change(pinned, { target: { value: alns?.value } });
    expect(screen.getByTestId("corpus-baseline")).toHaveTextContent("Baseline: alns 20");

    fireEvent.click(screen.getByLabelText("Type a value"));
    fireEvent.change(screen.getByLabelText("Typed baseline"), { target: { value: "12" } });
    expect(screen.getByTestId("corpus-baseline")).toHaveTextContent("12");
    expect(screen.getByTestId("corpus-baseline")).toHaveTextContent("Typed baseline");

    fireEvent.change(screen.getByLabelText("Typed baseline"), { target: { value: "" } });
    expect(useSimulationCorpusStore.getState().typedBaseline).toBeNull();
    expect(screen.getByTestId("corpus-baseline")).toHaveTextContent("Baseline: No baseline");
    expect(screen.getByTestId("corpus-baseline")).not.toHaveTextContent(/\b0\b/);
  });

  it("hides a run from the list", async () => {
    scanSimulationCorpus.mockResolvedValue(scan);
    await useSimulationCorpusStore.getState().addRoots(["/data/scenario"]);
    render(<CorpusStrip />);

    fireEvent.click(screen.getByRole("button", { name: "Hide alns" }));
    expect(screen.queryByRole("checkbox", { name: "alns" })).not.toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "one" })).toBeInTheDocument();
  });

  it("says the corpus is empty without throwing", async () => {
    scanSimulationCorpus.mockResolvedValue({
      roots: ["/empty"],
      files: [],
      kept: 0,
      skipped: 2,
      failed: [],
    });
    await useSimulationCorpusStore.getState().addRoots(["/empty"]);
    render(<CorpusStrip />);
    expect(screen.getByText("The corpus is empty.")).toBeInTheDocument();
    expect(screen.getByTestId("corpus-counts")).toHaveTextContent("Kept 0, skipped 2, failed 0.");
    expect(screen.getByTestId("corpus-baseline")).toHaveTextContent("No baseline");
  });
});
