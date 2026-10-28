import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AlgorithmBuilder } from "../../src/pages/algorithms/builder/AlgorithmBuilder";
import { draftPersist, readDraft } from "../../src/pages/algorithms/builder/draftBridge";
import { exportCompositionYaml } from "../../src/pages/algorithms/builder/exportCompositionYaml";
import { useAppStore } from "../../src/store/app";
import type { PolicyCatalog } from "../../src/types/algorithms";
import catalogJson from "../fixtures/algorithm-catalog.json";

const catalog = catalogJson as PolicyCatalog;

function resetDraft() {
  const draft = readDraft();
  if (typeof draft.clear === "function") draft.clear();
  draft.setDraft({
    packing: null,
    routing: null,
    model: null,
    routeImprovement: null,
    acceptance: null,
    operators: [],
    problem: "ptp",
    graphSize: 20,
    seed: 1,
  });
}

async function hydrationDone() {
  const persist = draftPersist();
  if (!persist || persist.hasHydrated()) return;
  await new Promise<void>((resolve) => {
    persist.onFinishHydration(() => resolve());
  });
}

beforeEach(async () => {
  await hydrationDone();
  resetDraft();
  useAppStore.getState().setMode("simulation");
});

afterEach(() => {
  cleanup();
});

function renderBuilder() {
  return render(<AlgorithmBuilder initialCatalog={catalog} />);
}

describe("AlgorithmBuilder", () => {
  it("names packing and routing on an empty canvas", () => {
    renderBuilder();
    const empty = screen.getByText(/no modules on the canvas/i);
    expect(empty).toHaveTextContent(/packing/i);
    expect(empty).toHaveTextContent(/routing/i);
    expect(empty).toHaveTextContent(/acceptance/i);
    expect(empty).toHaveTextContent(/operators/i);
  });

  it("exports regular, alns, and oi after those modules are connected", () => {
    renderBuilder();
    fireEvent.click(screen.getByRole("button", { name: "Add regular" }));
    fireEvent.click(screen.getByRole("button", { name: "Add alns" }));
    fireEvent.click(screen.getByRole("button", { name: "Add oi" }));
    fireEvent.click(screen.getByRole("button", { name: "Connect regular" }));
    fireEvent.click(screen.getByRole("button", { name: "Connect alns" }));
    fireEvent.click(screen.getByRole("button", { name: "Connect oi" }));
    fireEvent.click(screen.getByRole("button", { name: "Connect alns" }));

    const yaml = (screen.getByTestId("composition-yaml") as HTMLTextAreaElement).value;
    expect(yaml).toContain("regular");
    expect(yaml).toContain("alns");
    expect(yaml).toContain("oi");
    expect(yaml).toBe(
      exportCompositionYaml({
        problem: "ptp",
        graphSize: 20,
        seed: 1,
        nodes: [
          { uid: "regular-1", entryId: "regular", family: "packing", displayName: "regular", parameters: {} },
          { uid: "alns-2", entryId: "alns", family: "routing", displayName: "alns", parameters: {} },
          { uid: "oi-3", entryId: "oi", family: "acceptance", displayName: "oi", parameters: {} },
        ],
        edges: [
          { source: "regular-1", target: "alns-2" },
          { source: "oi-3", target: "alns-2" },
        ],
      })
    );
  });

  it("refuses a separation link to last_minute and names the missing runner", () => {
    renderBuilder();
    fireEvent.click(screen.getByRole("button", { name: "Add capacity_separation" }));
    fireEvent.click(screen.getByRole("button", { name: "Add last_minute" }));
    fireEvent.click(screen.getByRole("button", { name: "Connect capacity_separation" }));
    fireEvent.click(screen.getByRole("button", { name: "Connect last_minute" }));

    expect(screen.getByRole("alert").textContent?.toLowerCase()).toMatch(/refus|packing|last_minute|cut|separat/);
    expect(screen.queryAllByTestId("builder-edge")).toHaveLength(0);
    expect(screen.getByRole("button", { name: "Export" })).toBeDisabled();
    expect(screen.getByText(/capacity_separation has no runner/i)).toBeInTheDocument();
  });

  it("allows a separation link onto bpc", () => {
    renderBuilder();
    fireEvent.click(screen.getByRole("button", { name: "Add capacity_separation" }));
    fireEvent.click(screen.getByRole("button", { name: "Add bpc" }));
    fireEvent.click(screen.getByRole("button", { name: "capacity_separation output port" }));
    fireEvent.click(screen.getByRole("button", { name: "bpc input port" }));
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getAllByTestId("builder-edge")).toHaveLength(1);
  });

  it("shows a catalog error and does not place that module", () => {
    renderBuilder();
    fireEvent.click(screen.getByRole("button", { name: "Add broken_pack" }));
    expect(screen.getByRole("alert")).toHaveTextContent("ImportError: fixture module missing");
    expect(screen.queryByTestId("canvas-node-broken_pack")).toBeNull();
  });

  it("does not place WeightAdjustmentRNN", () => {
    renderBuilder();
    expect(screen.getByText(/not a route policy/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add WeightAdjustmentRNN" })).toBeDisabled();
  });

  it("restores packing, routing, and acceptance from the draft store", () => {
    readDraft().setDraft({ packing: "regular", routing: "alns", acceptance: "oi" });
    renderBuilder();
    expect(screen.getByTestId("canvas-node-regular")).toBeInTheDocument();
    expect(screen.getByTestId("canvas-node-alns")).toBeInTheDocument();
    expect(screen.getByTestId("canvas-node-oi")).toBeInTheDocument();
  });

  it("opens the visualizer with the same draft scalars", () => {
    renderBuilder();
    fireEvent.click(screen.getByRole("button", { name: "Add regular" }));
    fireEvent.click(screen.getByRole("button", { name: "Add alns" }));
    fireEvent.click(screen.getByRole("button", { name: "Open in visualizer" }));
    expect(useAppStore.getState().mode).toBe("algorithm_visualizer");
    const draft = readDraft();
    expect(draft.packing).toBe("regular");
    expect(draft.routing).toBe("alns");
  });
});
