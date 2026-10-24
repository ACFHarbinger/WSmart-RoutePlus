import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import {
  OrionConstellationGraph,
  OrionOrbitalBubbleChart,
  OrionHoneycombMap,
  OrionRadialRingMeter,
  getSampleRoutingOrionData,
} from "../../src/components/ui/orion";

vi.mock("echarts-for-react", () => ({
  default: ({ option }: { option: unknown }) => (
    <div data-testid="mock-echarts" data-option={JSON.stringify(option)} />
  ),
}));

describe("SetProduct Orion Visualizations Suite (Slice V3)", () => {
  const sample = getSampleRoutingOrionData();

  it("provides realistic WSmart-Route telemetry fixtures from sample generator", () => {
    expect(sample.radarCohort.length).toBeGreaterThanOrEqual(5);
    expect(sample.constellationNodes.length).toBeGreaterThanOrEqual(5);
    expect(sample.constellationLinks.length).toBeGreaterThanOrEqual(4);
    expect(sample.orbitalBubbles.length).toBeGreaterThanOrEqual(4);
    expect(sample.honeycombZones.length).toBeGreaterThanOrEqual(5);
    expect(sample.radialRings.length).toBeGreaterThanOrEqual(3);
    expect(sample.dotMatrixCells.length).toBeGreaterThanOrEqual(20);
  });

  it("renders OrionConstellationGraph with interactive node selection and telemetry drawer", () => {
    const onSelect = vi.fn();
    render(
      <OrionConstellationGraph
        nodes={sample.constellationNodes}
        links={sample.constellationLinks}
        onSelectNode={onSelect}
      />
    );
    expect(screen.getByTestId("orion-constellation-graph")).toBeTruthy();

    const buttons = screen.getAllByRole("button");
    expect(buttons.length).toBeGreaterThanOrEqual(5);

    // Click on worker node
    const workerNode = screen.getByRole("button", { name: /Worker VRPP/i });
    fireEvent.click(workerNode);
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ id: "worker-vrpp", label: "Worker VRPP" })
    );
  });

  it("renders OrionOrbitalBubbleChart with laser guide ray and HUD detail card", () => {
    const onSelect = vi.fn();
    render(
      <OrionOrbitalBubbleChart
        bubbles={sample.orbitalBubbles}
        onSelectBubble={onSelect}
      />
    );
    expect(screen.getByTestId("orion-orbital-bubble-chart")).toBeTruthy();

    const bubbleBtn = screen.getByRole("button", { name: /Trial #104/i });
    fireEvent.click(bubbleBtn);
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ id: "trial-104", value: 1890 })
    );
  });

  it("renders OrionHoneycombMap with severity filters and depot zones", () => {
    const onSelect = vi.fn();
    render(
      <OrionHoneycombMap
        zones={sample.honeycombZones}
        trajectories={sample.zoneTrajectories}
        onSelectZone={onSelect}
      />
    );
    expect(screen.getByTestId("orion-honeycomb-map")).toBeTruthy();

    // Check filter pills
    const overflowFilter = screen.getByRole("button", { name: /overflow/i });
    fireEvent.click(overflowFilter);

    // Click on a zone
    const zoneBtn = screen.getByRole("button", { name: /Sul/i });
    fireEvent.click(zoneBtn);
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ id: "z-sul", label: "Sul" })
    );
  });

  it("renders OrionRadialRingMeter with composite readout and interactive ring hover", () => {
    const onHover = vi.fn();
    render(
      <OrionRadialRingMeter
        rings={sample.radialRings}
        centerValue="94.2%"
        centerLabel="COMPOSITE SCORE"
        onHoverRing={onHover}
      />
    );
    expect(screen.getByTestId("orion-radial-ring-meter")).toBeTruthy();
    expect(screen.getByText("94.2%")).toBeTruthy();
    expect(screen.getByText("COMPOSITE SCORE")).toBeTruthy();

    const fleetCard = screen.getByText("Fleet Capacity").closest('[role="button"]');
    expect(fleetCard).toBeTruthy();
    fireEvent.click(fleetCard!);
    expect(onHover).toHaveBeenCalledWith("fleet-cap");
  });
});
