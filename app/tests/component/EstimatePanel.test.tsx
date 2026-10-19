import { render, screen, cleanup } from "@testing-library/react";
import { describe, expect, it, afterEach } from "vitest";
import { EstimatePanel } from "../../src/components/cloud/EstimatePanel";
import type { CloudJobEstimate } from "../../src/types/cloud";

describe("EstimatePanel (CR-14)", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders 'estimate unavailable' when estimate is null", () => {
    render(<EstimatePanel estimate={null} />);
    expect(screen.getByTestId("estimate-panel-unavailable")).toBeInTheDocument();
    expect(screen.getAllByText(/Estimate unavailable/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/CR-10 cost and ETA engine pending/i)).toBeInTheDocument();
  });

  it("renders 'estimate unavailable' when estimate is undefined", () => {
    render(<EstimatePanel />);
    expect(screen.getByTestId("estimate-panel-unavailable")).toBeInTheDocument();
    expect(screen.getAllByText(/Estimate unavailable/i).length).toBeGreaterThan(0);
  });

  it("renders 'estimate unavailable' when estimate contains only empty fields", () => {
    const emptyEstimate: CloudJobEstimate = {
      cost_usd: null,
      eta_seconds: null,
      memory_required_gb: null,
    };
    render(<EstimatePanel estimate={emptyEstimate} />);
    expect(screen.getByTestId("estimate-panel-unavailable")).toBeInTheDocument();
    expect(screen.getAllByText(/Estimate unavailable/i).length).toBeGreaterThan(0);
  });

  it("renders cost, runtime, and memory when an estimate is provided", () => {
    const validEstimate: CloudJobEstimate = {
      cost_usd: 12.5,
      eta_seconds: 3665, // 1h 1m
      memory_required_gb: 16,
      source: "GNN-Estimator-v1",
    };
    render(<EstimatePanel estimate={validEstimate} />);
    expect(screen.getByTestId("estimate-panel-available")).toBeInTheDocument();
    expect(screen.getByText("$12.50")).toBeInTheDocument();
    expect(screen.getByText("1h 1m")).toBeInTheDocument();
    expect(screen.getByText("16 GB")).toBeInTheDocument();
    expect(screen.getByText("GNN-Estimator-v1")).toBeInTheDocument();
  });
});
