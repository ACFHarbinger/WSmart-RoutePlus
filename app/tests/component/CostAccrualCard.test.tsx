import { render, screen, cleanup } from "@testing-library/react";
import { describe, expect, it, afterEach } from "vitest";
import { CostAccrualCard } from "../../src/components/cloud/CostAccrualCard";
import type { CloudJobAnalysis } from "../../src/types/cloud";

describe("CostAccrualCard (CR-16 / CR-10)", () => {
  afterEach(() => {
    cleanup();
  });

  const mockBaseAnalysis: CloudJobAnalysis = {
    job_id: "job-test-123",
    name: "Rio Maior 30d Evaluation",
    state: "succeeded",
    provider: "aws",
    compute: "batch",
    spot: true,
    machine_type: "c5.xlarge",
    region: "us-east-1",
    total_tasks: 4,
    succeeded_tasks: 4,
    failed_tasks: 0,
    cancelled_tasks: 0,
    preempted_tasks: 0,
    total_wall_seconds: 120.0,
    total_compute_seconds: 400.0,
    accrued_cost_usd: 0.0425,
    estimated_cost_usd: null,
    on_demand_baseline_cost_usd: 0.125,
    spot_savings_usd: 0.0825,
    spot_savings_pct: 66.0,
    tasks: [],
  };

  it("renders pending calibration when estimated_cost_usd is null", () => {
    render(<CostAccrualCard analysis={mockBaseAnalysis} />);
    expect(screen.getByText(/Cost & Budget Accrual \(CR-10\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Unbounded \/ No Cap/i)).toBeInTheDocument();
    expect(screen.getByText(/Pending CR-10 model calibration/i)).toBeInTheDocument();
    expect(screen.getByText(/\$0\.0425/i)).toBeInTheDocument();
  });

  it("renders budget cap and within estimate indicator when estimate is provided", () => {
    const analysisWithEst: CloudJobAnalysis = {
      ...mockBaseAnalysis,
      estimated_cost_usd: 0.05,
    };
    render(<CostAccrualCard analysis={analysisWithEst} />);
    expect(screen.getByText(/\$0\.0500/i)).toBeInTheDocument();
    expect(screen.getByText(/Within estimate/i)).toBeInTheDocument();
  });

  it("renders spot discount badge and spot savings percentage", () => {
    render(<CostAccrualCard analysis={mockBaseAnalysis} />);
    expect(screen.getByText(/Spot Discounted/i)).toBeInTheDocument();
    expect(screen.getByText(/-\$0\.0825/i)).toBeInTheDocument();
    expect(screen.getByText(/66\.0% savings vs on-demand/i)).toBeInTheDocument();
  });

  it("renders on-demand tier info when spot is false", () => {
    const onDemandAnalysis: CloudJobAnalysis = {
      ...mockBaseAnalysis,
      spot: false,
      spot_savings_usd: 0,
      spot_savings_pct: 0,
    };
    render(<CostAccrualCard analysis={onDemandAnalysis} />);
    expect(screen.queryByText(/Spot Discounted/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Standard On-Demand/i)).toBeInTheDocument();
  });
});
