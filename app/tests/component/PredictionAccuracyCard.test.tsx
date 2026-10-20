import { render, screen, cleanup } from "@testing-library/react";
import { describe, expect, it, afterEach } from "vitest";
import { PredictionAccuracyCard } from "../../src/components/cloud/PredictionAccuracyCard";
import type { CloudJobAnalysis } from "../../src/types/cloud";

describe("PredictionAccuracyCard (CR-16 / CR-11)", () => {
  afterEach(() => {
    cleanup();
  });

  const mockBaseAnalysis: CloudJobAnalysis = {
    job_id: "job-pred-test",
    name: "Rio Maior Runtime Prediction Test",
    state: "succeeded",
    provider: "gcp",
    compute: "batch",
    spot: true,
    total_tasks: 2,
    succeeded_tasks: 2,
    failed_tasks: 0,
    cancelled_tasks: 0,
    preempted_tasks: 0,
    total_wall_seconds: 60.0,
    total_compute_seconds: 100.0,
    accrued_cost_usd: 0.015,
    tasks: [
      {
        task_id: "task-001",
        cell_id: "30d/riomaior/100/emp/bpc",
        state: "succeeded",
        attempt: 1,
        duration_seconds: 50.0,
        expected_wall_seconds: null,
        cost_usd: 0.0075,
      },
      {
        task_id: "task-002",
        cell_id: "30d/riomaior/100/emp/alns",
        state: "succeeded",
        attempt: 1,
        duration_seconds: 50.0,
        expected_wall_seconds: null,
        cost_usd: 0.0075,
      },
    ],
  };

  it("renders pending calibration empty state when tasks lack expected_wall_seconds", () => {
    render(<PredictionAccuracyCard analysis={mockBaseAnalysis} />);
    expect(screen.getByText(/Runtime Prediction Accuracy \(CR-11\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Pending CR-11 Calibration/i)).toBeInTheDocument();
    expect(screen.getByText(/No Pre-Run Runtime Predictions Available/i)).toBeInTheDocument();
    expect(screen.getByText(/100\.0s/i)).toBeInTheDocument();
  });

  it("renders accuracy metrics and calibration details when tasks contain predictions", () => {
    const calibratedAnalysis: CloudJobAnalysis = {
      ...mockBaseAnalysis,
      tasks: [
        {
          task_id: "task-001",
          cell_id: "30d/riomaior/100/emp/bpc",
          state: "succeeded",
          attempt: 1,
          duration_seconds: 50.0,
          expected_wall_seconds: 48.0, // within ~4% error
          cost_usd: 0.0075,
        },
        {
          task_id: "task-002",
          cell_id: "30d/riomaior/100/emp/alns",
          state: "succeeded",
          attempt: 1,
          duration_seconds: 52.0,
          expected_wall_seconds: 50.0, // within ~4% error
          cost_usd: 0.0075,
        },
      ],
    };

    render(<PredictionAccuracyCard analysis={calibratedAnalysis} />);
    expect(screen.getByText(/Calibrated \(2\/2 tasks\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Model Accuracy/i)).toBeInTheDocument();
    expect(screen.getByText(/Observed vs Predicted/i)).toBeInTheDocument();
    expect(screen.getByText(/Within ±15% Tolerance/i)).toBeInTheDocument();
    expect(screen.getByText(/2 \/ 2/i)).toBeInTheDocument();
    expect(screen.getByText(/100% within margin/i)).toBeInTheDocument();
  });
  it("does not cancel opposing prediction errors", () => {
    render(<PredictionAccuracyCard analysis={{ ...mockBaseAnalysis, tasks: [
      { ...mockBaseAnalysis.tasks[0], duration_seconds: 50, expected_wall_seconds: 100 },
      { ...mockBaseAnalysis.tasks[1], duration_seconds: 150, expected_wall_seconds: 100 },
    ] }} />);
    expect(screen.getByText("Mean absolute error: 66.7%")).toBeInTheDocument();
  });

});
