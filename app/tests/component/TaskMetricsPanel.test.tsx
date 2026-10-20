import { render, screen, cleanup } from "@testing-library/react";
import { describe, expect, it, afterEach } from "vitest";
import { TaskMetricsPanel, type TaskMetricsData } from "../../src/components/cloud/TaskMetricsPanel";

describe("TaskMetricsPanel (CR-15 / CR-12)", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders empty state when metrics prop is null", () => {
    render(<TaskMetricsPanel metrics={null} />);
    expect(screen.getByTestId("task-metrics-empty")).toBeInTheDocument();
    expect(
      screen.getByText(/No resource samples for this task yet/i)
    ).toBeInTheDocument();
    expect(screen.getByText("No samples yet")).toBeInTheDocument();
  });

  it("renders empty state when points array is empty", () => {
    render(<TaskMetricsPanel metrics={{ task_id: "task-01", points: [] }} />);
    expect(screen.getByTestId("task-metrics-empty")).toBeInTheDocument();
  });

  it("renders active metrics when telemetry points are provided", () => {
    const data: TaskMetricsData = {
      task_id: "task-01",
      points: [
        {
          timestamp_ms: 1000,
          cpu_percent: 45.2,
          memory_rss_mb: 2048,
          disk_io_mbps: 12.4,
        },
      ],
    };
    render(<TaskMetricsPanel metrics={data} />);
    expect(screen.getByTestId("task-metrics-active")).toBeInTheDocument();
    expect(screen.getByText(/Task Resource Telemetry: task-01/)).toBeInTheDocument();
    expect(screen.getByText("45.2%")).toBeInTheDocument();
    expect(screen.getByText("2048 MB")).toBeInTheDocument();
    expect(screen.getByText("12.4 MB/s")).toBeInTheDocument();
  });
});
