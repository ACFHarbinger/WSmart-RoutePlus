import { render, screen, cleanup } from "@testing-library/react";
import { describe, expect, it, afterEach } from "vitest";
import { CloudStatusBadge } from "../../src/components/cloud/CloudStatusBadge";

describe("CloudStatusBadge (CR-15)", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders running state with pulsing dot", () => {
    const { container } = render(<CloudStatusBadge state="running" />);
    expect(screen.getByText("Running")).toBeInTheDocument();
    expect(container.querySelector(".animate-pulse")).not.toBeNull();
  });

  it("renders succeeded state correctly", () => {
    const { container } = render(<CloudStatusBadge state="succeeded" />);
    expect(screen.getByText("Succeeded")).toBeInTheDocument();
    expect(container.querySelector(".animate-pulse")).toBeNull();
  });

  it("renders failed and cancelled states", () => {
    const { rerender } = render(<CloudStatusBadge state="failed" />);
    expect(screen.getByText("Failed")).toBeInTheDocument();

    rerender(<CloudStatusBadge state="cancelled" />);
    expect(screen.getByText("Cancelled")).toBeInTheDocument();
  });

  it("renders task-specific states (placing, starting, retrying, preempted)", () => {
    const { rerender, container } = render(<CloudStatusBadge state="placing" />);
    expect(screen.getByText("Placing")).toBeInTheDocument();
    expect(container.querySelector(".animate-pulse")).not.toBeNull();

    rerender(<CloudStatusBadge state="preempted" />);
    expect(screen.getByText("Preempted")).toBeInTheDocument();

    rerender(<CloudStatusBadge state="retrying" />);
    expect(screen.getByText("Retrying")).toBeInTheDocument();
  });
});
