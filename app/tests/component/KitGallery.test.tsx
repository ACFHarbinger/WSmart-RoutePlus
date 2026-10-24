import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { KitGallery } from "../../src/components/ui/telemetry/KitGallery";
import {
  normalizeCohortMetric,
  SIM_METRICS,
} from "../../src/components/ui/telemetry/metrics";
import { StalenessDot } from "../../src/components/ui/telemetry/status";

describe("Telemetry Primitives & KitGallery", () => {
  it("renders KitGallery and switches subtabs via click and keyboard", () => {
    render(<KitGallery />);
    expect(screen.getByTestId("kit-gallery")).toBeTruthy();
    expect(screen.getByText("NOC Telemetry Kit Gallery")).toBeTruthy();

    const compareTab = screen.getByRole("tab", { name: /Compare/i });
    fireEvent.click(compareTab);
    expect(compareTab.getAttribute("aria-selected")).toBe("true");
  });

  it("computes StalenessDot states based on lastEventAt delta", () => {
    const now = Date.now();
    const { container, rerender } = render(
      <StalenessDot lastEventAt={now - 1000} />
    );
    expect(container.textContent).toMatch(/live •/i);

    rerender(<StalenessDot lastEventAt={now - 30000} staleAfterSec={10} />);
    expect(container.textContent).toMatch(/stale •/i);

    rerender(<StalenessDot lastEventAt={null} />);
    expect(container.textContent).toMatch(/awaiting stream/i);
  });

  it("normalizes cohort metrics with universal 1.0 = best direction contract", () => {
    // Higher is better (profit): [100, 150, 200] -> [0.15, 0.575, 1.0]
    const profitNorm = normalizeCohortMetric([100, 150, 200], true);
    expect(profitNorm[2]).toBeCloseTo(1.0);
    expect(profitNorm[0]).toBeCloseTo(0.15);

    // Lower is better (km, overflows, cost): [50, 75, 100] -> [1.0, 0.575, 0.15]
    const kmNorm = normalizeCohortMetric([50, 75, 100], false);
    expect(kmNorm[0]).toBeCloseTo(1.0);
    expect(kmNorm[2]).toBeCloseTo(0.15);
    expect(SIM_METRICS.overflows.higherIsBetter).toBe(false);
    expect(SIM_METRICS.profit.higherIsBetter).toBe(true);
  });
});
