import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { OrionGlowDefs } from "../../src/components/ui/orion/OrionGlowDefs";
import { OrionDotMatrixHeatmap } from "../../src/components/ui/orion/OrionDotMatrixHeatmap";
import { OrionHoneycombMatrix } from "../../src/components/ui/orion/OrionHoneycombMatrix";
import { OrionRadarChart } from "../../src/components/ui/orion/OrionRadarChart";

vi.mock("echarts-for-react", () => ({
  default: ({ option }: { option: unknown }) => (
    <div data-testid="mock-echarts" data-option={JSON.stringify(option)} />
  ),
}));

describe("Orion High-Density Scientific Visualizations (Slice V3)", () => {
  it("renders OrionGlowDefs without any feGaussianBlur nodes", () => {
    const { container } = render(
      <svg>
        <OrionGlowDefs idPrefix="test" />
      </svg>
    );
    expect(screen.getByTestId("orion-glow-defs")).toBeTruthy();
    expect(container.querySelector("feGaussianBlur")).toBeNull();
  });

  it("renders OrionDotMatrixHeatmap with overflow counts and peak fill", () => {
    render(
      <OrionDotMatrixHeatmap
        dayLabels={["D1", "D2", "D3"]}
        binLabels={["B1", "B2"]}
        matrix={[
          [0.45, 0.75],
          [0.85, 1.08],
          [0.5, 1.15],
        ]}
      />
    );
    expect(screen.getByTestId("orion-dot-matrix-heatmap")).toBeTruthy();
    expect(screen.getByText("115%")).toBeTruthy(); // Peak fill 1.15 -> 115%
    expect(screen.getByText("2")).toBeTruthy(); // 2 cells >= 1.0
  });

  it("renders OrionHoneycombMatrix with keyboard focusable gridcells", () => {
    const onSelect = vi.fn();
    render(
      <OrionHoneycombMatrix
        cells={[
          {
            id: "c1",
            row: 0,
            col: 0,
            label: "LR=1e-4",
            xParam: "LR 1e-4",
            yParam: "KL 0.01",
            value: 1420.5,
            unit: "€",
          },
          {
            id: "c2",
            row: 0,
            col: 1,
            label: "LR=5e-4",
            xParam: "LR 5e-4",
            yParam: "KL 0.01",
            value: 1580.2,
            unit: "€",
          },
        ]}
        onSelectCell={onSelect}
      />
    );
    const cells = screen.getAllByRole("gridcell");
    expect(cells).toHaveLength(2);
    fireEvent.keyDown(cells[1]!, { key: "Enter" });
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ id: "c2", value: 1580.2 })
    );
  });

  it("renders OrionRadarChart capped at 4 policies by default with interactive toggles", () => {
    const cohort = ["bpc", "hgs", "alns", "swc_tcf", "pg_clns", "sans"].map(
      (p, i) => ({
        policy: p,
        means: {
          profit: 1500 - i * 50,
          km: 80 + i * 5,
          kg: 2400 - i * 40,
          "kg/km": 30 - i,
          overflows: i,
          cost: 200 + i * 10,
        },
      })
    );
    render(<OrionRadarChart cohort={cohort} />);
    expect(screen.getByText(/Showing 4 \/ 6 policies/i)).toBeTruthy();

    const sansBtn = screen.getByRole("button", { name: /sans/i });
    fireEvent.click(sansBtn);
    expect(screen.getByText(/Showing 5 \/ 6 policies/i)).toBeTruthy();
  });
});
