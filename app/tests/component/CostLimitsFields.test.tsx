import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CostLimitsFields } from "../../src/components/cloud/CostLimitsFields";

describe("CostLimitsFields (CR-10)", () => {
  it("reports machine type, region and a non-negative budget", () => {
    const onChange = vi.fn();
    render(<CostLimitsFields machineType="" region="" budgetUsd={0} onChange={onChange} />);
    fireEvent.change(screen.getByPlaceholderText(/c6i.large/), { target: { value: "m6i.large" } });
    fireEvent.change(screen.getAllByPlaceholderText("backend default")[0], { target: { value: "eu-west-1" } });
    fireEvent.change(screen.getByPlaceholderText("0 = no cap"), { target: { value: "-5" } });
    fireEvent.change(screen.getByPlaceholderText("0 = no cap"), { target: { value: "12.5" } });
    expect(onChange.mock.calls).toEqual([
      [{ machineType: "m6i.large" }],
      [{ region: "eu-west-1" }],
      [{ budgetUsd: 0 }],
      [{ budgetUsd: 12.5 }],
    ]);
  });
});
