import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PlacementFields, parseTargets } from "../../src/components/cloud/PlacementFields";

describe("PlacementFields (CR-09)", () => {
  it("parses one target per non-empty line", () => {
    expect(parseTargets(" gcp/batch:2 \n\naws/vm\n")).toEqual(["gcp/batch:2", "aws/vm"]);
  });

  it("reports priority, targets and pool", () => {
    const onChange = vi.fn();
    render(<PlacementFields priority={0} targets="" pool={false} onChange={onChange} />);
    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "3.7" } });
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "gcp/batch" } });
    fireEvent.click(screen.getByRole("checkbox"));
    expect(onChange.mock.calls).toEqual([[{ priority: 3 }], [{ targets: "gcp/batch" }], [{ pool: true }]]);
  });
});
