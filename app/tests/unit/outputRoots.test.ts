import { describe, expect, it } from "vitest";
import { simOutputRoot, simOutputRoots } from "../../src/utils/runs/outputRoots";

describe("simulation output roots", () => {
  it("puts the current folder first and keeps the pre-2026-10-13 one", () => {
    expect(simOutputRoots("/repo/")).toEqual(["/repo/outputs/simulations", "/repo/assets/output"]);
  });

  it("joins without doubling separators", () => {
    expect(simOutputRoot("C:\\wsr\\")).toBe("C:\\wsr/outputs/simulations");
    expect(simOutputRoot("/repo")).toBe("/repo/outputs/simulations");
  });
});
