import { describe, expect, it } from "vitest";
import { buildHpoHoneycomb } from "../../src/components/analysis/hpo/hpoHoneycomb";
import type { OptunaTrial } from "../../src/types";

function trial(
  number: number,
  lr: number,
  embed: number,
  value: number
): OptunaTrial {
  return {
    number,
    value,
    state: "COMPLETE",
    params: { lr, embed },
  };
}

describe("buildHpoHoneycomb", () => {
  it("places cells on ranked hyperparameter values, not trial order", () => {
    const config = buildHpoHoneycomb(
      [
        trial(3, 0.01, 128, 4),
        trial(1, 0.001, 64, 9),
        trial(2, 0.001, 64, 2),
      ],
      { lr: 0.8, embed: 0.4 }
    );
    expect(config).not.toBeNull();
    const cells = config!.cells;
    const lowLr = cells.find((cell) => cell.xParam === "lr=0.001");
    const highLr = cells.find((cell) => cell.xParam === "lr=0.01");
    expect(lowLr?.col).toBe(0);
    expect(highLr?.col).toBe(1);
    expect(lowLr?.label).toBe("#2");
    expect(lowLr?.value).toBe(2);
    expect(lowLr?.row).toBe(0);
    expect(cells.find((cell) => cell.xParam === "lr=0.01")?.row).toBe(1);
  });
});
