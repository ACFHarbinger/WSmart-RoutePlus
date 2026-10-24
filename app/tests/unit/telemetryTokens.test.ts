import { describe, expect, it } from "vitest";
import {
  NOC_COLORS,
  NOC_LIGHT_SURFACES,
  NOC_LIGHT_TEXT_COLORS,
  NOC_SURFACES,
  contrastRatio,
} from "../../src/components/ui/telemetry/tokens";
import {
  NOC_POLICY_PALETTE,
  getBinFillColor,
  getPolicyColor,
  getPolicyRgb,
} from "../../src/utils/charts/palettes";

describe("NOC Design Tokens & WCAG AA Contrast (Slice V1)", () => {
  it("dark-mode text tokens exceed 4.5:1 WCAG AA contrast on both bgDeep and bgPanel", () => {
    const darkTextTokens = [
      NOC_SURFACES.textPrimary,
      NOC_SURFACES.textSecondary,
      NOC_SURFACES.textMuted,
      NOC_COLORS.ok,
      NOC_COLORS.warn,
      NOC_COLORS.crit,
      NOC_COLORS.cyan,
      NOC_COLORS.teal,
      NOC_COLORS.indigoText,
      NOC_COLORS.purpleText,
      NOC_COLORS.pink,
      NOC_COLORS.amber,
      NOC_COLORS.orange,
      NOC_COLORS.emerald,
    ];
    for (const hex of darkTextTokens) {
      expect(contrastRatio(hex, NOC_SURFACES.bgDeep)).toBeGreaterThanOrEqual(4.5);
      expect(contrastRatio(hex, NOC_SURFACES.bgPanel)).toBeGreaterThanOrEqual(4.5);
    }
  });

  it("all dark-mode graphical stroke accents exceed 3.0:1 against bgDeep and bgPanel", () => {
    for (const hex of Object.values(NOC_COLORS)) {
      expect(contrastRatio(hex, NOC_SURFACES.bgDeep)).toBeGreaterThanOrEqual(3.0);
      expect(contrastRatio(hex, NOC_SURFACES.bgPanel)).toBeGreaterThanOrEqual(3.0);
    }
  });

  it("light-mode text tokens exceed 4.5:1 WCAG AA contrast on light bgDeep and bgPanel", () => {
    const lightTextTokens = [
      NOC_LIGHT_SURFACES.textPrimary,
      NOC_LIGHT_SURFACES.textSecondary,
      NOC_LIGHT_SURFACES.textMuted,
      ...Object.values(NOC_LIGHT_TEXT_COLORS),
    ];
    for (const hex of lightTextTokens) {
      expect(contrastRatio(hex, NOC_LIGHT_SURFACES.bgDeep)).toBeGreaterThanOrEqual(4.5);
      expect(contrastRatio(hex, NOC_LIGHT_SURFACES.bgPanel)).toBeGreaterThanOrEqual(4.5);
    }
  });

  it("provides 12 distinct CVD-safe policy colors keyed by real policy IDs", () => {
    expect(NOC_POLICY_PALETTE).toHaveLength(12);
    const uniqueHex = new Set(NOC_POLICY_PALETTE.map((p) => p.hex));
    expect(uniqueHex.size).toBe(12);
    expect(getPolicyColor("bpc")).toBe("#38bdf8");
    expect(getPolicyColor("hna")).toBe("#fb7185");
    expect(getPolicyColor("hna_emp")).toBe("#fb7185");
    expect(getPolicyColor("na")).toBe("#4ade80");
    expect(getPolicyColor("ms_bpc_sp")).toBe("#38bdf8");
    expect(getPolicyColor("abpc_hg")).toBe(getPolicyColor("not_a_policy"));
    expect(getPolicyColor("last_minute_cf70_hgs_cls_1N")).toBe("#f59e0b");
    expect(getPolicyRgb("alns")).toEqual([129, 140, 248]);
  });

  it("maps bin fill levels with a critical crimson inflection at >= 100% overflow", () => {
    expect(getBinFillColor(0.25)[0]).toBe(16); // Emerald
    expect(getBinFillColor(0.85)[0]).toBe(245); // Amber
    expect(getBinFillColor(1.0)[0]).toBe(239); // Crimson overflow
  });
});
