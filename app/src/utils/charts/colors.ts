export {
  ECHARTS_FILL_VISUAL_MAP,
  NOC_DIVERGING_DELTA,
  NOC_POLICY_PALETTE,
  POLICY_HEX_PALETTE,
  getBinFillColor,
  getDeltaColor,
  getPolicyColor,
  getPolicyRgb,
} from "./palettes";

/** Parse `#rrggbb` to deck.gl RGB tuple. */
export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
