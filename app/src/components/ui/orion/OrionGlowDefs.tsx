import React from "react";
import { NOC_COLORS } from "../telemetry/tokens";

export interface OrionGlowDefsProps {
  idPrefix?: string;
}

/**
 * Reusable SVG `<defs>` sprite sheet for Orion scientific widgets (Slice V3).
 *
 * Uses strictly `<radialGradient>` and `<linearGradient>` stops so that
 * WebKitGTK on Linux renders at 60 FPS with zero `<feGaussianBlur>` GPU cost.
 */
export const OrionGlowDefs: React.FC<OrionGlowDefsProps> = ({
  idPrefix = "orion",
}) => {
  return (
    <defs data-testid="orion-glow-defs">
      <radialGradient id={`${idPrefix}-glow-cyan`} cx="50%" cy="50%" r="50%">
        <stop offset="0%" stopColor={NOC_COLORS.cyan} stopOpacity={0.55} />
        <stop offset="65%" stopColor={NOC_COLORS.cyan} stopOpacity={0.18} />
        <stop offset="100%" stopColor={NOC_COLORS.cyan} stopOpacity={0.0} />
      </radialGradient>

      <radialGradient id={`${idPrefix}-glow-ok`} cx="50%" cy="50%" r="50%">
        <stop offset="0%" stopColor={NOC_COLORS.ok} stopOpacity={0.55} />
        <stop offset="65%" stopColor={NOC_COLORS.ok} stopOpacity={0.18} />
        <stop offset="100%" stopColor={NOC_COLORS.ok} stopOpacity={0.0} />
      </radialGradient>

      <radialGradient id={`${idPrefix}-glow-warn`} cx="50%" cy="50%" r="50%">
        <stop offset="0%" stopColor={NOC_COLORS.warn} stopOpacity={0.55} />
        <stop offset="65%" stopColor={NOC_COLORS.warn} stopOpacity={0.18} />
        <stop offset="100%" stopColor={NOC_COLORS.warn} stopOpacity={0.0} />
      </radialGradient>

      <radialGradient id={`${idPrefix}-glow-crit`} cx="50%" cy="50%" r="50%">
        <stop offset="0%" stopColor={NOC_COLORS.crit} stopOpacity={0.62} />
        <stop offset="65%" stopColor={NOC_COLORS.crit} stopOpacity={0.22} />
        <stop offset="100%" stopColor={NOC_COLORS.crit} stopOpacity={0.0} />
      </radialGradient>

      <linearGradient
        id={`${idPrefix}-fill-cyan`}
        x1="0%"
        y1="0%"
        x2="0%"
        y2="100%"
      >
        <stop offset="0%" stopColor={NOC_COLORS.cyan} stopOpacity={0.38} />
        <stop offset="100%" stopColor={NOC_COLORS.indigo} stopOpacity={0.06} />
      </linearGradient>

      <linearGradient
        id={`${idPrefix}-fill-emerald`}
        x1="0%"
        y1="0%"
        x2="0%"
        y2="100%"
      >
        <stop offset="0%" stopColor={NOC_COLORS.emerald} stopOpacity={0.38} />
        <stop offset="100%" stopColor={NOC_COLORS.teal} stopOpacity={0.06} />
      </linearGradient>

      <linearGradient
        id={`${idPrefix}-hex-optimal`}
        x1="0%"
        y1="0%"
        x2="100%"
        y2="100%"
      >
        <stop offset="0%" stopColor={NOC_COLORS.emerald} stopOpacity={0.85} />
        <stop offset="100%" stopColor={NOC_COLORS.cyan} stopOpacity={0.75} />
      </linearGradient>

      <linearGradient
        id={`${idPrefix}-hex-nominal`}
        x1="0%"
        y1="0%"
        x2="100%"
        y2="100%"
      >
        <stop offset="0%" stopColor={NOC_COLORS.cyan} stopOpacity={0.65} />
        <stop offset="100%" stopColor={NOC_COLORS.indigo} stopOpacity={0.55} />
      </linearGradient>

      <linearGradient
        id={`${idPrefix}-hex-warn`}
        x1="0%"
        y1="0%"
        x2="100%"
        y2="100%"
      >
        <stop offset="0%" stopColor={NOC_COLORS.amber} stopOpacity={0.8} />
        <stop offset="100%" stopColor={NOC_COLORS.orange} stopOpacity={0.7} />
      </linearGradient>

      <linearGradient
        id={`${idPrefix}-hex-crit`}
        x1="0%"
        y1="0%"
        x2="100%"
        y2="100%"
      >
        <stop offset="0%" stopColor={NOC_COLORS.crit} stopOpacity={0.9} />
        <stop offset="100%" stopColor={NOC_COLORS.pink} stopOpacity={0.75} />
      </linearGradient>
    </defs>
  );
};

export default OrionGlowDefs;
