import React, { useId, useState } from "react";
import { NOC_COLORS, type NocTone } from "../telemetry/tokens";
import { nocPanelStyle } from "../telemetry/panel";

export interface RadialRingSpec {
  id: string;
  label: string;
  value: number; // e.g. 0..100+
  max?: number;
  unit?: string;
  tone?: NocTone;
  color?: string;
  warnAt?: number;
  critAt?: number;
}

export interface OrionRadialRingMeterProps {
  title?: string;
  subtitle?: string;
  rings: readonly RadialRingSpec[];
  centerValue: string | number;
  centerLabel: string;
  size?: number;
  onHoverRing?: (ringId: string | null) => void;
}

const TONE_COLOR: Record<NocTone, string> = {
  ok: NOC_COLORS.ok,
  warn: NOC_COLORS.warn,
  crit: NOC_COLORS.crit,
  idle: NOC_COLORS.idle,
  cyan: NOC_COLORS.cyan,
  indigo: NOC_COLORS.indigoText,
  purple: NOC_COLORS.purpleText,
  amber: NOC_COLORS.amber,
};

export const OrionRadialRingMeter: React.FC<OrionRadialRingMeterProps> = ({
  title = "Telemetry Multi-Ring Meter",
  subtitle = "Resource capacity, GPU load, & fleet tour completion",
  rings,
  centerValue,
  centerLabel,
  size = 280,
  onHoverRing,
}) => {
  const gradId = useId().replace(/:/g, "");
  const [hoveredRingId, setHoveredRingId] = useState<string | null>(null);

  const cx = size / 2;
  const cy = size / 2;
  const strokeWidth = 10;
  const ringGap = 16;
  const outerRadius = size / 2 - strokeWidth - 16;

  const handleHover = (id: string | null) => {
    setHoveredRingId(id);
    onHoverRing?.(id);
  };

  return (
    <div
      className="rounded-xl border p-4 space-y-3"
      style={nocPanelStyle}
      data-testid="orion-radial-ring-meter"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-noc-textPrimary">
            {title}
          </h3>
          <p className="text-[11px] text-noc-textMuted">{subtitle}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
        {/* SVG Concentric Gauge */}
        <div className="flex justify-center select-none">
          <svg
            width={size}
            height={size}
            viewBox={`0 0 ${size} ${size}`}
            role="img"
            aria-label={`${centerLabel}: ${centerValue}`}
          >
            <defs>
              <filter id={`${gradId}-ring-glow`} x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor={NOC_COLORS.cyan} floodOpacity="0.4" />
              </filter>
            </defs>

            {/* Outer Perimeter Degree Ticks */}
            {Array.from({ length: 36 }).map((_, i) => {
              const deg = i * 10;
              const rad = (deg * Math.PI) / 180;
              const r1 = size / 2 - 6;
              const r2 = size / 2 - (i % 9 === 0 ? 12 : 9);
              const x1 = cx + r1 * Math.cos(rad);
              const y1 = cy + r1 * Math.sin(rad);
              const x2 = cx + r2 * Math.cos(rad);
              const y2 = cy + r2 * Math.sin(rad);
              return (
                <line
                  key={deg}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke="var(--noc-border-strong)"
                  strokeWidth="1"
                />
              );
            })}

            {/* Concentric Arc Rings */}
            {rings.map((ring, idx) => {
              const r = outerRadius - idx * ringGap;
              const circumference = 2 * Math.PI * r;
              const safeMax = ring.max && ring.max > 0 ? ring.max : 100;
              const ratio = Math.max(0, Math.min(1, ring.value / safeMax));
              const dashOffset = circumference * (1 - ratio);

              const isHovered = hoveredRingId === ring.id;
              const isDimmed = hoveredRingId !== null && !isHovered;

              const isCrit = ring.critAt !== undefined && ring.value >= ring.critAt;
              const isWarn = !isCrit && ring.warnAt !== undefined && ring.value >= ring.warnAt;
              const baseColor =
                ring.color ?? (ring.tone ? TONE_COLOR[ring.tone] : [NOC_COLORS.cyan, NOC_COLORS.indigoText, NOC_COLORS.purpleText][idx % 3]);
              const strokeColor = isCrit ? NOC_COLORS.crit : isWarn ? NOC_COLORS.warn : baseColor;

              return (
                <g
                  key={ring.id}
                  opacity={isDimmed ? 0.3 : 1}
                  className="transition-opacity duration-200"
                  onMouseEnter={() => handleHover(ring.id)}
                onMouseOver={() => handleHover(ring.id)}
                  onMouseLeave={() => handleHover(null)}
                >
                  {/* Track Background */}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={r}
                    fill="none"
                    stroke="var(--noc-border)"
                    strokeWidth={strokeWidth}
                  />
                  {/* Active Value Arc */}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={r}
                    fill="none"
                    stroke={strokeColor}
                    strokeWidth={isHovered ? strokeWidth + 2 : strokeWidth}
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={dashOffset}
                    transform={`rotate(-90 ${cx} ${cy})`}
                    filter={isHovered ? `url(#${gradId}-ring-glow)` : undefined}
                  />
                </g>
              );
            })}

            {/* Center Composite KPI Readout */}
            <text
              x={cx}
              y={cy - 4}
              textAnchor="middle"
              dominantBaseline="central"
              fill="var(--noc-text-primary)"
              fontSize="20"
              fontWeight="bold"
              className="tabular-nums"
              style={{ fontVariantNumeric: "tabular-nums" }}
            >
              {centerValue}
            </text>
            <text
              x={cx}
              y={cy + 16}
              textAnchor="middle"
              dominantBaseline="central"
              fill="var(--noc-text-muted)"
              fontSize="10"
              fontWeight="bold"
              className="uppercase tracking-wider"
            >
              {centerLabel}
            </text>
          </svg>
        </div>

        {/* Interactive Legend Cards */}
        <div className="space-y-2">
          {rings.map((ring, idx) => {
            const isHovered = hoveredRingId === ring.id;
            const safeMax = ring.max && ring.max > 0 ? ring.max : 100;
            const pct = Math.round((ring.value / safeMax) * 100);
            const isCrit = ring.critAt !== undefined && ring.value >= ring.critAt;
            const isWarn = !isCrit && ring.warnAt !== undefined && ring.value >= ring.warnAt;
            const baseColor =
              ring.color ?? (ring.tone ? TONE_COLOR[ring.tone] : [NOC_COLORS.cyan, NOC_COLORS.indigoText, NOC_COLORS.purpleText][idx % 3]);
            const color = isCrit ? NOC_COLORS.crit : isWarn ? NOC_COLORS.warn : baseColor;

            return (
              <div
                key={ring.id}
                role="button"
                tabIndex={0}
                onMouseEnter={() => handleHover(ring.id)}
                onMouseOver={() => handleHover(ring.id)}
                onMouseLeave={() => handleHover(null)}
                onClick={() => handleHover(isHovered ? null : ring.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    handleHover(isHovered ? null : ring.id);
                  }
                }}
                style={{
                  borderColor: isHovered ? color : "var(--noc-border)",
                  backgroundColor: isHovered ? `${color}18` : "var(--noc-bg-card)",
                }}
                className="p-2.5 rounded-lg border transition-all cursor-pointer flex items-center justify-between text-xs"
              >
                <div className="flex items-center gap-2">
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: color }}
                  />
                  <div>
                    <div className="font-medium text-noc-textPrimary">{ring.label}</div>
                    <div className="text-[10px] text-noc-textMuted font-mono">
                      Ring {idx + 1}
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <div className="font-mono font-bold text-noc-textPrimary tabular-nums">
                    {ring.value.toLocaleString(undefined, { maximumFractionDigits: 1 })}
                    {ring.unit ? ` ${ring.unit}` : ""}
                  </div>
                  <div className="text-[10px] text-noc-textMuted tabular-nums">
                    {pct}% of {safeMax}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
