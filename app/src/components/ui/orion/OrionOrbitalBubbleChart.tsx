import React, { useId, useMemo, useState } from "react";
import { NOC_COLORS, type NocTone } from "../telemetry/tokens";
import { nocPanelStyle } from "../telemetry/panel";

export interface OrbitalBubble {
  id: string;
  label: string;
  orbitIdx: number; // 0..2 (inner, mid, outer)
  angleDeg: number; // 0..360
  radius: number; // 10..32
  value: number;
  unit?: string;
  tone?: NocTone;
  metrics?: Record<string, string | number>;
  trialMeta?: Record<string, string | number>;
}

export interface OrionOrbitalBubbleChartProps {
  title?: string;
  subtitle?: string;
  centerLabel?: string;
  centerValue?: string;
  orbits?: readonly { label: string; radius: number }[];
  bubbles: readonly OrbitalBubble[];
  selectedBubbleId?: string;
  onSelectBubble?: (bubble: OrbitalBubble) => void;
  height?: number;
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

export const OrionOrbitalBubbleChart: React.FC<OrionOrbitalBubbleChartProps> = ({
  title = "Pareto & HPO Orbital Bubble Distribution",
  subtitle = "Multi-objective trial configurations & policy frontiers",
  centerLabel = "Frontier",
  centerValue = "Optimal",
  orbits = [
    { label: "High Reward", radius: 70 },
    { label: "Nominal", radius: 120 },
    { label: "Exploration", radius: 165 },
  ],
  bubbles,
  selectedBubbleId,
  onSelectBubble,
  height = 380,
}) => {
  const gradId = useId().replace(/:/g, "");
  const [hoveredBubbleId, setHoveredBubbleId] = useState<string | null>(null);
  const [internalSelectedId, setInternalSelectedId] = useState<string | null>(
    bubbles[0]?.id ?? null
  );

  const activeId = selectedBubbleId ?? internalSelectedId;
  const cx = 230;
  const cy = 190;

  // Compute position for each bubble
  const positionedBubbles = useMemo(() => {
    return bubbles.map((b) => {
      const orbit = orbits[b.orbitIdx] ?? orbits[0]!;
      const rad = ((b.angleDeg - 90) * Math.PI) / 180;
      const x = cx + orbit.radius * Math.cos(rad);
      const y = cy + orbit.radius * Math.sin(rad);
      return { ...b, x, y };
    });
  }, [bubbles, orbits]);

  const activeBubble = useMemo(
    () => positionedBubbles.find((b) => b.id === activeId),
    [positionedBubbles, activeId]
  );

  const hoveredBubble = useMemo(
    () => positionedBubbles.find((b) => b.id === hoveredBubbleId),
    [positionedBubbles, hoveredBubbleId]
  );

  const guideBubble = hoveredBubble ?? activeBubble;

  return (
    <div
      className="rounded-xl border p-4 space-y-3"
      style={nocPanelStyle}
      data-testid="orion-orbital-bubble-chart"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-noc-textPrimary">
            {title}
          </h3>
          <p className="text-[11px] text-noc-textMuted">{subtitle}</p>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono text-noc-textMuted">
          {orbits.map((o, idx) => (
            <span key={o.label} className="inline-flex items-center gap-1">
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{
                  backgroundColor:
                    idx === 0
                      ? NOC_COLORS.cyan
                      : idx === 1
                      ? NOC_COLORS.indigoText
                      : NOC_COLORS.purpleText,
                }}
              />
              {o.label}
            </span>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 items-center">
        {/* SVG Orbital Canvas */}
        <div className="lg:col-span-3 flex justify-center overflow-hidden">
          <svg
            viewBox="0 0 460 380"
            className="w-full max-w-[460px] select-none"
            style={{ height }}
            role="img"
            aria-label="Orion orbital bubble visualization"
          >
            <defs>
              {/* Radial shaders for 3D sphere glow */}
              {Object.entries(TONE_COLOR).map(([tone, hex]) => (
                <radialGradient
                  key={tone}
                  id={`${gradId}-sphere-${tone}`}
                  cx="32%"
                  cy="32%"
                  r="68%"
                >
                  <stop offset="0%" stopColor="#ffffff" stopOpacity="0.85" />
                  <stop offset="25%" stopColor={hex} stopOpacity="0.9" />
                  <stop offset="80%" stopColor={hex} stopOpacity="0.45" />
                  <stop offset="100%" stopColor="#090e1a" stopOpacity="0.8" />
                </radialGradient>
              ))}

              <radialGradient id={`${gradId}-hub`} cx="40%" cy="40%" r="60%">
                <stop offset="0%" stopColor="#ffffff" stopOpacity="0.9" />
                <stop offset="40%" stopColor={NOC_COLORS.cyan} stopOpacity="0.8" />
                <stop offset="100%" stopColor="#090e1a" stopOpacity="0.4" />
              </radialGradient>

              <filter id={`${gradId}-bubble-glow`} x="-30%" y="-30%" width="160%" height="160%">
                <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor={NOC_COLORS.cyan} floodOpacity="0.5" />
              </filter>
            </defs>

            {/* Concentric Orbit Tracks with peripheral tick marks */}
            {orbits.map((orbit) => (
              <g key={orbit.label}>
                <circle
                  cx={cx}
                  cy={cy}
                  r={orbit.radius}
                  fill="none"
                  stroke="var(--noc-border)"
                  strokeWidth="1"
                  strokeDasharray="4 4"
                />
              </g>
            ))}

            {/* Perimeter Degree Marks */}
            {Array.from({ length: 24 }).map((_, i) => {
              const deg = i * 15;
              const rad = (deg * Math.PI) / 180;
              const maxR = orbits[orbits.length - 1]!.radius + 6;
              const x1 = cx + maxR * Math.cos(rad);
              const y1 = cy + maxR * Math.sin(rad);
              const x2 = cx + (maxR + (i % 6 === 0 ? 8 : 4)) * Math.cos(rad);
              const y2 = cy + (maxR + (i % 6 === 0 ? 8 : 4)) * Math.sin(rad);
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

            {/* Laser Guide Ray from Hub to active/hovered bubble */}
            {guideBubble && (
              <line
                x1={cx}
                y1={cy}
                x2={guideBubble.x}
                y2={guideBubble.y}
                stroke={TONE_COLOR[guideBubble.tone ?? "cyan"]}
                strokeWidth="1.5"
                strokeDasharray="3 3"
                strokeOpacity="0.8"
              />
            )}

            {/* Center Core Hub */}
            <circle
              cx={cx}
              cy={cy}
              r={24}
              fill={`url(#${gradId}-hub)`}
              stroke={NOC_COLORS.cyan}
              strokeWidth="2"
            />
            <text
              x={cx}
              y={cy - 2}
              textAnchor="middle"
              dominantBaseline="central"
              fill="var(--noc-text-primary)"
              fontSize="10"
              fontWeight="bold"
              className="select-none pointer-events-none"
            >
              {centerValue}
            </text>
            <text
              x={cx}
              y={cy + 10}
              textAnchor="middle"
              dominantBaseline="central"
              fill="var(--noc-text-muted)"
              fontSize="8"
              className="select-none pointer-events-none uppercase tracking-wider"
            >
              {centerLabel}
            </text>

            {/* Orbital Spherical Bubbles */}
            {positionedBubbles.map((bubble) => {
              const isSelected = activeId === bubble.id;
              const isHovered = hoveredBubbleId === bubble.id;
              const tone = bubble.tone ?? "cyan";
              const strokeColor = TONE_COLOR[tone];

              return (
                <g
                  key={bubble.id}
                  role="button"
                  tabIndex={0}
                  aria-label={`${bubble.label}: ${bubble.value} ${bubble.unit ?? ""}`}
                  aria-pressed={isSelected}
                  onClick={() => {
                    setInternalSelectedId(bubble.id);
                    onSelectBubble?.(bubble);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setInternalSelectedId(bubble.id);
                      onSelectBubble?.(bubble);
                    }
                  }}
                  onMouseEnter={() => setHoveredBubbleId(bubble.id)}
                  onMouseLeave={() => setHoveredBubbleId(null)}
                  className="cursor-pointer focus:outline-none"
                >
                  {/* Selection / Hover Aura Ring */}
                  {(isSelected || isHovered) && (
                    <circle
                      cx={bubble.x}
                      cy={bubble.y}
                      r={bubble.radius + 5}
                      fill="none"
                      stroke={strokeColor}
                      strokeWidth="1.5"
                      strokeDasharray="2 2"
                      filter={`url(#${gradId}-bubble-glow)`}
                    />
                  )}
                  {/* 3D Shaded Sphere */}
                  <circle
                    cx={bubble.x}
                    cy={bubble.y}
                    r={bubble.radius}
                    fill={`url(#${gradId}-sphere-${tone})`}
                    stroke={strokeColor}
                    strokeWidth={isSelected ? 2 : 1}
                  />
                  {/* Bubble Inner Value / Label */}
                  <text
                    x={bubble.x}
                    y={bubble.y}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="#ffffff"
                    fontSize={bubble.radius > 18 ? "10" : "8"}
                    fontWeight="bold"
                    fontFamily="monospace"
                    className="select-none pointer-events-none tabular-nums"
                  >
                    {bubble.value}
                  </text>
                  <text
                    x={bubble.x}
                    y={bubble.y + bubble.radius + 10}
                    textAnchor="middle"
                    fill="var(--noc-text-secondary)"
                    fontSize="9"
                    className="select-none pointer-events-none"
                  >
                    {bubble.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* HUD Detail Card */}
        <div className="rounded-lg border border-noc-border bg-noc-card/70 p-3 text-xs space-y-2">
          <div className="border-b border-noc-border pb-1.5 flex items-center justify-between">
            <span className="font-bold text-noc-textPrimary uppercase tracking-wider truncate">
              {activeBubble ? activeBubble.label : "Configuration HUD"}
            </span>
            {activeBubble && (
              <span className="text-[10px] px-1.5 py-0.5 rounded font-mono bg-noc-deep text-noc-cyan">
                Orbit {activeBubble.orbitIdx + 1}
              </span>
            )}
          </div>

          {activeBubble ? (
            <>
              <div className="flex justify-between items-baseline pt-1">
                <span className="text-noc-textMuted text-[11px]">Primary Value:</span>
                <span className="font-mono font-bold text-sm text-noc-cyan tabular-nums">
                  {activeBubble.value} {activeBubble.unit ?? ""}
                </span>
              </div>

              {activeBubble.metrics && Object.keys(activeBubble.metrics).length > 0 && (
                <div className="space-y-1 pt-1.5 border-t border-noc-border">
                  <div className="text-[10px] uppercase font-bold text-noc-textMuted">Metrics</div>
                  {Object.entries(activeBubble.metrics).map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center text-[11px] tabular-nums font-mono">
                      <span className="text-noc-textSecondary truncate">{k}:</span>
                      <span className="font-semibold text-noc-textPrimary ml-2">{v}</span>
                    </div>
                  ))}
                </div>
              )}

              {activeBubble.trialMeta && Object.keys(activeBubble.trialMeta).length > 0 && (
                <div className="space-y-1 pt-1.5 border-t border-noc-border">
                  <div className="text-[10px] uppercase font-bold text-noc-textMuted">Hyperparameters</div>
                  {Object.entries(activeBubble.trialMeta).map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center text-[11px] tabular-nums font-mono">
                      <span className="text-noc-textMuted truncate">{k}:</span>
                      <span className="text-noc-indigo ml-2">{v}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="text-[11px] text-noc-textMuted py-4 text-center">
              Click any orbital bubble to inspect trial parameters
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
