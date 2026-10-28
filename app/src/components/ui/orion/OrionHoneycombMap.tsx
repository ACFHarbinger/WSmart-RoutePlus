import React, { useId, useMemo, useState } from "react";
import { NOC_COLORS, type NocTone } from "../telemetry/tokens";
import { nocPanelStyle } from "../telemetry/panel";

export interface HoneycombZone {
  id: string;
  row: number;
  col: number;
  label: string;
  fillRatio: number; // 0..1.5+
  binCount: number;
  overflowRisk: boolean;
  status: "idle" | "ok" | "active" | "warn" | "crit";
  depot?: boolean;
  metrics?: Record<string, string | number>;
}

export interface ZoneTrajectory {
  id: string;
  fromZoneId: string;
  toZoneId: string;
  vehicleId?: string;
  loadKg?: number;
  tone?: NocTone;
  active?: boolean;
}

export interface OrionHoneycombMapProps {
  title?: string;
  subtitle?: string;
  zones: readonly HoneycombZone[];
  trajectories?: readonly ZoneTrajectory[];
  hexRadius?: number;
  onSelectZone?: (zone: HoneycombZone) => void;
  height?: number;
}

const STATUS_COLOR: Record<HoneycombZone["status"], string> = {
  idle: "#334155",
  ok: NOC_COLORS.ok,
  active: NOC_COLORS.cyan,
  warn: NOC_COLORS.warn,
  crit: NOC_COLORS.crit,
};

function hexFlatPoints(cx: number, cy: number, r: number): string {
  const pts: string[] = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 180) * (60 * i);
    const px = cx + r * Math.cos(angle);
    const py = cy + r * Math.sin(angle);
    pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
  }
  return pts.join(" ");
}

export const OrionHoneycombMap: React.FC<OrionHoneycombMapProps> = ({
  title = "Situational Honeycomb Matrix & Tour Trajectories",
  subtitle = "Regional bin fill density & dynamic vehicle tour corridors",
  zones,
  trajectories = [],
  hexRadius = 24,
  onSelectZone,
  height = 380,
}) => {
  const gradId = useId().replace(/:/g, "");
  const [filter, setFilter] = useState<"all" | "active" | "overflow" | "depots">("all");
  const [activeZoneId, setActiveZoneId] = useState<string | null>(zones[0]?.id ?? null);
  const [hoveredZoneId, setHoveredZoneId] = useState<string | null>(null);

  // Compute 2D layout for flat-topped hexagons
  const layout = useMemo(() => {
    const hexH = Math.sqrt(3) * hexRadius;
    const horizStep = 1.5 * hexRadius;
    const vertStep = hexH;
    const padX = hexRadius + 24;
    const padY = hexH / 2 + 20;

    let maxX = 0;
    let maxY = 0;

    const coords = new Map<string, { cx: number; cy: number; zone: HoneycombZone }>();

    zones.forEach((z) => {
      const cx = padX + z.col * horizStep;
      const cy = padY + z.row * vertStep + (z.col % 2 === 1 ? hexH / 2 : 0);
      coords.set(z.id, { cx, cy, zone: z });
      if (cx > maxX) maxX = cx;
      if (cy > maxY) maxY = cy;
    });

    return { coords, width: Math.max(480, maxX + padX), height: Math.max(340, maxY + padY) };
  }, [zones, hexRadius]);

  // Filtered zones
  const visibleZones = useMemo(() => {
    return zones.filter((z) => {
      if (filter === "overflow") return z.overflowRisk || z.fillRatio >= 1.0;
      if (filter === "active") return z.status === "active" || z.status === "ok";
      if (filter === "depots") return !!z.depot;
      return true;
    });
  }, [zones, filter]);

  const activeZone = useMemo(
    () => zones.find((z) => z.id === activeZoneId),
    [zones, activeZoneId]
  );

  return (
    <div
      className="rounded-xl border p-4 space-y-3"
      style={nocPanelStyle}
      data-testid="orion-honeycomb-map"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-noc-textPrimary">
            {title}
          </h3>
          <p className="text-[11px] text-noc-textMuted">{subtitle}</p>
        </div>

        {/* Severity Filter Pills */}
        <div className="flex items-center gap-1 bg-noc-deep p-1 rounded-lg border border-noc-border text-xs">
          {(["all", "active", "overflow", "depots"] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => setFilter(mode)}
              className={`px-2 py-0.5 rounded text-[11px] font-medium uppercase transition-colors ${
                filter === mode
                  ? "bg-noc-card text-noc-cyan border border-noc-borderStrong"
                  : "text-noc-textMuted hover:text-noc-textPrimary"
              }`}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 items-center">
        {/* SVG Honeycomb + Bezier Corridor Map */}
        <div className="lg:col-span-3 flex justify-center overflow-auto">
          <svg
            viewBox={`0 0 ${layout.width} ${layout.height}`}
            className="w-full select-none"
            style={{ height }}
            role="img"
            aria-label="Honeycomb situational matrix"
          >
            <defs>
              <linearGradient id={`${gradId}-arc-glow`} x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor={NOC_COLORS.cyan} stopOpacity="0.8" />
                <stop offset="50%" stopColor={NOC_COLORS.indigoText} stopOpacity="0.7" />
                <stop offset="100%" stopColor={NOC_COLORS.purpleText} stopOpacity="0.6" />
              </linearGradient>
            </defs>

            {/* Honeycomb Cells */}
            {visibleZones.map((zone) => {
              const pos = layout.coords.get(zone.id);
              if (!pos) return null;

              const isSelected = activeZoneId === zone.id;
              const isHovered = hoveredZoneId === zone.id;
              const color = STATUS_COLOR[zone.status];
              const fillIsGap = zone.metrics?.fill === "gap";
              const fillOpacity = zone.status === "idle" ? 0.35 : 0.7;
              const fillLabel = fillIsGap
                ? "fill not measured"
                : `${(zone.fillRatio * 100).toFixed(0)}% fill`;

              return (
                <g
                  key={zone.id}
                  role="button"
                  tabIndex={0}
                  aria-label={`${zone.label}: ${fillLabel}`}
                  aria-pressed={isSelected}
                  onClick={() => {
                    setActiveZoneId(zone.id);
                    onSelectZone?.(zone);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setActiveZoneId(zone.id);
                      onSelectZone?.(zone);
                    }
                  }}
                  onMouseEnter={() => setHoveredZoneId(zone.id)}
                  onMouseLeave={() => setHoveredZoneId(null)}
                  className="cursor-pointer focus:outline-none"
                >
                  <polygon
                    points={hexFlatPoints(pos.cx, pos.cy, hexRadius - 1.5)}
                    fill={color}
                    fillOpacity={fillOpacity}
                    stroke={isSelected || isHovered ? "#ffffff" : color}
                    strokeWidth={isSelected ? 2 : 1}
                    className="transition-all duration-200"
                  />
                  {/* Depot indicator ring */}
                  {zone.depot && (
                    <circle
                      cx={pos.cx}
                      cy={pos.cy}
                      r={hexRadius * 0.45}
                      fill="none"
                      stroke="#ffffff"
                      strokeWidth="1.5"
                      strokeDasharray="2 2"
                    />
                  )}
                  {/* Label */}
                  <text
                    x={pos.cx}
                    y={pos.cy - 3}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="#ffffff"
                    fontSize="9"
                    fontWeight="bold"
                    className="select-none pointer-events-none"
                  >
                    {zone.label}
                  </text>
                  <text
                    x={pos.cx}
                    y={pos.cy + 7}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="var(--noc-text-secondary)"
                    fontSize="8"
                    fontFamily="monospace"
                    className="select-none pointer-events-none tabular-nums"
                  >
                    {(zone.fillRatio * 100).toFixed(0)}%
                  </text>
                </g>
              );
            })}

            {/* Bezier Trajectory Arcs (Tour flows) */}
            {trajectories.map((traj) => {
              const from = layout.coords.get(traj.fromZoneId);
              const to = layout.coords.get(traj.toZoneId);
              if (!from || !to) return null;

              // Quadratic bezier control point offset perpendicular to line
              const mx = (from.cx + to.cx) / 2;
              const my = (from.cy + to.cy) / 2;
              const dx = to.cx - from.cx;
              const dy = to.cy - from.cy;
              const len = Math.hypot(dx, dy) || 1;
              const offset = Math.min(35, len * 0.25);
              const qx = mx - (dy / len) * offset;
              const qy = my + (dx / len) * offset;

              const stroke = traj.tone ? STATUS_COLOR[traj.tone === "crit" ? "crit" : "ok"] : `url(#${gradId}-arc-glow)`;

              return (
                <g key={traj.id}>
                  <path
                    d={`M ${from.cx.toFixed(1)},${from.cy.toFixed(1)} Q ${qx.toFixed(1)},${qy.toFixed(1)} ${to.cx.toFixed(1)},${to.cy.toFixed(1)}`}
                    fill="none"
                    stroke={stroke}
                    strokeWidth={traj.active ? 2.5 : 1.5}
                    strokeDasharray={traj.active ? "4 3" : undefined}
                    strokeOpacity={traj.active ? 0.9 : 0.6}
                  />
                  {/* Trajectory Direction Head Marker */}
                  <circle
                    cx={to.cx}
                    cy={to.cy}
                    r={3}
                    fill={stroke}
                  />
                </g>
              );
            })}
          </svg>
        </div>

        {/* HUD Situational Inspector Card */}
        <div className="rounded-lg border border-noc-border bg-noc-card/70 p-3 text-xs space-y-2">
          <div className="border-b border-noc-border pb-1.5 flex items-center justify-between">
            <span className="font-bold text-noc-textPrimary uppercase tracking-wider truncate">
              {activeZone ? activeZone.label : "Zone Telemetry"}
            </span>
            {activeZone && (
              <span
                style={{
                  backgroundColor: `${STATUS_COLOR[activeZone.status]}22`,
                  color: STATUS_COLOR[activeZone.status],
                  borderColor: `${STATUS_COLOR[activeZone.status]}55`,
                }}
                className="text-[10px] px-1.5 py-0.5 rounded border font-mono uppercase"
              >
                {activeZone.status}
              </span>
            )}
          </div>

          {activeZone ? (
            <>
              <div className="flex justify-between items-baseline pt-1">
                <span className="text-noc-textMuted text-[11px]">Fill Level:</span>
                <span
                  style={{
                    color:
                      activeZone.fillRatio >= 1.0
                        ? NOC_COLORS.crit
                        : activeZone.fillRatio >= 0.8
                        ? NOC_COLORS.warn
                        : NOC_COLORS.ok,
                  }}
                  className="font-mono font-bold text-sm tabular-nums"
                >
                  {(activeZone.fillRatio * 100).toFixed(1)}%
                </span>
              </div>

              <div className="flex justify-between items-center text-[11px] tabular-nums font-mono text-noc-textSecondary">
                <span>Total Bins:</span>
                <span className="font-semibold text-noc-textPrimary">{activeZone.binCount}</span>
              </div>

              <div className="flex justify-between items-center text-[11px] tabular-nums font-mono text-noc-textSecondary">
                <span>Overflow Risk:</span>
                <span className={activeZone.overflowRisk ? "text-noc-crit font-bold" : "text-noc-ok"}>
                  {activeZone.overflowRisk ? "CRITICAL RISK" : "NOMINAL"}
                </span>
              </div>

              {activeZone.depot && (
                <div className="text-[10px] uppercase font-bold text-noc-cyan pt-1">
                  Primary Fleet Depot Hub
                </div>
              )}

              {activeZone.metrics && Object.keys(activeZone.metrics).length > 0 && (
                <div className="space-y-1 pt-1.5 border-t border-noc-border">
                  <div className="text-[10px] uppercase font-bold text-noc-textMuted">Metrics</div>
                  {Object.entries(activeZone.metrics).map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center text-[11px] tabular-nums font-mono">
                      <span className="text-noc-textSecondary truncate">{k}:</span>
                      <span className="font-semibold text-noc-textPrimary ml-2">{v}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="text-[11px] text-noc-textMuted py-4 text-center">
              Click a honeycomb zone to view telemetry
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
