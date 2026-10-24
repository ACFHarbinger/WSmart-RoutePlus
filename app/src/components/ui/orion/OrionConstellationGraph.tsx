import React, { useId, useMemo, useState } from "react";
import { NOC_COLORS, type NocTone } from "../telemetry/tokens";
import { nocPanelStyle } from "../telemetry/panel";

export interface ConstellationNode {
  id: string;
  label: string;
  tier: 0 | 1 | 2; // 0 = Core Hub, 1 = Worker/Subnet, 2 = Action/Leaf
  group?: string;
  status?: "nominal" | "active" | "warn" | "crit";
  metrics?: Record<string, string | number>;
  details?: string;
  tone?: NocTone;
}

export interface ConstellationLink {
  source: string;
  target: string;
  label?: string;
  weight?: number;
  active?: boolean;
  tone?: NocTone;
}

export interface OrionConstellationGraphProps {
  title?: string;
  subtitle?: string;
  nodes: readonly ConstellationNode[];
  links: readonly ConstellationLink[];
  selectedNodeId?: string;
  onSelectNode?: (node: ConstellationNode) => void;
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

export const OrionConstellationGraph: React.FC<OrionConstellationGraphProps> = ({
  title = "Orion Constellation Network Topology",
  subtitle = "Hub-and-spoke decision tree & attention connectivity",
  nodes,
  links,
  selectedNodeId,
  onSelectNode,
  height = 380,
}) => {
  const gradId = useId().replace(/:/g, "");
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [internalSelectedId, setInternalSelectedId] = useState<string | null>(
    nodes[0]?.id ?? null
  );

  const activeId = selectedNodeId ?? internalSelectedId;

  // Compute node positions: Tier 0 at center, Tier 1 at orbit R1, Tier 2 at orbit R2
  const layout = useMemo(() => {
    const cx = 240;
    const cy = 190;
    const r1 = 85;
    const r2 = 150;

    const tier0 = nodes.filter((n) => n.tier === 0);
    const tier1 = nodes.filter((n) => n.tier === 1);
    const tier2 = nodes.filter((n) => n.tier === 2);

    const pos = new Map<string, { x: number; y: number; r: number }>();

    // Hub node(s)
    tier0.forEach((n, i) => {
      if (tier0.length === 1) {
        pos.set(n.id, { x: cx, y: cy, r: 24 });
      } else {
        const angle = (2 * Math.PI * i) / tier0.length;
        pos.set(n.id, { x: cx + 25 * Math.cos(angle), y: cy + 25 * Math.sin(angle), r: 18 });
      }
    });

    // Tier 1 orbit
    tier1.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / Math.max(1, tier1.length) - Math.PI / 2;
      pos.set(n.id, {
        x: cx + r1 * Math.cos(angle),
        y: cy + r1 * Math.sin(angle),
        r: 14,
      });
    });

    // Tier 2 orbit
    tier2.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / Math.max(1, tier2.length) - Math.PI / 2;
      pos.set(n.id, {
        x: cx + r2 * Math.cos(angle),
        y: cy + r2 * Math.sin(angle),
        r: 10,
      });
    });

    return { cx, cy, r1, r2, pos };
  }, [nodes]);

  // Determine connected neighbor IDs for hovered node
  const highlightedNeighbors = useMemo(() => {
    if (!hoveredNodeId) return null;
    const neighbors = new Set<string>([hoveredNodeId]);
    links.forEach((l) => {
      if (l.source === hoveredNodeId) neighbors.add(l.target);
      if (l.target === hoveredNodeId) neighbors.add(l.source);
    });
    return neighbors;
  }, [hoveredNodeId, links]);

  const activeNode = useMemo(
    () => nodes.find((n) => n.id === activeId),
    [nodes, activeId]
  );

  return (
    <div
      className="rounded-xl border p-4 space-y-3"
      style={nocPanelStyle}
      data-testid="orion-constellation-graph"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-noc-textPrimary">
            {title}
          </h3>
          <p className="text-[11px] text-noc-textMuted">{subtitle}</p>
        </div>
        <div className="flex items-center gap-2 text-[11px] font-mono text-noc-textMuted">
          <span className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-noc-cyan" /> Tier 0 Hub
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-noc-indigo" /> Tier 1 Worker
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-noc-purple" /> Tier 2 Leaf
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 items-center">
        {/* SVG Network Visualizer */}
        <div className="lg:col-span-3 flex justify-center overflow-hidden">
          <svg
            viewBox="0 0 480 380"
            className="w-full max-w-[480px] select-none"
            style={{ height }}
            role="img"
            aria-label="Orion constellation topology"
          >
            <defs>
              <radialGradient id={`${gradId}-hub`} cx="40%" cy="40%" r="60%">
                <stop offset="0%" stopColor="#ffffff" stopOpacity="0.9" />
                <stop offset="35%" stopColor={NOC_COLORS.cyan} stopOpacity="0.8" />
                <stop offset="100%" stopColor="#0369a1" stopOpacity="0.3" />
              </radialGradient>
              <radialGradient id={`${gradId}-tier1`} cx="40%" cy="40%" r="60%">
                <stop offset="0%" stopColor="#ffffff" stopOpacity="0.8" />
                <stop offset="40%" stopColor={NOC_COLORS.indigoText} stopOpacity="0.7" />
                <stop offset="100%" stopColor="#312e81" stopOpacity="0.3" />
              </radialGradient>
              <radialGradient id={`${gradId}-tier2`} cx="40%" cy="40%" r="60%">
                <stop offset="0%" stopColor="#ffffff" stopOpacity="0.8" />
                <stop offset="50%" stopColor={NOC_COLORS.purpleText} stopOpacity="0.6" />
                <stop offset="100%" stopColor="#581c87" stopOpacity="0.3" />
              </radialGradient>
              <filter id={`${gradId}-glow`} x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor={NOC_COLORS.cyan} floodOpacity="0.6" />
              </filter>
            </defs>

            {/* Concentric Orbit Tracks & Cardinal Ticks */}
            <circle
              cx={layout.cx}
              cy={layout.cy}
              r={layout.r1}
              fill="none"
              stroke="var(--noc-border)"
              strokeDasharray="3 3"
              strokeWidth="1"
            />
            <circle
              cx={layout.cx}
              cy={layout.cy}
              r={layout.r2}
              fill="none"
              stroke="var(--noc-border)"
              strokeDasharray="4 4"
              strokeWidth="1"
            />
            {/* Degree ticks */}
            {[0, 45, 90, 135, 180, 225, 270, 315].map((deg) => {
              const rad = (deg * Math.PI) / 180;
              const x1 = layout.cx + (layout.r2 + 4) * Math.cos(rad);
              const y1 = layout.cy + (layout.r2 + 4) * Math.sin(rad);
              const x2 = layout.cx + (layout.r2 + 10) * Math.cos(rad);
              const y2 = layout.cy + (layout.r2 + 10) * Math.sin(rad);
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

            {/* Cyber Link Edges */}
            {links.map((link, idx) => {
              const src = layout.pos.get(link.source);
              const tgt = layout.pos.get(link.target);
              if (!src || !tgt) return null;

              const isConnected =
                !highlightedNeighbors ||
                (highlightedNeighbors.has(link.source) &&
                  highlightedNeighbors.has(link.target));
              const strokeColor = link.tone ? TONE_COLOR[link.tone] : NOC_COLORS.cyan;
              const opacity = isConnected ? (link.active ? 0.9 : 0.6) : 0.12;

              return (
                <g key={`${link.source}-${link.target}-${idx}`}>
                  <line
                    x1={src.x}
                    y1={src.y}
                    x2={tgt.x}
                    y2={tgt.y}
                    stroke={strokeColor}
                    strokeWidth={link.active ? 2 : 1}
                    strokeDasharray={link.active ? "4 3" : undefined}
                    strokeOpacity={opacity}
                    className="transition-opacity duration-200"
                  />
                  {link.label && (
                    <text
                      x={(src.x + tgt.x) / 2}
                      y={(src.y + tgt.y) / 2 - 4}
                      fill="var(--noc-text-muted)"
                      fontSize="9"
                      fontFamily="monospace"
                      textAnchor="middle"
                      opacity={opacity}
                    >
                      {link.label}
                    </text>
                  )}
                </g>
              );
            })}

            {/* Constellation Nodes */}
            {nodes.map((node) => {
              const p = layout.pos.get(node.id);
              if (!p) return null;

              const isSelected = activeId === node.id;
              const isHovered = hoveredNodeId === node.id;
              const isDimmed = highlightedNeighbors && !highlightedNeighbors.has(node.id);
              const fillGrad =
                node.tier === 0
                  ? `url(#${gradId}-hub)`
                  : node.tier === 1
                  ? `url(#${gradId}-tier1)`
                  : `url(#${gradId}-tier2)`;
              const strokeColor = node.tone
                ? TONE_COLOR[node.tone]
                : node.tier === 0
                ? NOC_COLORS.cyan
                : node.tier === 1
                ? NOC_COLORS.indigoText
                : NOC_COLORS.purpleText;

              return (
                <g
                  key={node.id}
                  role="button"
                  tabIndex={0}
                  aria-label={`${node.label} (${node.tier === 0 ? "Hub" : "Worker"})`}
                  aria-pressed={isSelected}
                  onClick={() => {
                    setInternalSelectedId(node.id);
                    onSelectNode?.(node);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setInternalSelectedId(node.id);
                      onSelectNode?.(node);
                    }
                  }}
                  onMouseEnter={() => setHoveredNodeId(node.id)}
                  onMouseLeave={() => setHoveredNodeId(null)}
                  className="cursor-pointer focus:outline-none"
                  opacity={isDimmed ? 0.25 : 1}
                  style={{ transition: "opacity 0.2s, transform 0.2s" }}
                >
                  {/* Outer selection/hover ring */}
                  {(isSelected || isHovered) && (
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={p.r + 5}
                      fill="none"
                      stroke={strokeColor}
                      strokeWidth="1.5"
                      strokeDasharray="2 2"
                      filter={isSelected ? `url(#${gradId}-glow)` : undefined}
                    />
                  )}
                  {/* Node core sphere */}
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={p.r}
                    fill={fillGrad}
                    stroke={strokeColor}
                    strokeWidth={isSelected ? 2 : 1}
                  />
                  {/* Label */}
                  <text
                    x={p.x}
                    y={p.y + p.r + 12}
                    textAnchor="middle"
                    fill={isSelected ? "var(--noc-text-primary)" : "var(--noc-text-secondary)"}
                    fontSize={node.tier === 0 ? "11" : "10"}
                    fontWeight={isSelected || node.tier === 0 ? "bold" : "normal"}
                    className="select-none pointer-events-none"
                  >
                    {node.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Floating HUD Inspector Drawer */}
        <div className="rounded-lg border border-noc-border bg-noc-card/70 p-3 text-xs space-y-2">
          <div className="border-b border-noc-border pb-1.5 flex items-center justify-between">
            <span className="font-bold text-noc-textPrimary uppercase tracking-wider">
              {activeNode ? activeNode.label : "Node Telemetry"}
            </span>
            {activeNode && (
              <span className="text-[10px] px-1.5 py-0.5 rounded font-mono bg-noc-deep text-noc-cyan">
                Tier {activeNode.tier}
              </span>
            )}
          </div>

          {activeNode ? (
            <>
              {activeNode.details && (
                <p className="text-[11px] text-noc-textSecondary">{activeNode.details}</p>
              )}
              {activeNode.group && (
                <div className="text-[11px] text-noc-textMuted">
                  Group: <span className="font-mono text-noc-textPrimary">{activeNode.group}</span>
                </div>
              )}
              {activeNode.metrics && Object.keys(activeNode.metrics).length > 0 && (
                <div className="space-y-1 pt-1">
                  <div className="text-[10px] uppercase font-bold text-noc-textMuted">Metrics</div>
                  <div className="space-y-1">
                    {Object.entries(activeNode.metrics).map(([k, v]) => (
                      <div key={k} className="flex justify-between items-center text-[11px] tabular-nums font-mono">
                        <span className="text-noc-textSecondary">{k}:</span>
                        <span className="font-semibold text-noc-cyan">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-[11px] text-noc-textMuted py-4 text-center">
              Select or hover a node to inspect its policy state
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
