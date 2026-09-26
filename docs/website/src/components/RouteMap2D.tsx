import { useMemo, useState } from "react";
import type { RouteScenario } from "../data/types";

/**
 * V3 (2-D projection) — the multi-period route made legible: one day's tour
 * over the network, scrub through 30 days, see which bins are revisited and
 * which are skipped. Rendered as plain SVG so it works everywhere and needs
 * no WebGL; the three.js stack view is an optional, lazy-loaded upgrade.
 */
export default function RouteMap2D({ scenario }: { scenario: RouteScenario }) {
  const [day, setDay] = useState(0);
  const current = scenario.days[day];

  const tourPath = useMemo(() => {
    if (!current) return "";
    return current.tour.map((i) => `${scenario.nodes[i].x * 100},${scenario.nodes[i].y * 100}`).join(" ");
  }, [current, scenario]);

  const mandatory = useMemo(() => new Set(current?.mandatory ?? []), [current]);

  const depot = scenario.nodes.find((n) => n.depot);

  return (
    <div>
      <div className="viz-route-controls">
        <label htmlFor="viz-route-day">Day</label>
        <input
          id="viz-route-day"
          className="viz-slider"
          type="range"
          min={0}
          max={scenario.days.length - 1}
          value={day}
          onChange={(e) => setDay(Number(e.target.value))}
          style={{ flex: 1 }}
        />
        <span className="viz-day-readout">Day {day + 1} / {scenario.days.length}</span>
      </div>

      <svg
        className="viz-map"
        viewBox="0 0 100 100"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Route for ${scenario.city} on day ${day + 1}: ${current?.ncol ?? 0} bins collected, ${current?.km ?? 0} km`}
      >
        {/* faint graticule */}
        {[20, 40, 60, 80].map((g) => (
          <g key={g} stroke="var(--border-subtle)" strokeWidth="0.1">
            <line x1={g} y1={0} x2={g} y2={100} />
            <line x1={0} y1={g} x2={100} y2={g} />
          </g>
        ))}

        {/* all bins */}
        {scenario.nodes.map((n) => {
          if (n.depot) return null;
          return (
            <circle
              key={n.id}
              cx={n.x * 100}
              cy={n.y * 100}
              r={scenario.N >= 300 ? 0.35 : 0.6}
              fill={mandatory.has(scenario.nodes.indexOf(n)) ? "var(--brand-warn)" : "var(--map-line)"}
              opacity={mandatory.has(scenario.nodes.indexOf(n)) ? 0.95 : 0.45}
            />
          );
        })}

        {/* tour */}
        <polyline
          points={tourPath}
          fill="none"
          stroke="var(--brand-route)"
          strokeWidth={scenario.N >= 300 ? 0.28 : 0.45}
          strokeLinejoin="round"
          strokeLinecap="round"
          opacity={0.85}
        />

        {/* depot */}
        {depot && (
          <g>
            <circle
              cx={depot.x * 100}
              cy={depot.y * 100}
              r={2.2}
              fill="var(--brand-eco)"
              stroke="var(--bg-surface)"
              strokeWidth={0.4}
            />
            <text
              x={depot.x * 100}
              y={depot.y * 100 - 3}
              textAnchor="middle"
              fontSize={2.6}
              fill="var(--text-main)"
              fontFamily="var(--font-mono)"
            >
              D
            </text>
          </g>
        )}
      </svg>

      <div className="viz-metric-strip">
        <span>
          Bins collected
          <strong>{current?.ncol ?? 0}</strong>
        </span>
        <span>
          Distance
          <strong>{current?.km.toLocaleString()} km</strong>
        </span>
        <span>
          Efficiency
          <strong>{current?.kgkm.toFixed(2)} kg/km</strong>
        </span>
        <span>
          Overflows
          <strong>{current?.overflows}</strong>
        </span>
      </div>

      <div className="viz-legend">
        <span>
          <span className="viz-swatch" style={{ background: "var(--brand-eco)" }} /> depot
        </span>
        <span>
          <span className="viz-swatch" style={{ background: "var(--brand-warn)" }} /> mandatory bin
        </span>
        <span>
          <span className="viz-swatch" style={{ background: "var(--map-line)", opacity: 0.5 }} /> optional / skipped
        </span>
        <span>
          <span className="viz-swatch" style={{ background: "var(--brand-route)" }} /> route
        </span>
      </div>
    </div>
  );
}
