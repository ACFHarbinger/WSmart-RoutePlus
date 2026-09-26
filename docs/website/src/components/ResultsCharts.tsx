import type { Results } from "../data/types";

/**
 * V4 — results charts with a point of view: the Pareto front, the paired
 * improver comparison, and the constructor × scenario heatmap, each with the
 * paper's caveats built in rather than buried in a footnote.
 */

const W = 460;
const H = 300;
const PAD = { l: 48, r: 16, t: 16, b: 40 };

function ParetoChart({ results }: { results: Results }) {
  const points = results.pareto;
  const colors = results.colors.constructors;
  const xs = points.map((p) => p.overflows);
  const ys = points.map((p) => p.kgkm);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const xPad = (xMax - xMin) * 0.08 || 1;
  const yPad = (yMax - yMin) * 0.08 || 1;

  const sx = (v: number) => PAD.l + ((v - (xMin - xPad)) / (xMax - xMin + 2 * xPad)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - (yMin - yPad)) / (yMax - yMin + 2 * yPad)) * (H - PAD.t - PAD.b);

  const front = points.filter((p) => p.front).sort((a, b) => a.overflows - b.overflows);
  const frontPath = front
    .map((p, i) => {
      const next = front[i + 1];
      const x = sx(p.overflows);
      const y = sy(p.kgkm);
      return `${i === 0 ? "M" : "L"} ${x} ${y}${next ? ` L ${sx(next.overflows)} ${y}` : ""}`;
    })
    .join(" ");

  const yTicks = 4;
  const xTicks = 4;

  return (
    <svg className="viz-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Pareto front of efficiency against overflow events">
      {Array.from({ length: xTicks + 1 }, (_, i) => {
        const v = xMin - xPad + ((xMax - xMin + 2 * xPad) * i) / xTicks;
        const x = sx(v);
        return (
          <g key={i}>
            <line x1={x} y1={H - PAD.b} x2={x} y2={PAD.t} className="viz-axis-line" strokeDasharray="2 3" opacity={0.4} />
            <text x={x} y={H - PAD.b + 16} textAnchor="middle" className="viz-axis">
              {Math.round(v)}
            </text>
          </g>
        );
      })}
      {Array.from({ length: yTicks + 1 }, (_, i) => {
        const v = yMin - yPad + ((yMax - yMin + 2 * yPad) * i) / yTicks;
        const y = sy(v);
        return (
          <g key={i}>
            <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} className="viz-axis-line" strokeDasharray="2 3" opacity={0.4} />
            <text x={PAD.l - 6} y={y + 3} textAnchor="end" className="viz-axis">
              {v.toFixed(1)}
            </text>
          </g>
        );
      })}

      {frontPath && <path d={frontPath} fill="none" stroke="var(--text-muted)" strokeWidth={1.5} strokeDasharray="4 3" />}

      {points.map((p) => (
        <g key={p.name}>
          <circle
            cx={sx(p.overflows)}
            cy={sy(p.kgkm)}
            r={5}
            fill={colors[p.name] ?? "var(--brand-route)"}
          />
          <text x={sx(p.overflows) + 8} y={sy(p.kgkm) + 3} className="viz-axis" style={{ fill: "var(--text-main)" }}>
            {p.name}
          </text>
        </g>
      ))}

      <text x={(PAD.l + W - PAD.r) / 2} y={H - 4} textAnchor="middle" className="viz-axis">
        Overflow events (mean)
      </text>
      <text x={14} y={(PAD.t + H - PAD.b) / 2} textAnchor="middle" className="viz-axis" transform={`rotate(-90 14 ${(PAD.t + H - PAD.b) / 2})`}>
        Efficiency (kg/km)
      </text>
    </svg>
  );
}

function ImproverChart({ results }: { results: Results }) {
  const imp = results.improvers;
  const maxLoss = Math.min(0, ...imp.loss_breakdown.map((l) => l.mean)) * 1.2;
  const chartH = 140;
  const barW = 26;

  return (
    <div>
      <div className="viz-metric-strip">
        <span>
          CLS wins
          <strong style={{ color: "var(--brand-eco)" }}>
            {imp.wins} / {imp.pairs}
          </strong>
        </span>
        <span>
          Mean Δ kg/km
          <strong>+{imp.mean_delta_kgkm.toFixed(2)}</strong>
        </span>
        <span>
          Mean route length
          <strong>
            {imp.mean_km_cls.toLocaleString()} vs {imp.mean_km_ftsp.toLocaleString()} km
          </strong>
        </span>
      </div>
      <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
        The {imp.losses} losses are deeper than the wins and confined to three
        constructors; 21 of 22 occur at N=350:
      </p>
      <svg className="viz-svg" viewBox={`0 0 ${W} ${chartH + 30}`} role="img" aria-label="CLS minus Fast-TSP efficiency, by losing constructor">
        <line x1={0} y1={chartH - 10} x2={W} y2={chartH - 10} className="viz-axis-line" />
        {imp.loss_breakdown.map((l, i) => {
          const h = ((0 - l.mean) / (0 - maxLoss)) * (chartH - 30);
          const x = 20 + i * 130;
          const y = chartH - 10 - h;
          return (
            <g key={l.constructor}>
              <rect x={x} y={y} width={barW} height={h} fill="var(--brand-warn)" rx={2} />
              <text x={x + barW / 2} y={chartH + 4} textAnchor="middle" className="viz-axis">
                {l.constructor}
              </text>
              <text x={x + barW / 2} y={y - 4} textAnchor="middle" className="viz-axis">
                {l.mean.toFixed(2)}
              </text>
            </g>
          );
        })}
        <text x={0} y={8} className="viz-axis">
          Δ kg/km (CLS − Fast-TSP), mean of {imp.losses} losing pairs
        </text>
      </svg>
    </div>
  );
}

function Heatmap({ results }: { results: Results }) {
  const { columns, rows } = results.heatmap;
  if (!columns.length) return null;
  const values = rows.flatMap((r) => r.values);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const cellW = (W - 130) / columns.length;

  return (
    <div>
      <svg className="viz-svg" viewBox={`0 0 ${W} ${rows.length * 26 + 30}`} role="img" aria-label="Constructor by scenario efficiency heatmap">
        {columns.map((c, ci) => (
          <text key={c} x={130 + ci * cellW + cellW / 2} y={12} textAnchor="middle" className="viz-axis" style={{ fontSize: 8 }}>
            {c}
          </text>
        ))}
        {rows.map((r, ri) => {
          const y = 24 + ri * 26;
          return (
            <g key={r.constructor}>
              <text x={124} y={y + 14} textAnchor="end" className="viz-axis" style={{ fill: "var(--text-main)" }}>
                {r.constructor}
              </text>
              {r.values.map((v, ci) => {
                const t = (v - lo) / (hi - lo || 1);
                return (
                  <g key={ci}>
                    <rect
                      x={130 + ci * cellW}
                      y={y}
                      width={cellW - 2}
                      height={22}
                      rx={2}
                      fill="var(--brand-eco)"
                      opacity={0.15 + t * 0.85}
                    />
                    <text x={130 + ci * cellW + (cellW - 2) / 2} y={y + 15} textAnchor="middle" className="viz-axis" style={{ fill: t > 0.6 ? "var(--text-inverse)" : "var(--text-main)" }}>
                      {v.toFixed(1)}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      <p className="viz-note">
        Gamma-3 and Empirical columns are shown separately — pooling them would be
        misleading, as the two processes produce very different load regimes. Higher
        efficiency on Gamma-3 is expected: there is simply more waste per kilometre.
      </p>
    </div>
  );
}

export default function ResultsCharts({ results }: { results: Results }) {
  return (
    <div className="viz-chart-grid">
      <div className="viz-panel">
        <div className="viz-panel-head">
          <span>Efficiency / service Pareto front</span>
          <span>30-day horizon</span>
        </div>
        <ParetoChart results={results} />
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", margin: 0 }}>
          Read with the medians: the horizontal spread is mostly tail behaviour — seven of
          eight constructors have a median overflow count of exactly 4.
        </p>
      </div>

      <div className="viz-panel">
        <div className="viz-panel-head">
          <span>CLS vs Fast-TSP, matched demand</span>
          <span>{results.improvers.pairs} pairs</span>
        </div>
        <ImproverChart results={results} />
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", margin: 0 }}>
          These are matched configurations, not a controlled improver treatment:
          upstream collected-bin counts also differ in some pairs. The deltas are
          descriptive, not causal.
        </p>
      </div>

      <div className="viz-panel" style={{ gridColumn: "1 / -1" }}>
        <div className="viz-panel-head">
          <span>Constructor × scenario efficiency</span>
          <span>kg/km, mean over matched scenarios</span>
        </div>
        <Heatmap results={results} />
      </div>
    </div>
  );
}
