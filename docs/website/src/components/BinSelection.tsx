import { useMemo, useState } from "react";
import type { BinFills, Results } from "../data/types";
import { useJson } from "../hooks/useJson";

/**
 * V2 — Bin selection, animated. Re-simulates the Last-Minute rule on the real,
 * policy-independent daily waste increments recovered from the simulator's
 * fill-history spreadsheets, so the visitor can drag the threshold and watch
 * the selected set and the overflow count move. The re-simulation is a
 * documented model of the rule, not a re-run of the simulator.
 */

interface SimDay {
  levels: number[];
  collections: number;
  overflows: number;
}

interface Sim {
  days: SimDay[];
  totalCollections: number;
  totalOverflows: number;
}

function simulateLastMinute(increments: number[][], tau: number, capacity: number): Sim {
  const nBins = increments.length;
  const days = increments[0].length;
  const level = new Array<number>(nBins).fill(0);
  const out: SimDay[] = [];
  let totalCollections = 0;
  let totalOverflows = 0;

  for (let d = 0; d < days; d++) {
    const willCollect = level.map((l) => l >= tau);
    let overflows = 0;
    for (let b = 0; b < nBins; b++) {
      const next = level[b] + increments[b][d];
      if (next > capacity) {
        overflows += 1;
        level[b] = capacity;
      } else {
        level[b] = next;
      }
    }
    let collections = 0;
    for (let b = 0; b < nBins; b++) {
      if (willCollect[b]) {
        collections += 1;
        level[b] = 0;
      }
    }
    totalCollections += collections;
    totalOverflows += overflows;
    out.push({ levels: [...level], collections, overflows });
  }
  return { days: out, totalCollections, totalOverflows };
}

const PRESETS = [
  { label: "CF70", tau: 70 },
  { label: "CF90", tau: 90 },
] as const;

export default function BinSelection({ results }: { results: Results | null }) {
  const { data, status, error } = useJson<BinFills>("/data/bin_fills.json");
  const [scenarioIdx, setScenarioIdx] = useState(0);
  const [day, setDay] = useState(0);
  const [tau, setTau] = useState(70);

  const scenario = data?.scenarios[scenarioIdx] ?? null;

  const sim = useMemo(() => {
    if (!scenario) return null;
    return simulateLastMinute(scenario.increments, tau, scenario.capacity);
  }, [scenario, tau]);

  if (status === "loading") return <div className="viz-loading">Loading bin fills…</div>;
  if (status === "error" || !data)
    return <div className="viz-error">Could not load bin fills: {error ?? "unknown error"}</div>;

  const current = sim?.days[day];
  const bins = scenario ? scenario.fills.length : 0;
  const cols = scenario && scenario.N >= 300 ? 20 : 10;

  const variantRows = results?.strategies ?? [];

  return (
    <div className="viz-panel">
      <div className="viz-panel-head">
        <span>Bin selection · Last-Minute rule</span>
        <select
          className="viz-select"
          value={scenarioIdx}
          onChange={(e) => setScenarioIdx(Number(e.target.value))}
          style={{ width: "auto", minWidth: 180 }}
          aria-label="Choose a network"
        >
          {data.scenarios.map((s, i) => (
            <option key={i} value={i}>
              {s.city} · N={s.N}
            </option>
          ))}
        </select>
      </div>

      <div className="viz-bin-layout">
        <div>
          <div className="viz-route-controls">
            <label htmlFor="viz-day">Day</label>
            <input
              id="viz-day"
              className="viz-slider"
              type="range"
              min={0}
              max={(scenario?.days ?? 1) - 1}
              value={day}
              onChange={(e) => setDay(Number(e.target.value))}
              style={{ flex: 1 }}
            />
            <span className="viz-day-readout">Day {day + 1}</span>
          </div>

          <div
            className="viz-bin-grid"
            role="img"
            aria-label={`Fill state of ${bins} bins on day ${day + 1} under threshold ${tau}%`}
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(${cols}, 1fr)`,
              gap: 3,
              marginBottom: 12,
            }}
          >
            {Array.from({ length: bins }, (_, b) => {
              const level = current?.levels[b] ?? 0;
              const overflow = level >= 100;
              const selected = current ? level >= tau : false;
              return (
                <div
                  key={b}
                  aria-hidden="true"
                  style={{
                    aspectRatio: "1",
                    borderRadius: 2,
                    background: "var(--bg-surface-elevated)",
                    position: "relative",
                    boxSizing: "border-box",
                    border: overflow
                      ? "1px solid var(--brand-warn)"
                      : selected
                        ? "1px solid var(--brand-route)"
                        : "1px solid var(--border-subtle)",
                  }}
                >
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      background: overflow
                        ? "var(--brand-warn)"
                        : "var(--brand-eco)",
                      opacity: Math.max(0.12, level / 100),
                    }}
                  />
                </div>
              );
            })}
          </div>

          <div className="viz-legend">
            <span>
              <span className="viz-swatch" style={{ background: "var(--brand-eco)", opacity: 0.5 }} />
              fill level
            </span>
            <span>
              <span
                className="viz-swatch"
                style={{ background: "var(--bg-surface-elevated)", border: "1px solid var(--brand-route)" }}
              />
              selected today
            </span>
            <span>
              <span className="viz-swatch" style={{ background: "var(--brand-warn)" }} />
              at capacity
            </span>
          </div>
        </div>

        <div className="viz-threshold">
          <div>
            <div className="viz-kicker">Critical-fill threshold</div>
            <div className="viz-slider-row">
              <input
                className="viz-slider"
                type="range"
                min={40}
                max={95}
                value={tau}
                onChange={(e) => setTau(Number(e.target.value))}
                aria-label="Critical-fill threshold, percent"
              />
              <span className="viz-readout">{tau}%</span>
            </div>
          </div>

          <div className="viz-presets" role="group" aria-label="Benchmarked thresholds">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                className="viz-preset"
                aria-pressed={tau === p.tau}
                onClick={() => setTau(p.tau)}
              >
                {p.label}
              </button>
            ))}
          </div>

          <div className="viz-metric-strip" aria-live="polite">
            <span>
              Collections
              <strong>{sim ? sim.totalCollections.toLocaleString() : "—"}</strong>
            </span>
            <span>
              Overflows
              <strong style={{ color: "var(--brand-warn)" }}>
                {sim ? sim.totalOverflows.toLocaleString() : "—"}
              </strong>
            </span>
          </div>

          <p className="viz-note" style={{ fontSize: "0.82rem", margin: 0 }}>
            Collect later (higher threshold) and you haul fewer, fuller loads but overflow
            more. This is the paper's central finding, reproduced here on the real demand
            increments — not hand-copied numbers.
          </p>
        </div>
      </div>

      {variantRows.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <div className="viz-kicker">The trade-off in the real 30-day study</div>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>kg/km</th>
                <th>Overflows</th>
                <th>km</th>
              </tr>
            </thead>
            <tbody>
              {variantRows.map((v) => (
                <tr key={v.key}>
                  <td>{v.name}</td>
                  <td>{v.kgkm.toFixed(2)}</td>
                  <td>{v.overflows.toFixed(1)}</td>
                  <td>{v.km.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
