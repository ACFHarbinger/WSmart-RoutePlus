import { useState } from "react";
import { ArrowLeft, CircleAlert } from "lucide-react";
import { Link } from "react-router-dom";
import BinSelection from "../components/BinSelection";
import PolicyPipeline from "../components/PolicyPipeline";
import ResultsCharts from "../components/ResultsCharts";
import RouteViewer from "../components/RouteViewer";
import "../components/viz.css";
import type { Results } from "../data/types";
import { useJson } from "../hooks/useJson";

export default function Simulation() {
  const { data, status, error } = useJson<Results>("/data/results.json");
  const [showExcluded, setShowExcluded] = useState(false);

  return (
    <div className="interior">
      <Link className="back-link" to="/">
        <ArrowLeft size={15} /> Home
      </Link>
      <header className="page-hero">
        <span className="eyebrow">
          <span className="eyebrow-dot" />
          Simulation · the framework running
        </span>
        <h1>
          The policy space,
          <br />
          <em className="inline-em-route">made interactive.</em>
        </h1>
        <p>
          Every number below is generated from{" "}
          <code>public/global/simulation/simulation_summary*.csv</code> and the simulator's
          own logs — nothing is hand-copied, and the exclusions the paper applies are
          applied here too.
        </p>
      </header>

      <section className="viz-section">
        <div className="viz-kicker">How a policy is assembled</div>
        <PolicyPipeline />
      </section>

      <section className="viz-section">
        <div className="viz-kicker">Which bins get skipped today</div>
        <BinSelection results={data} />
      </section>

      <section className="viz-section">
        <div className="viz-kicker">Thirty days of routes</div>
        <RouteViewer />
      </section>

      <section className="viz-section">
        <div className="viz-kicker">The results, with their caveats</div>
        {status === "loading" && <div className="viz-loading">Loading results…</div>}
        {status === "error" && (
          <div className="viz-error">Could not load results: {error ?? "unknown error"}</div>
        )}
        {status === "ready" && data && <ResultsCharts results={data} />}
      </section>

      <section className="viz-section">
        <div className="viz-panel">
          <div className="viz-panel-head">
            <span>Data integrity</span>
            <CircleAlert size={15} />
          </div>
          <p style={{ color: "var(--text-muted)", lineHeight: 1.55 }}>
            Four runs are degenerate — three SWC-TCF runs at 30 days on Figueira da Foz
            (N=350, Gamma-3) and one at 90 days — having collected 30–86% less than their
            scenario median. They and their whole scenario cells are excluded so that every
            constructor is averaged over an identical set of scenarios. The 90-day sample is
            also conditioned on 30-day Pareto-front membership, so cross-constructor 90-day
            comparisons are not shown anywhere on this page; horizon results are paired
            against the same configuration.
          </p>
          {data && data.excluded.length > 0 && (
            <button
              className="viz-preset"
              style={{ marginTop: 12 }}
              onClick={() => setShowExcluded((v) => !v)}
              aria-expanded={showExcluded}
            >
              {showExcluded ? "Hide" : "Show"} excluded runs ({data.excluded.length})
            </button>
          )}
          {showExcluded && data && (
            <table className="viz-table" style={{ marginTop: 12 }}>
              <thead>
                <tr>
                  <th>Constructor</th>
                  <th>Horizon</th>
                  <th>Scenario</th>
                  <th>Policy</th>
                  <th>kg</th>
                  <th>Shortfall</th>
                </tr>
              </thead>
              <tbody>
                {data.excluded.map((e, i) => (
                  <tr key={i}>
                    <td>{e.constructor}</td>
                    <td>{e.horizon}d</td>
                    <td>
                      {e.city} · N={e.N}
                    </td>
                    <td>{e.policy}</td>
                    <td>{e.kg.toLocaleString()}</td>
                    <td>{Math.round(e.shortfall * 100)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </div>
  );
}
