import { ArrowLeft, ArrowUpRight, Cpu, Layers3, MonitorCog } from "lucide-react";
import { Link } from "react-router-dom";
import { platformTracks } from "../data/site";

export default function Platform() {
  return (
    <div className="interior">
      <Link className="back-link" to="/">
        <ArrowLeft size={15} /> Home
      </Link>
      <header className="page-hero">
        <span className="eyebrow">
          <span className="eyebrow-dot" />
          Platform · architecture
        </span>
        <h1>
          The laboratory behind
          <br />
          <em className="inline-em-route">the route.</em>
        </h1>
        <p>
          WSmart Route+ separates computational logic from the Studio interface
          so research can evolve without turning experimentation into a black
          box.
        </p>
      </header>

      <div className="card-grid interior-grid">
        <article className="card route">
          <div className="card-top">
            <span>LOGIC</span>
            <Cpu size={18} />
          </div>
          <h3>Algorithms and environments</h3>
          <p>
            Neural models, classical policies, data generation, configurations,
            and simulation pipelines live in the logic layer.
          </p>
        </article>
        <article className="card eco">
          <div className="card-top">
            <span>STUDIO</span>
            <MonitorCog size={18} />
          </div>
          <h3>One operational surface</h3>
          <p>
            The Tauri Studio launches workflows, watches processes, explores
            DuckDB-backed outputs, and turns runs into reports.
          </p>
        </article>
        <article className="card amber">
          <div className="card-top">
            <span>EVIDENCE</span>
            <Layers3 size={18} />
          </div>
          <h3>Every result has context</h3>
          <p>
            Benchmarks, telemetry, configurations, and scenario metadata make
            comparisons reproducible rather than anecdotal.
          </p>
        </article>
      </div>

      <section className="section">
        <div className="section-head">
          <div>
            <div className="kicker">Tracks</div>
            <h2>
              Three modes,
              <br />
              <em>one experimental surface.</em>
            </h2>
          </div>
          <p>
            Learn with neural policies, prove with classical solvers, and test
            under multi-day operational stress — without switching frameworks.
          </p>
        </div>
        <div className="card-grid">
          {platformTracks.map((track, i) => (
            <article className={`card ${track.color}`} key={track.key}>
              <div className="card-top">
                <span>{track.eyebrow}</span>
                <span>0{i + 1}</span>
              </div>
              <h3>{track.label}</h3>
              <p>
                <strong className="card-title-line">{track.title}</strong>
                {track.body}
              </p>
            </article>
          ))}
        </div>
      </section>

      <section className="text-panel">
        <div className="kicker">Design principle</div>
        <h2>
          Research freedom.
          <br />
          <em className="inline-em-eco">Operational discipline.</em>
        </h2>
        <p>
          The public website explains the architecture and its trade-offs. The
          desktop Studio performs the work. Neither surface should pretend to be
          the other.
        </p>
        <Link className="btn btn-secondary" to="/studio">
          See the Studio surface <ArrowUpRight size={15} />
        </Link>
      </section>
    </div>
  );
}
