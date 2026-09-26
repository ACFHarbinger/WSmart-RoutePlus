import {
  ArrowUpRight,
  Compass,
  Database,
  FlaskConical,
  Route,
  
} from "lucide-react";
import { Link } from "react-router-dom";
import RouteAtlas from "../components/RouteAtlas";
import RouteGraph from "../components/RouteGraph";
import { metrics, platformTracks, solvers } from "../data/site";

export default function Home() {
  return (
    <>
      <section className="hero hero-carto">
        <div className="hero-content" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '48px', alignItems: 'center' }}>
          <div>
            <div className="civic-heading">Operations Research</div>
            <h1 className="hero-title">
              Routing intelligence for
              <br />
              <span style={{ color: 'var(--brand-route)' }}>civic infrastructure.</span>
            </h1>
            <p className="hero-subtitle">
              WSmart+ Route bridges deep reinforcement learning and classical
              operations research for waste collection vehicle routing — 
              train policies, prove baselines, and stress multi-day scenarios.
            </p>
            <div style={{ display: 'flex', gap: '12px', marginBottom: '32px', fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
              <span style={{ border: '1px solid var(--border-strong)', padding: '4px 8px' }}>VRPP</span>
              <span style={{ border: '1px solid var(--border-strong)', padding: '4px 8px' }}>CWC VRP</span>
              <span style={{ border: '1px solid var(--border-strong)', padding: '4px 8px' }}>SCWCVRP</span>
            </div>
            <div className="hero-actions" style={{ display: 'flex', gap: '16px' }}>
              <Link to="/platform" style={{ background: 'var(--btn-bg)', color: 'var(--btn-text)', padding: '12px 24px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                Explore Platform <ArrowUpRight size={16} />
              </Link>
              <Link to="/research" style={{ border: '1px solid var(--border-strong)', padding: '12px 24px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                Research Map <Compass size={16} />
              </Link>
            </div>
          </div>

          <div className="hero-visual" style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-strong)', padding: '16px', position: 'relative' }}>
            <div style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px', marginBottom: '16px', display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              <span>Live Route Canvas</span>
              <span style={{ color: 'var(--brand-eco)' }}>● Active</span>
            </div>
            <RouteGraph />
          </div>
        </div>
      </section>

      <section className="metric-rail" aria-label="Platform signals">
        {metrics.map((m) => (
          <div key={m.label}>
            <span className="label">{m.label}</span>
            <strong>{m.value}</strong>
            <small>{m.note}</small>
          </div>
        ))}
      </section>

      <section className="section">
        <div className="section-head">
          <div>
            <div className="kicker">The workbench</div>
            <h2>
              One platform,
              <br />
              <em>three ways to think.</em>
            </h2>
          </div>
          <p>
            Routing is not one algorithmic personality. WSmart+ Route lets you
            learn, prove, and test decisions against the conditions that make
            operations difficult.
          </p>
        </div>
        <div className="card-grid">
          {platformTracks.map((track, index) => (
            <article className={`card ${track.color}`} key={track.key}>
              <div className="card-top">
                <span>{track.eyebrow}</span>
                <span>0{index + 1}</span>
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

      <section className="section">
        <div className="section-head">
          <div>
            <div className="kicker">Classical stack</div>
            <h2>
              Baselines that
              <br />
              <em>earn their place.</em>
            </h2>
          </div>
          <p>
            Neural agents are only as credible as the solvers they compete with.
            Exact methods, metaheuristics, and constraint solvers share the same
            problem surfaces.
          </p>
        </div>
        <div className="solver-row">
          {solvers.map((s) => (
            <div className="solver-pill" key={s.name}>
              <strong>{s.name}</strong>
              <span>{s.kind}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="split">
          <div>
            <div className="kicker">A route, made legible</div>
            <h2 className="inline-h2">
              See the decision
              <br />
              <em className="inline-em-route">before the score.</em>
            </h2>
            <p className="muted-lede">
              Data, experiment, and decision form one loop. The real platform
              scales this way of thinking to policies, fleets, time horizons, and
              uncertainty.
            </p>
            <div className="loop-list">
              <div className="loop-item">
                <Database size={18} />
                <div>
                  <strong>Data</strong>
                  <p>Instances, scenarios, fills, demands, geography.</p>
                </div>
              </div>
              <div className="loop-item">
                <FlaskConical size={18} />
                <div>
                  <strong>Experiment</strong>
                  <p>Policies, solvers, sweeps, reproducible runs.</p>
                </div>
              </div>
              <div className="loop-item">
                <Route size={18} />
                <div>
                  <strong>Decision</strong>
                  <p>Routes inspected, compared, and improved.</p>
                </div>
              </div>
            </div>
          </div>
          <div className="panel">
            <div className="panel-head">
              <span>Route atlas · illustrative</span>
              <span className="ok">interactive</span>
            </div>
            <RouteAtlas />
          </div>
        </div>
      </section>

      <section className="closing">
        <div>
          <div className="kicker">Enter the Studio</div>
          <h2>
            Make the hard parts{" "}
            <em className="inline-em-eco">visible.</em>
          </h2>
          <p>
            Explore the framework, inspect the evidence, or open the desktop
            Studio when you are ready to run a scenario.
          </p>
        </div>
        <Link className="btn btn-primary" to="/studio">
          Discover Studio <ArrowUpRight size={16} />
        </Link>
      </section>
    </>
  );
}
