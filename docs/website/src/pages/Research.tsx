import { ArrowLeft, ArrowUpRight } from "lucide-react";
import { Link } from "react-router-dom";
import { researchNotes } from "../data/site";

export default function Research() {
  return (
    <div className="interior">
      <Link className="back-link" to="/">
        <ArrowLeft size={15} /> Home
      </Link>
      <header className="page-hero">
        <span className="eyebrow">
          <span className="eyebrow-dot" />
          Research map · open questions
        </span>
        <h1>
          Where routing
          <br />
          <em className="inline-em-route">gets interesting.</em>
        </h1>
        <p>
          From profit-aware selection to stochastic waste collection, the
          research surface is organized around the decisions real operations
          force us to make.
        </p>
      </header>

      <div className="research-list">
        {researchNotes.map((note, index) => (
          <article key={note.title}>
            <span className="research-index">0{index + 1}</span>
            <div>
              <h2>{note.title}</h2>
              <p>{note.body}</p>
            </div>
            <ArrowUpRight size={17} className="research-arrow" />
          </article>
        ))}
      </div>

      <section className="text-panel">
        <div className="kicker">Research workflow</div>
        <h2>
          Formulate → train →
          <br />
          <em className="inline-em-eco">stress → learn.</em>
        </h2>
        <p>
          Research is only useful when a policy survives the conditions it was
          designed to navigate. WSmart Route+ keeps simulation and evaluation
          close to the model.
        </p>
        <Link className="btn btn-secondary" to="/benchmarks">
          View benchmark philosophy <ArrowUpRight size={15} />
        </Link>
      </section>
    </div>
  );
}
