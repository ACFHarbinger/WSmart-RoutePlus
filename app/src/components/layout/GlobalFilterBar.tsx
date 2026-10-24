/**
 * App-wide policy/sample filter controls (§G.7).
 * Options derived from entries loaded in the simulation log store.
 */
import { X } from "lucide-react";
import { useGlobalFiltersStore } from "../../store/filters";
import { useSimStore, uniquePolicies, uniqueSamples } from "../../store/sim";

interface Props {
  /** Override policy options (e.g. CSV-derived values in Data Explorer). */
  policies?: string[];
  /** Portfolio ``run_label`` options when a multi-run table is active (§G.6). */
  runLabels?: string[];
  /** City/scale group options for portfolio city brush (§G.6). */
  cities?: string[];
  /** Show global log-scale toggle for analytics charts (§G.7). */
  showLogScale?: boolean;
}

export function GlobalFilterBar({
  policies: policiesProp,
  runLabels = [],
  cities = [],
  showLogScale = false,
}: Props) {
  const { entries } = useSimStore();
  const {
    policy,
    sampleId,
    runLabel,
    brushedCity,
    logScale,
    setPolicy,
    setSampleId,
    setRunLabel,
    setBrushedCity,
    setLogScale,
    reset,
  } = useGlobalFiltersStore();

  const policies = policiesProp?.length ? policiesProp : uniquePolicies(entries);
  const samples = uniqueSamples(entries);

  if (
    !showLogScale &&
    policies.length === 0 &&
    entries.length === 0 &&
    policy == null &&
    sampleId == null &&
    runLabel == null &&
    brushedCity == null &&
    runLabels.length === 0 &&
    cities.length === 0
  ) {
    return null;
  }

  return (
    <div className="flex items-center gap-2 flex-wrap text-xs bg-noc-panel border border-noc-border rounded-xl px-3 py-1.5 select-none shadow-sm">
      <span className="text-noc-textMuted shrink-0 font-medium">Filters</span>

      {policies.length > 0 && (
        <select
          className="select-base w-36 text-xs"
          value={policy ?? ""}
          onChange={(e) => setPolicy(e.target.value || null)}
        >
          <option value="">All policies</option>
          {policies.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      )}

      {samples.length > 1 && (
        <select
          className="select-base w-28 text-xs"
          value={sampleId ?? ""}
          onChange={(e) => setSampleId(e.target.value ? Number(e.target.value) : null)}
        >
          <option value="">All samples</option>
          {samples.map((s) => (
            <option key={s} value={s}>Sample {s}</option>
          ))}
        </select>
      )}

      {cities.length > 1 && (
        <select
          className="select-base w-36 text-xs"
          value={brushedCity ?? ""}
          onChange={(e) => {
            setRunLabel(null);
            setBrushedCity(e.target.value || null);
          }}
        >
          <option value="">All cities</option>
          {cities.map((city) => (
            <option key={city} value={city}>{city}</option>
          ))}
        </select>
      )}

      {runLabels.length > 0 && (
        <select
          className="select-base w-44 text-xs"
          value={runLabel ?? ""}
          onChange={(e) => {
            setBrushedCity(null);
            setRunLabel(e.target.value || null);
          }}
        >
          <option value="">All runs</option>
          {runLabels.map((label) => (
            <option key={label} value={label}>{label}</option>
          ))}
        </select>
      )}

      {showLogScale && (
        <button
          type="button"
          onClick={() => setLogScale(!logScale)}
          className={`btn-ghost text-xs ${logScale ? "text-accent-secondary" : ""}`}
        >
          {logScale ? "Log scale (on)" : "Log scale (off)"}
        </button>
      )}

      {(policy || sampleId != null || runLabel || brushedCity || logScale) && (
        <button
          onClick={reset}
          className="btn-ghost text-xs flex items-center gap-1 text-canvas-muted"
          title="Clear filters"
        >
          <X size={11} />
          Clear
        </button>
      )}
    </div>
  );
}
