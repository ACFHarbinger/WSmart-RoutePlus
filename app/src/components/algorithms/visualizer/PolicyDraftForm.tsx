import type { FormEvent } from "react";
import type { PolicyCatalog } from "../../../types/algorithms";
import {
  catalogSection,
  entryId,
  entryStatus,
  entryTitle,
  isRouteProducer,
  statusClass,
} from "./catalogEntries";

export interface DraftPatch {
  problem?: string;
  graphSize?: number;
  seed?: number;
  packing?: string | null;
  routing?: string | null;
  model?: string | null;
  routeImprovement?: string | null;
  acceptance?: string | null;
  operators?: string[];
}

interface PolicyDraftFormProps {
  problem: string;
  graphSize: number;
  seed: number;
  packing: string;
  routing: string;
  model: string;
  routeImprovement: string;
  acceptance: string;
  operators: string[];
  catalog: PolicyCatalog | null;
  catalogPhase: "idle" | "loading" | "ready" | "empty" | "error";
  catalogError: string | null;
  runNotice: string | null;
  runError: string | null;
  running: boolean;
  onPatch: (patch: DraftPatch) => void;
  onRun: () => void;
}

const PROBLEMS = ["ptp", "cptp", "cptptc", "cptptwtc", "cvrp"] as const;

function CatalogSelect({
  id,
  label,
  value,
  section,
  catalog,
  optional,
  routeProducers,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  section: string;
  catalog: PolicyCatalog | null;
  optional: boolean;
  routeProducers: boolean;
  onChange: (value: string) => void;
}) {
  const entries = catalogSection(catalog, section).filter((entry) => !routeProducers || isRouteProducer(entry));
  const known = new Set(entries.map((entry) => entryId(entry)));
  const selected = entries.find((entry) => entryId(entry) === value);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={id} className="text-xs text-noc-textMuted">
          {label}
        </label>
        {selected ? (
          <span className={`text-[10px] uppercase ${statusClass(entryStatus(selected))}`} data-testid={`${id}-status`}>
            {entryStatus(selected)}
          </span>
        ) : null}
      </div>
      <select
        id={id}
        className="select-base text-xs"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {optional ? <option value="">None</option> : null}
        {value && !known.has(value) ? <option value={value}>{value}</option> : null}
        {entries.map((entry, index) => {
          const entryKey = entryId(entry);
          const broken = Boolean((entry as unknown as { error?: string }).error);
          return (
            <option
              key={`${entryKey}-${index}`}
              value={entryKey}
              className={broken ? "text-noc-crit" : statusClass(entryStatus(entry))}
            >
              {entryTitle(entry)} · {entryStatus(entry)}
              {broken ? " · error" : ""}
            </option>
          );
        })}
      </select>
    </div>
  );
}

export function PolicyDraftForm({
  problem,
  graphSize,
  seed,
  packing,
  routing,
  model,
  routeImprovement,
  acceptance,
  operators,
  catalog,
  catalogPhase,
  catalogError,
  runNotice,
  runError,
  running,
  onPatch,
  onRun,
}: PolicyDraftFormProps) {
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!running) onRun();
  };
  return (
    <form onSubmit={submit} className="bg-noc-card border border-noc-border rounded-xl p-3 space-y-3">
      <div>
        <h2 className="text-sm font-medium text-noc-textPrimary">Trace setup</h2>
        <p className="text-[11px] text-noc-textMuted mt-1">
          <span className="text-noc-cyan">Registered</span>
          {" and shipped modules are distinct from "}
          <span className="text-noc-textMuted">present</span>
          {" modules. The trace command solves alns only."}
        </p>
      </div>
      {catalogPhase === "loading" ? (
        <p className="text-xs text-noc-textMuted" role="status">
          Loading policy catalog…
        </p>
      ) : null}
      {catalogPhase === "empty" ? (
        <p className="text-xs text-noc-textMuted" role="status">
          Policy catalog is empty.
        </p>
      ) : null}
      {catalogPhase === "error" ? (
        <p className="text-xs text-noc-crit" role="alert">
          Catalog command failed{catalogError ? `: ${catalogError}` : "."}
        </p>
      ) : null}
      {catalogPhase === "idle" ? (
        <p className="text-xs text-noc-textMuted" role="status">
          Set a project root to load the policy catalog.
        </p>
      ) : null}
      <div className="flex flex-col gap-1">
        <label htmlFor="algo-problem" className="text-xs text-noc-textMuted">
          Problem
        </label>
        <select
          id="algo-problem"
          className="select-base text-xs"
          value={problem}
          onChange={(event) => onPatch({ problem: event.target.value })}
        >
          {PROBLEMS.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="flex flex-col gap-1">
          <label htmlFor="algo-graph-size" className="text-xs text-noc-textMuted">
            Graph size
          </label>
          <input
            id="algo-graph-size"
            className="input-base text-xs"
            type="number"
            min={3}
            value={graphSize}
            onChange={(event) => onPatch({ graphSize: Number(event.target.value) })}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="algo-seed" className="text-xs text-noc-textMuted">
            Seed
          </label>
          <input
            id="algo-seed"
            className="input-base text-xs"
            type="number"
            value={seed}
            onChange={(event) => onPatch({ seed: Number(event.target.value) })}
          />
        </div>
      </div>
      <CatalogSelect
        id="algo-packing"
        label="Packing"
        value={packing}
        section="packing"
        catalog={catalog}
        optional
        routeProducers={false}
        onChange={(next) => onPatch({ packing: next || null })}
      />
      <CatalogSelect
        id="algo-routing"
        label="Routing"
        value={routing}
        section="routing"
        catalog={catalog}
        optional
        routeProducers
        onChange={(next) => onPatch({ routing: next || null, model: next ? null : model || null })}
      />
      <CatalogSelect
        id="algo-model"
        label="Model"
        value={model}
        section="models"
        catalog={catalog}
        optional
        routeProducers
        onChange={(next) => onPatch({ model: next || null, routing: next ? null : routing || null })}
      />
      <CatalogSelect
        id="algo-improvement"
        label="Route improvement"
        value={routeImprovement}
        section="route_improvement"
        catalog={catalog}
        optional
        routeProducers={false}
        onChange={(next) => onPatch({ routeImprovement: next || null })}
      />
      <CatalogSelect
        id="algo-acceptance"
        label="Acceptance"
        value={acceptance}
        section="acceptance"
        catalog={catalog}
        optional
        routeProducers={false}
        onChange={(next) => onPatch({ acceptance: next || null })}
      />
      {operators.length > 0 ? (
        <p className="text-[11px] text-noc-textMuted">Draft operators: {operators.join(", ")}</p>
      ) : null}
      {runNotice ? (
        <p className="text-xs text-noc-amber" role="status" data-testid="trace-run-state">
          {runNotice}
        </p>
      ) : null}
      {runError ? (
        <p className="text-xs text-noc-crit" role="alert">
          Trace command failed{runError ? `: ${runError}` : "."}
        </p>
      ) : null}
      <button type="submit" className="btn-primary text-sm w-full" disabled={running}>
        {running ? "Running trace…" : "Run trace"}
      </button>
    </form>
  );
}
