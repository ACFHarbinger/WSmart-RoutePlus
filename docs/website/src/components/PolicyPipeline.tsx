import { useMemo, useState } from "react";
import type { Pipeline } from "../data/types";
import { useJson } from "../hooks/useJson";

/**
 * V1 — The three-stage policy pipeline as a working diagram. The visitor
 * picks one option per stage and sees the configuration they have built,
 * with the real algorithm names enumerated from the plugin registries.
 */
export default function PolicyPipeline() {
  const { data, status, error } = useJson<Pipeline>("/data/pipeline.json");
  const [selection, setSelection] = useState<string>("");
  const [constructor, setConstructor] = useState<string>("");
  const [improver, setImprover] = useState<string>("");

  const constructors = useMemo(() => {
    if (!data) return [];
    return data.construction.flatMap((f) =>
      f.constructors.map((c) => ({ ...c, family: f.family })),
    );
  }, [data]);

  if (status === "loading") return <div className="viz-loading">Loading policy space…</div>;
  if (status === "error" || !data)
    return <div className="viz-error">Could not load policy space: {error ?? "unknown error"}</div>;

  const selName = data.selection.find((s) => s.key === selection)?.name;
  const ctorName = constructors.find((c) => c.key === constructor)?.name;
  const impName = data.improvement.find((i) => i.key === improver)?.name;

  const isBenchmarked = (kind: "selection" | "constructors" | "improvers", name?: string) =>
    !!name && data.benchmark[kind].includes(name);

  return (
    <div className="viz-panel">
      <div className="viz-panel-head">
        <span>Policy configuration space</span>
        <span>
          {data.counts.selection} strategies · {data.counts.families} families ·{" "}
          {data.counts.constructors} constructors · {data.counts.improvers} improvers
        </span>
      </div>
      <div className="viz-pipeline">
        <div className="viz-stage">
          <div className="viz-stage-label">Stage 1 · Mandatory selection</div>
          <div className="viz-stage-count">{data.counts.selection} strategies</div>
          <select
            className="viz-select"
            value={selection}
            onChange={(e) => setSelection(e.target.value)}
            aria-label="Choose a mandatory-selection strategy"
          >
            <option value="">Select a strategy…</option>
            {data.selection.map((s) => (
              <option key={s.key} value={s.key}>
                {s.name}
              </option>
            ))}
          </select>
        </div>

        <div className="viz-stage">
          <div className="viz-stage-label">Stage 2 · Route construction</div>
          <div className="viz-stage-count">
            {data.counts.constructors} constructors · {data.counts.families} families
          </div>
          <select
            className="viz-select"
            value={constructor}
            onChange={(e) => setConstructor(e.target.value)}
            aria-label="Choose a route constructor"
          >
            <option value="">Select a constructor…</option>
            {data.construction.map((f) => (
              <optgroup key={f.key} label={f.family}>
                {f.constructors.map((c) => (
                  <option key={c.key} value={c.key}>
                    {c.name}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>

        <div className="viz-stage">
          <div className="viz-stage-label">Stage 3 · Route improvement</div>
          <div className="viz-stage-count">{data.counts.improvers} improvers</div>
          <select
            className="viz-select"
            value={improver}
            onChange={(e) => setImprover(e.target.value)}
            aria-label="Choose a route improver"
          >
            <option value="">Select an improver…</option>
            {data.improvement.map((i) => (
              <option key={i.key} value={i.key}>
                {i.name}
              </option>
            ))}
          </select>
        </div>

        <div className="viz-config" aria-live="polite">
          {selName || ctorName || impName ? (
            <>
              Your policy: <strong>{selName ?? "?"}</strong> →{" "}
              <strong>{ctorName ?? "?"}</strong> → <strong>{impName ?? "?"}</strong>
              {isBenchmarked("selection", selName) && <span className="viz-badge">BENCHMARKED</span>}
            </>
          ) : (
            <>Pick one option per stage to assemble a policy.</>
          )}
        </div>
      </div>
      <p className="viz-note">
        The benchmark in the paper crosses a deliberately small, representative subset:
        five selection variants, eight constructors and two improvers. Everything else is
        registered and interchangeable by configuration, but not exercised in the study.
      </p>
    </div>
  );
}
