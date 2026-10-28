import type { CatalogEntry, PolicyCatalog } from "../../../types/algorithms";
import { declaredParameters, findEntry } from "../../../pages/algorithms/builder/catalogView";
import type { BuilderNode } from "../../../pages/algorithms/builder/graphTypes";

interface ModuleInspectorProps {
  catalog: PolicyCatalog | null;
  node: BuilderNode | null;
  onParameter: (uid: string, name: string, value: string) => void;
  onRemove: (uid: string) => void;
}

export function ModuleInspector({ catalog, node, onParameter, onRemove }: ModuleInspectorProps) {
  const entry = node && catalog ? findEntry(catalog, node.entryId, node.family) : undefined;
  const params = entry ? declaredParameters(entry) : [];
  const consumes = stringList(entry, "consumes");
  const produces = stringList(entry, "produces");
  const operatorNames = stringList(entry, "operators");

  return (
    <aside className="w-80 shrink-0 overflow-auto border-l border-noc-border bg-noc-panel p-3 space-y-3">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-noc-textMuted">Inspector</h2>
      {!node && <p className="text-xs text-noc-textMuted">Select a module on the canvas.</p>}
      {node && (
        <div className="space-y-3">
          <div>
            <p className="text-sm font-medium text-noc-textPrimary">{node.displayName}</p>
            <p className="text-[11px] uppercase tracking-wide text-noc-textMuted">
              {node.family} · {node.status}
            </p>
          </div>
          <PortList title="Consumes" values={consumes} />
          <PortList title="Produces" values={produces} />
          {operatorNames.length > 0 && <PortList title="Operator files" values={operatorNames} />}
          {params.length === 0 ? (
            <p className="text-xs text-noc-textMuted">This module does not declare parameters.</p>
          ) : (
            <div className="space-y-2">
              {params.map((param) => (
                <label key={param.name} className="flex flex-col gap-1 text-xs text-noc-textMuted">
                  {param.label || param.name}
                  <input
                    className="rounded border border-noc-border bg-noc-deep px-2 py-1 text-sm text-noc-textPrimary"
                    value={node.parameters[param.name] ?? ""}
                    onChange={(event) => onParameter(node.uid, param.name, event.target.value)}
                  />
                </label>
              ))}
            </div>
          )}
          <button
            type="button"
            className="rounded border border-noc-border px-2 py-1 text-xs text-noc-textPrimary"
            onClick={() => onRemove(node.uid)}
          >
            Remove from canvas
          </button>
        </div>
      )}
    </aside>
  );
}

function PortList({ title, values }: { title: string; values: string[] }) {
  if (values.length === 0) return null;
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wide text-noc-textMuted">{title}</p>
      <p className="text-xs text-noc-textSecondary">{values.join(", ")}</p>
    </div>
  );
}

function stringList(entry: CatalogEntry | undefined, key: "consumes" | "produces" | "operators"): string[] {
  if (!entry) return [];
  const value = (entry as unknown as Record<string, unknown>)[key];
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}
