import type { CatalogEntry, PolicyCatalog } from "../../../types/algorithms";
import {
  PALETTE_GROUPS,
  entryError,
  entryLabel,
  groupEntries,
  isNotRoutePolicy,
  statusClass,
} from "../../../pages/algorithms/builder/catalogView";

interface ModulePaletteProps {
  catalog: PolicyCatalog | null;
  onAdd: (entry: CatalogEntry) => void;
}

export function ModulePalette({ catalog, onAdd }: ModulePaletteProps) {
  return (
    <aside className="w-72 shrink-0 overflow-auto border-r border-noc-border bg-noc-panel p-3 space-y-4">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-noc-textMuted">Modules</h2>
      {!catalog && (
        <p className="text-xs text-noc-textMuted">Set a project root to load the module catalog.</p>
      )}
      {catalog &&
        PALETTE_GROUPS.map((group) => {
          const entries = groupEntries(catalog, group.key);
          return (
            <section key={group.key} aria-label={group.label} className="space-y-1.5">
              <h3 className="text-[11px] font-semibold uppercase tracking-wide text-noc-textSecondary">
                {group.label}
              </h3>
              {entries.length === 0 && <p className="text-[11px] text-noc-textMuted">None in catalog.</p>}
              {entries.map((entry) => (
                <PaletteRow key={`${group.key}-${entry.id}`} entry={entry} onAdd={onAdd} />
              ))}
            </section>
          );
        })}
    </aside>
  );
}

function PaletteRow({ entry, onAdd }: { entry: CatalogEntry; onAdd: (entry: CatalogEntry) => void }) {
  const blocked = isNotRoutePolicy(entry);
  const error = entryError(entry);
  const status = String((entry as { status?: string }).status ?? "present");
  const tone = blocked ? "bg-noc-card border-noc-border text-noc-textMuted" : statusClass(status);
  return (
    <div className={`rounded-lg border px-2 py-1.5 ${tone}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-xs font-medium">{entryLabel(entry)}</p>
          <p className="text-[10px] uppercase tracking-wide opacity-80">
            {blocked ? "not a route policy" : status}
          </p>
        </div>
        <button
          type="button"
          className="shrink-0 rounded border border-noc-border px-2 py-0.5 text-[11px] text-noc-textPrimary disabled:cursor-not-allowed disabled:opacity-40"
          aria-label={`Add ${entry.id}`}
          disabled={blocked}
          onClick={() => onAdd(entry)}
        >
          Add
        </button>
      </div>
      {error && <p className="mt-1 text-[10px] text-noc-crit">{error}</p>}
    </div>
  );
}
