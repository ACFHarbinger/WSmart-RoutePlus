import type { CatalogEntry, PolicyCatalog } from "../../../types/algorithms";

export interface DeclaredParameter {
  name: string;
  label?: string;
  default?: string | number | boolean;
}

export const PALETTE_GROUPS: { key: string; label: string }[] = [
  { key: "packing", label: "Packing" },
  { key: "routing", label: "Routing" },
  { key: "route_improvement", label: "Route improvement" },
  { key: "acceptance", label: "Acceptance" },
  { key: "operators", label: "Operators" },
  { key: "separation", label: "Cuts and inequalities" },
  { key: "models", label: "Models" },
];

export function groupEntries(catalog: PolicyCatalog, key: string): CatalogEntry[] {
  const bag = catalog as unknown as Record<string, unknown>;
  const rows = bag[key];
  return Array.isArray(rows) ? (rows as CatalogEntry[]) : [];
}

export function entryLabel(entry: CatalogEntry): string {
  const row = entry as CatalogEntry & { display_name?: string; displayName?: string };
  return row.display_name || row.displayName || entry.id;
}

export function entryError(entry: CatalogEntry): string | null {
  const value = (entry as { error?: unknown }).error;
  if (typeof value !== "string") return null;
  const text = value.trim();
  return text.length > 0 ? text : null;
}

export function entryRole(entry: CatalogEntry): string | undefined {
  const role = (entry as { role?: unknown }).role;
  return typeof role === "string" && role.length > 0 ? role : undefined;
}

/** WeightAdjustmentRNN adjusts rewards. It is not a module the canvas can place. */
export function isNotRoutePolicy(entry: CatalogEntry): boolean {
  const role = entryRole(entry) ?? "";
  return (
    entry.id === "WeightAdjustmentRNN" ||
    role === "reward_weight_adjustment" ||
    role === "reward-weight-adjustment"
  );
}

export function statusClass(status: string): string {
  if (status === "present") {
    return "bg-noc-card border-noc-border text-noc-textMuted";
  }
  return "bg-noc-card border-noc-cyan text-noc-cyan";
}

export function declaredParameters(entry: CatalogEntry): DeclaredParameter[] {
  const raw = (entry as { parameters?: unknown }).parameters;
  if (!Array.isArray(raw)) return [];
  const params: DeclaredParameter[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object" || !("name" in item)) continue;
    const name = (item as { name?: unknown }).name;
    if (typeof name !== "string" || name.length === 0) continue;
    const label = (item as { label?: unknown }).label;
    const fallback = (item as { default?: unknown }).default;
    params.push({
      name,
      label: typeof label === "string" ? label : undefined,
      default:
        typeof fallback === "string" || typeof fallback === "number" || typeof fallback === "boolean"
          ? fallback
          : undefined,
    });
  }
  return params;
}

export function findEntry(
  catalog: PolicyCatalog,
  entryId: string,
  family?: string
): CatalogEntry | undefined {
  for (const group of PALETTE_GROUPS) {
    const match = groupEntries(catalog, group.key).find((entry) => entry.id === entryId);
    if (!match) continue;
    if (!family || match.family === family) return match;
  }
  for (const group of PALETTE_GROUPS) {
    const match = groupEntries(catalog, group.key).find((entry) => entry.id === entryId);
    if (match) return match;
  }
  return undefined;
}
