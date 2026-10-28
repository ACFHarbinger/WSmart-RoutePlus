import type { CatalogEntry, PolicyCatalog } from "../../../types/algorithms";

const SECTION_ALIASES: Record<string, string[]> = {
  packing: ["packing"],
  routing: ["routing"],
  route_improvement: ["route_improvement", "routeImprovement"],
  acceptance: ["acceptance"],
  models: ["models"],
  operators: ["operators"],
};

export function catalogSection(catalog: PolicyCatalog | null, section: string): CatalogEntry[] {
  if (!catalog) return [];
  const record = catalog as unknown as Record<string, unknown>;
  for (const key of SECTION_ALIASES[section] ?? [section]) {
    const value = record[key];
    if (Array.isArray(value)) return value as CatalogEntry[];
  }
  return [];
}

export function isCatalogEmpty(catalog: PolicyCatalog): boolean {
  const sections = ["packing", "routing", "route_improvement", "acceptance", "models"];
  return sections.every((section) => catalogSection(catalog, section).length === 0);
}

/** Reward-weight models do not produce routes. */
export function isRouteProducer(entry: CatalogEntry): boolean {
  const record = entry as unknown as Record<string, unknown>;
  const role = typeof record.role === "string" ? record.role : "";
  const id = String(record.id ?? "").toLowerCase();
  if (role === "reward_weight_adjustment") return false;
  if (id === "weight_adjustment_rnn") return false;
  return true;
}

export function entryId(entry: CatalogEntry): string {
  const id = (entry as unknown as { id?: string }).id;
  return typeof id === "string" ? id : "";
}

export function entryStatus(entry: CatalogEntry): string {
  const status = (entry as unknown as { status?: string }).status;
  return typeof status === "string" ? status : "";
}

export function entryTitle(entry: CatalogEntry): string {
  const record = entry as unknown as { display_name?: string; displayName?: string; id?: string };
  return record.display_name || record.displayName || record.id || "";
}

/** Registered and shipped ids use cyan. Present-only modules stay muted. */
export function statusClass(status: string): string {
  if (status === "present") return "text-noc-textMuted";
  if (status === "registered" || status === "shipped") return "text-noc-cyan";
  return "text-noc-textSecondary";
}
