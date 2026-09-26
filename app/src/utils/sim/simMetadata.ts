/**
 * Parse simulation log paths and policy labels (§G.1 metadata).
 * Mirrors conventions in logic/gen/json/simulation_metadata.json.
 */

export interface LogPathMeta {
  city: string | null;
  cityShort: string | null;
  scale: number | null;
  distribution: string | null;
  distributionKey: string | null;
  strategyFolder: string | null;
  improver: string | null;
}

export interface PolicyMeta {
  selectionStrategy: string;
  constructor: string;
  improver: string;
  distribution: string;
}

const CITY_SHORT: Record<string, string> = {
  "Rio Maior": "RM",
  "Figueira da Foz": "FFZ",
};

const DIST_LABELS: Record<string, string> = {
  emp: "Empirical",
  empirical: "Empirical",
  gamma1: "Gamma-1",
  gamma2: "Gamma-2",
  gamma3: "Gamma-3",
};

/** Parse output-directory segments for city / distribution / strategy folder. */
export function parseLogPath(path: string | null | undefined): LogPathMeta {
  if (!path) {
    return {
      city: null,
      cityShort: null,
      scale: null,
      distribution: null,
      distributionKey: null,
      strategyFolder: null,
      improver: null,
    };
  }

  const lower = path.toLowerCase();
  const parts = lower.split(/[/\\]/);

  let city: string | null = null;
  let scale: number | null = null;
  for (const part of parts) {
    const rm = part.match(/riomaior(\d+)?/);
    const ffz = part.match(/figueiradafoz(\d+)?|figdafoz(\d+)?/);
    if (rm) {
      city = "Rio Maior";
      scale = rm[1] ? Number(rm[1]) : scale;
    }
    if (ffz) {
      city = "Figueira da Foz";
      scale = ffz[1] ? Number(ffz[1]) : ffz[2] ? Number(ffz[2]) : scale;
    }
    const num = part.match(/(\d{2,3})(?:v|bins)?/i);
    if (num && !scale) scale = Number(num[1]);
  }

  let distributionKey: string | null = null;
  for (const key of Object.keys(DIST_LABELS)) {
    if (parts.some((p) => p === key || p.includes(key))) {
      distributionKey = key;
      break;
    }
  }

  const strategyFolder =
    parts.find((p) => /^(la|lm|sl)_(ftsp|cls)$/i.test(p) || /^(lookahead|last_minute|service_level)/i.test(p)) ??
    null;

  let improver: string | null = null;
  if (strategyFolder) {
    if (strategyFolder.includes("ftsp")) improver = "FTSP";
    if (strategyFolder.includes("cls")) improver = "CLS";
  }

  const cityShort = city ? CITY_SHORT[city] ?? null : null;
  const distribution = distributionKey ? DIST_LABELS[distributionKey] ?? distributionKey : null;

  return {
    city,
    cityShort,
    scale,
    distribution,
    distributionKey,
    strategyFolder,
    improver,
  };
}

/** Parse a GUI policy label into selection / constructor / improver / distribution. */
export function parsePolicyLabel(policy: string): PolicyMeta {
  const distMatch = policy.match(/\(([^)]+)\)\s*$/i);
  const distribution = distMatch?.[1]?.trim() ?? "—";

  const body = distMatch ? policy.slice(0, distMatch.index).trim() : policy;
  const segments = body.split(/\s*\+\s*/).map((s) => s.trim()).filter(Boolean);

  let selectionStrategy = segments[0] ?? policy;
  let constructor = segments[1] ?? "—";
  let improver = segments[2] ?? "—";

  const sl = selectionStrategy.match(/last\s+minute\s+cf(\d+)/i);
  if (sl) selectionStrategy = `LM-CF${sl[1]}`;
  else if (/last\s+minute/i.test(selectionStrategy)) selectionStrategy = "LM";
  else if (/look[- ]?ahead/i.test(selectionStrategy)) selectionStrategy = "LA";
  else if (/service\s*level\s*1|sl[- ]?1/i.test(selectionStrategy)) selectionStrategy = "SL-SL1";
  else if (/service\s*level\s*2|sl[- ]?2/i.test(selectionStrategy)) selectionStrategy = "SL-SL2";

  constructor = constructor.replace(/_CUSTOM$/i, "").replace(/_/g, "-").toUpperCase();
  improver = improver.replace(/ftsp/i, "FTSP").replace(/cls/i, "CLS");

  return { selectionStrategy, constructor, improver, distribution };
}

export function formatLogMeta(meta: LogPathMeta): string {
  const bits = [
    meta.cityShort && meta.scale ? `${meta.cityShort}-${meta.scale}` : meta.city,
    meta.distribution,
    meta.strategyFolder?.replace(/_/g, "-").toUpperCase(),
    meta.improver,
  ].filter(Boolean);
  return bits.join(" · ") || "—";
}

export function formatPolicyMeta(meta: PolicyMeta): string {
  return `${meta.selectionStrategy} · ${meta.constructor} · ${meta.improver} · ${meta.distribution}`;
}

export const SELECTION_STRATEGY_LEGEND = [
  "LA",
  "LM",
  "LM-CF70",
  "LM-CF90",
  "SL-SL1",
  "SL-SL2",
] as const;

const STRATEGY_COLORS: Record<string, string> = {
  LA: "#6366f1",
  LM: "#fbbf24",
  "LM-CF70": "#fb923c",
  "LM-CF90": "#f87171",
  "SL-SL1": "#34d399",
  "SL-SL2": "#38bdf8",
};

export function strategyColor(policy: string, policyMeta?: Record<string, PolicyMeta>): string {
  const strat = policyMeta?.[policy]?.selectionStrategy ?? parsePolicyLabel(policy).selectionStrategy;
  return selectionStrategyColor(strat);
}

/** Colour for a resolved mandatory-selection strategy label (LA · LM-CF70 · …). */
export function selectionStrategyColor(strategy: string): string {
  return STRATEGY_COLORS[strategy] ?? "#a78bfa";
}

/** Resolve mandatory-selection strategy for a portfolio run (log path or dominant policy). */
export function resolveRunSelectionStrategy(path: string, policies: string[] = []): string {
  const lower = path.toLowerCase();
  if (/cf\s*90|cf90/i.test(lower)) return "LM-CF90";
  if (/cf\s*70|cf70/i.test(lower)) return "LM-CF70";

  const { strategyFolder } = parseLogPath(path);
  if (strategyFolder) {
    const folder = strategyFolder.toLowerCase();
    if (folder.startsWith("la") || folder.includes("lookahead")) return "LA";
    if (folder.startsWith("lm") || folder.includes("last_minute")) return "LM";
    if (folder.includes("sl1") || folder.includes("sl_1")) return "SL-SL1";
    if (folder.includes("sl2") || folder.includes("sl_2")) return "SL-SL2";
    if (folder.startsWith("sl") || folder.includes("service_level")) return "SL-SL1";
  }

  if (policies.length) {
    const counts = new Map<string, number>();
    for (const policy of policies) {
      const strat = parsePolicyLabel(policy).selectionStrategy;
      counts.set(strat, (counts.get(strat) ?? 0) + 1);
    }
    let best = "—";
    let bestN = 0;
    for (const [strat, n] of counts) {
      if (n > bestN) {
        best = strat;
        bestN = n;
      }
    }
    if (best !== "—") return best;
  }

  return "—";
}

export function citySymbol(logMeta: LogPathMeta | null): "circle" | "rect" | "diamond" {
  if (logMeta?.cityShort === "FFZ") return "diamond";
  if (logMeta?.scale === 170) return "rect";
  return "circle";
}

export function cityScaleLabel(logMeta: LogPathMeta): string {
  if (logMeta.cityShort && logMeta.scale) return `${logMeta.cityShort}-${logMeta.scale}`;
  return logMeta.city ?? "Run";
}
