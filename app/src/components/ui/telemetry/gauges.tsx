import { NOC_COLORS } from "./tokens";

export type ProvenanceTier =
  | "exact"
  | "incumbent"
  | "metaheuristic"
  | "neural"
  | "simulated"
  | "estimated";

const PROVENANCE_META: Record<
  ProvenanceTier,
  { label: string; title: string; color: string; bg: string }
> = {
  exact: {
    label: "EXACT (GAP 0%)",
    title: "Proven optimal bound (Lagrangian / MIP gap = 0%)",
    color: NOC_COLORS.cyan,
    bg: "rgba(56, 189, 248, 0.14)",
  },
  incumbent: {
    label: "EXACT INCUMBENT",
    title: "Exact method under time cap — best feasible incumbent",
    color: NOC_COLORS.indigoText,
    bg: "rgba(129, 140, 248, 0.14)",
  },
  metaheuristic: {
    label: "METAHEURISTIC",
    title: "Classical local search / metaheuristic rollout",
    color: NOC_COLORS.amber,
    bg: "rgba(245, 158, 11, 0.14)",
  },
  neural: {
    label: "NEURAL DRL",
    title: "Constructive or hierarchical deep reinforcement learning policy",
    color: NOC_COLORS.emerald,
    bg: "rgba(34, 197, 94, 0.14)",
  },
  simulated: {
    label: "SIMULATED",
    title: "Multi-day discrete-event simulation trajectory",
    color: NOC_COLORS.purpleText,
    bg: "rgba(192, 132, 252, 0.14)",
  },
  estimated: {
    label: "ESTIMATED",
    title: "Stochastic / heuristic approximation",
    color: NOC_COLORS.idle,
    bg: "rgba(148, 163, 184, 0.14)",
  },
};

export function ProvenanceBadge({
  tier,
  className = "",
}: {
  tier: ProvenanceTier;
  className?: string;
}) {
  const meta = PROVENANCE_META[tier];
  return (
    <span
      title={meta.title}
      style={{
        color: meta.color,
        backgroundColor: meta.bg,
        borderColor: `${meta.color}44`,
      }}
      className={`inline-flex items-center px-1.5 py-0.5 rounded border text-[10px] font-semibold tracking-wider uppercase tabular-nums ${className}`}
    >
      {meta.label}
    </span>
  );
}

export interface MeterProps {
  label: string;
  value: number;
  max?: number;
  unit?: string;
  warnThreshold?: number;
  critThreshold?: number;
  lowerIsBetter?: boolean;
  provenance?: ProvenanceTier;
}

export function Meter({
  label,
  value,
  max = 100,
  unit = "%",
  warnThreshold = 80,
  critThreshold = 100,
  lowerIsBetter = true,
  provenance,
}: MeterProps) {
  const safeMax = max > 0 ? max : 100;
  const pct = Math.max(0, Math.min(100, (value / safeMax) * 100));
  const isCrit = lowerIsBetter ? value >= critThreshold : value <= critThreshold;
  const isWarn =
    !isCrit && (lowerIsBetter ? value >= warnThreshold : value <= warnThreshold);

  const barColor = isCrit
    ? NOC_COLORS.crit
    : isWarn
    ? NOC_COLORS.warn
    : NOC_COLORS.ok;

  return (
    <div className="flex flex-col gap-1.5 p-2.5 rounded-lg border border-noc-border bg-noc-card/60">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="text-noc-textSecondary font-medium truncate">{label}</span>
        <div className="flex items-center gap-1.5">
          {provenance && <ProvenanceBadge tier={provenance} />}
          <span className="font-mono font-semibold tabular-nums text-noc-textPrimary">
            {value.toLocaleString(undefined, { maximumFractionDigits: 1 })}
            {max !== 100 ? ` / ${max}` : ""}
            {unit ? ` ${unit}` : ""}
          </span>
        </div>
      </div>
      <div
        role="progressbar"
        aria-label={label}
        aria-valuenow={Math.round(value)}
        aria-valuemin={0}
        aria-valuemax={safeMax}
        className="w-full h-2 rounded-full bg-noc-deep overflow-hidden border border-noc-border"
      >
        <div
          style={{ width: `${pct}%`, backgroundColor: barColor }}
          className="h-full rounded-full transition-all duration-200"
        />
      </div>
    </div>
  );
}

export interface ArcRingSpec {
  label: string;
  value: number; // 0..100+
  max?: number;
  color?: string;
  warnAt?: number;
  critAt?: number;
}

export function ConcentricArcGauge({
  rings,
  centerValue,
  centerLabel,
  size = 148,
}: {
  rings: ArcRingSpec[];
  centerValue: string;
  centerLabel: string;
  size?: number;
}) {
  // Strictly cap at 3 rings per scientific fidelity audit
  const boundedRings = rings.slice(0, 3);
  const cx = size / 2;
  const cy = size / 2;
  const strokeWidth = 8;
  const ringGap = 12;
  const outerRadius = size / 2 - strokeWidth - 4;

  return (
    <div className="inline-flex flex-col items-center gap-2">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        role="img"
        aria-label={`${centerLabel}: ${centerValue}`}
      >
        {boundedRings.map((ring, idx) => {
          const r = outerRadius - idx * ringGap;
          const circumference = 2 * Math.PI * r;
          const safeMax = ring.max && ring.max > 0 ? ring.max : 100;
          const ratio = Math.max(0, Math.min(1, ring.value / safeMax));
          const dashOffset = circumference * (1 - ratio);
          const isOverflow = ring.critAt !== undefined && ring.value >= ring.critAt;
          const isWarn =
            !isOverflow && ring.warnAt !== undefined && ring.value >= ring.warnAt;
          const stroke = isOverflow
            ? NOC_COLORS.crit
            : isWarn
            ? NOC_COLORS.warn
            : ring.color ?? [NOC_COLORS.cyan, NOC_COLORS.emerald, NOC_COLORS.indigo][idx % 3];

          return (
            <g key={ring.label}>
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill="none"
                stroke="var(--noc-border)"
                strokeWidth={strokeWidth}
              />
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill="none"
                stroke={stroke}
                strokeWidth={strokeWidth}
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={dashOffset}
                transform={`rotate(-90 ${cx} ${cy})`}
              />
            </g>
          );
        })}
        <text
          x={cx}
          y={cy - 2}
          textAnchor="middle"
          dominantBaseline="central"
          fill="var(--noc-text-primary)"
          className="text-base font-bold tabular-nums"
          style={{ fontVariantNumeric: "tabular-nums" }}
        >
          {centerValue}
        </text>
        <text
          x={cx}
          y={cy + 14}
          textAnchor="middle"
          dominantBaseline="central"
          fill="var(--noc-text-muted)"
          className="text-[10px] uppercase tracking-wider"
        >
          {centerLabel}
        </text>
      </svg>
    </div>
  );
}
