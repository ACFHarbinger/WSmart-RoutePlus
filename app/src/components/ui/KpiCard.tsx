import { TrendingDown, TrendingUp } from "lucide-react";
import { ProvenanceBadge, type ProvenanceTier } from "./telemetry/gauges";
import { NOC_COLORS, type NocTone } from "./telemetry/tokens";

export interface KpiCardProps {
  label: string;
  value: number | string | null | undefined;
  unit?: string;
  delta?: number | null;
  deltaLabel?: string;
  colorize?: boolean; // true = green when value is good (low for overflows, high for profit)
  lowerIsBetter?: boolean;
  tone?: NocTone;
  provenance?: ProvenanceTier;
  footer?: string;
}

const TONE_BORDER: Record<NocTone, string> = {
  ok: NOC_COLORS.ok,
  warn: NOC_COLORS.warn,
  crit: NOC_COLORS.crit,
  idle: NOC_COLORS.idle,
  cyan: NOC_COLORS.cyan,
  indigo: NOC_COLORS.indigo,
  purple: NOC_COLORS.purple,
  amber: NOC_COLORS.amber,
};

export function KpiCard({
  label,
  value,
  unit,
  delta,
  deltaLabel,
  colorize,
  lowerIsBetter,
  tone,
  provenance,
  footer,
}: KpiCardProps) {
  const displayValue =
    value === null || value === undefined
      ? "—"
      : typeof value === "number"
      ? value.toLocaleString(undefined, { maximumFractionDigits: 2 })
      : value;

  const hasDelta = delta !== null && delta !== undefined;
  const isPositive = delta !== null && delta !== undefined && delta >= 0;
  const isGood = lowerIsBetter ? !isPositive : isPositive;
  const topBorderColor = tone ? TONE_BORDER[tone] : undefined;

  return (
    <div
      style={
        topBorderColor ? { borderTop: `2px solid ${topBorderColor}` } : undefined
      }
      className="kpi-card"
    >
      <div className="flex items-center justify-between gap-1.5">
        <span className="kpi-label truncate">{label}</span>
        {provenance && <ProvenanceBadge tier={provenance} />}
      </div>
      <span className="kpi-value tabular-nums">
        {displayValue}
        {unit && (
          <span className="text-sm font-normal text-canvas-muted ml-1 tabular-nums">
            {unit}
          </span>
        )}
      </span>
      {hasDelta && colorize && (
        <span className={isGood ? "kpi-delta-pos" : "kpi-delta-neg"}>
          <span className="inline-flex items-center gap-0.5 tabular-nums">
            {isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
            {delta! > 0 ? "+" : ""}
            {delta!.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            {deltaLabel && (
              <span className="text-xs text-canvas-muted ml-1">{deltaLabel}</span>
            )}
          </span>
        </span>
      )}
      {footer && (
        <span className="text-[11px] text-canvas-muted truncate mt-0.5 tabular-nums">
          {footer}
        </span>
      )}
    </div>
  );
}
