import { formatObserved } from "../../utils/cloud/analysis";
/**
 * Cost Accrual Card — displays financial telemetry, estimated budget vs accrued cost,
 * and spot cost reduction breakdown for CR-10 / CR-16 (§J.1, Slice V5).
 */

import { DollarSign, TrendingDown, Cpu } from "lucide-react";
import type { CloudJobAnalysis } from "../../types/cloud";
import { ProvenanceBadge } from "../ui/telemetry/gauges";
import { Meter } from "../ui/telemetry/gauges";

interface CostAccrualCardProps {
  analysis: CloudJobAnalysis;
}

export function CostAccrualCard({ analysis }: CostAccrualCardProps) {
  const hasEstimate = analysis.estimated_cost_usd != null && analysis.estimated_cost_usd >= 0;
  const hasSpotSavings = analysis.spot_savings_usd != null && analysis.spot_savings_usd > 0;
  const accrued = analysis.accrued_cost_usd ?? 0;
  const estimate = analysis.estimated_cost_usd ?? 0;

  return (
    <div className="bg-noc-panel border border-noc-border rounded-xl p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-noc-border">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-noc-amber/15 text-noc-amber border border-noc-amber/30">
            <DollarSign size={18} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-noc-textPrimary">Compute Cost Estimates</h3>
              <ProvenanceBadge tier={hasEstimate ? "estimated" : "simulated"} />
            </div>
            <p className="text-xs text-noc-textMuted mt-0.5">
              Provider: <span className="uppercase text-noc-textSecondary font-mono">{analysis.provider}</span> •{" "}
              Compute: <span className="text-noc-textSecondary font-mono">{analysis.compute}</span> •{" "}
              Tier: <span className="text-noc-textSecondary font-mono">{analysis.spot == null ? "Unavailable" : analysis.spot ? "Spot / Preemptible" : "On-Demand"}</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {analysis.spot && (
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium bg-noc-ok/15 text-noc-ok border border-noc-ok/30">
              <TrendingDown size={12} />
              Spot Discounted
            </span>
          )}
          {analysis.machine_type && (
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-mono bg-noc-deep text-noc-textSecondary border border-noc-border">
              <Cpu size={12} />
              {analysis.machine_type}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Estimated Compute Cost */}
        <div className="bg-noc-deep border border-noc-border rounded-lg p-3 space-y-1">
          <span className="text-xs text-noc-textMuted font-medium">Estimated Compute Cost</span>
          <div className="text-xl font-bold font-mono text-noc-textPrimary tabular-nums">
            ${formatObserved(analysis.accrued_cost_usd, 4)}
          </div>
          <p className="text-[11px] text-noc-textMuted">
            Prorated price hint over {formatObserved(analysis.total_compute_seconds, 1)}s scheduler time; not an invoice
          </p>
        </div>

        {/* Estimated / Budget Cap */}
        <div className="bg-noc-deep border border-noc-border rounded-lg p-3 space-y-1">
          <span className="text-xs text-noc-textMuted font-medium">Predicted Cost Estimate</span>
          {hasEstimate ? (
            <>
              <div className="text-xl font-bold font-mono text-noc-textPrimary mt-1 tabular-nums">
                ${analysis.estimated_cost_usd?.toFixed(4)}
              </div>
              <span className="text-[11px]">
                {analysis.accrued_cost_usd == null ? (
                  <span className="text-noc-textMuted">Observed duration unavailable</span>
                ) : analysis.accrued_cost_usd <= (analysis.estimated_cost_usd ?? 0) ? (
                  <span className="text-noc-ok font-medium">Within estimate</span>
                ) : (
                  <span className="text-noc-warn font-medium">Above predicted estimate</span>
                )}
              </span>
            </>
          ) : (
            <>
              <div className="text-sm font-mono text-noc-textMuted mt-2">Estimate unavailable</div>
              <span className="text-[11px] text-noc-textMuted">No runtime predictions available</span>
            </>
          )}
        </div>

        {/* Spot Savings */}
        <div className="bg-noc-deep border border-noc-border rounded-lg p-3 space-y-1">
          <span className="text-xs text-noc-textMuted font-medium">Spot Savings</span>
          {hasSpotSavings ? (
            <>
              <div className="text-xl font-bold font-mono text-noc-ok mt-1 tabular-nums">
                -${analysis.spot_savings_usd?.toFixed(4)}
              </div>
              <span className="text-[11px] text-noc-ok font-medium tabular-nums">
                {analysis.spot_savings_pct?.toFixed(1)}% savings vs on-demand
              </span>
            </>
          ) : (
            <>
              <div className="text-sm font-mono text-noc-textMuted mt-2">
                {analysis.spot_savings_usd == null ? "Unavailable" : "$0.0000"}
              </div>
              <span className="text-[11px] text-noc-textMuted">
                {analysis.spot == null ? "Pricing unavailable" : analysis.spot ? "Spot rate applied" : "Running on standard on-demand"}
              </span>
            </>
          )}
        </div>

        {/* Compute Hours */}
        <div className="bg-noc-deep border border-noc-border rounded-lg p-3 space-y-1">
          <span className="text-xs text-noc-textMuted font-medium">Aggregate CPU Hours</span>
          <div className="text-xl font-bold font-mono text-noc-textPrimary mt-1 tabular-nums">
            {formatObserved(analysis.total_compute_seconds == null ? null : analysis.total_compute_seconds / 3600, 3)} hrs
          </div>
          <span className="text-[11px] text-noc-textMuted tabular-nums">
            {analysis.total_tasks} parallel task allocations
          </span>
        </div>
      </div>

      {hasEstimate && estimate > 0 && (
        <Meter
          label="Budget Accrual vs Hard Cap"
          value={Math.round((accrued / estimate) * 100)}
          max={100}
          unit="%"
          warnThreshold={80}
          critThreshold={100}
          lowerIsBetter
          provenance="estimated"
        />
      )}
    </div>
  );
}
