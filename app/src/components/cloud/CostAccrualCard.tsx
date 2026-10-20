import { formatObserved } from "../../utils/cloud/analysis";
/**
 * Cost Accrual Card — displays financial telemetry, estimated budget vs accrued cost,
 * and spot cost reduction breakdown for CR-10 / CR-16 (§J.1).
 */

import { DollarSign, TrendingDown, Cpu } from "lucide-react";
import type { CloudJobAnalysis } from "../../types/cloud";

interface CostAccrualCardProps {
  analysis: CloudJobAnalysis;
}

export function CostAccrualCard({ analysis }: CostAccrualCardProps) {
  const hasEstimate = analysis.estimated_cost_usd != null && analysis.estimated_cost_usd >= 0;
  const hasSpotSavings = analysis.spot_savings_usd != null && analysis.spot_savings_usd > 0;

  return (
    <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-5">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-neutral-800">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded bg-amber-500/10 text-amber-400">
            <DollarSign size={18} />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-neutral-200">Cost & Budget Accrual (CR-10)</h3>
            <p className="text-xs text-neutral-500">
              Provider: <span className="uppercase text-neutral-300 font-mono">{analysis.provider}</span> •{" "}
              Compute: <span className="text-neutral-300 font-mono">{analysis.compute}</span> •{" "}
              Tier: <span className="text-neutral-300 font-mono">{analysis.spot == null ? "Unavailable" : analysis.spot ? "Spot / Preemptible" : "On-Demand"}</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {analysis.spot && (
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <TrendingDown size={12} />
              Spot Discounted
            </span>
          )}
          {analysis.machine_type && (
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-mono bg-neutral-800 text-neutral-300 border border-neutral-700">
              <Cpu size={12} />
              {analysis.machine_type}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Accrued Cost */}
        <div className="bg-neutral-950/60 border border-neutral-800/80 rounded p-3">
          <span className="text-xs text-neutral-400">Total Accrued Cost</span>
          <div className="text-xl font-bold font-mono text-neutral-100 mt-1">
            ${formatObserved(analysis.accrued_cost_usd, 4)}
          </div>
          <span className="text-[11px] text-neutral-500">
            Across {formatObserved(analysis.total_compute_seconds, 1)}s compute time
          </span>
        </div>

        {/* Estimated / Budget Cap */}
        <div className="bg-neutral-950/60 border border-neutral-800/80 rounded p-3">
          <span className="text-xs text-neutral-400">Pre-Run Estimate (CR-10)</span>
          {hasEstimate ? (
            <>
              <div className="text-xl font-bold font-mono text-neutral-200 mt-1">
                ${analysis.estimated_cost_usd?.toFixed(4)}
              </div>
              <span className="text-[11px] text-neutral-500">
                {(analysis.accrued_cost_usd ?? Infinity) <= (analysis.estimated_cost_usd ?? 0) ? (
                  <span className="text-emerald-400">Within estimate</span>
                ) : (
                  <span className="text-amber-400">Over initial estimate</span>
                )}
              </span>
            </>
          ) : (
            <>
              <div className="text-sm font-mono text-neutral-500 mt-2">Unbounded / No Cap</div>
              <span className="text-[11px] text-neutral-600">Pending CR-10 model calibration</span>
            </>
          )}
        </div>

        {/* Spot Savings */}
        <div className="bg-neutral-950/60 border border-neutral-800/80 rounded p-3">
          <span className="text-xs text-neutral-400">Spot Savings</span>
          {hasSpotSavings ? (
            <>
              <div className="text-xl font-bold font-mono text-emerald-400 mt-1">
                -${analysis.spot_savings_usd?.toFixed(4)}
              </div>
              <span className="text-[11px] text-emerald-500/90 font-medium">
                {analysis.spot_savings_pct?.toFixed(1)}% savings vs on-demand
              </span>
            </>
          ) : (
            <>
              <div className="text-sm font-mono text-neutral-500 mt-2">
                {analysis.spot_savings_usd == null ? "Unavailable" : "$0.0000"}
              </div>
              <span className="text-[11px] text-neutral-600">
                {analysis.spot == null ? "Pricing unavailable" : analysis.spot ? "Spot rate applied" : "Running on standard on-demand"}
              </span>
            </>
          )}
        </div>

        {/* Compute Hours */}
        <div className="bg-neutral-950/60 border border-neutral-800/80 rounded p-3">
          <span className="text-xs text-neutral-400">Aggregate CPU Hours</span>
          <div className="text-xl font-bold font-mono text-neutral-100 mt-1">
            {formatObserved(analysis.total_compute_seconds == null ? null : analysis.total_compute_seconds / 3600, 3)} hrs
          </div>
          <span className="text-[11px] text-neutral-500">
            {analysis.total_tasks} parallel task allocations
          </span>
        </div>
      </div>
    </div>
  );
}
