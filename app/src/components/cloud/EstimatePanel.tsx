import { Clock, Coins, Cpu, Info } from "lucide-react";
import type { CloudJobEstimate } from "../../types/cloud";

export interface EstimatePanelProps {
  /**
   * Typed estimate prop.
   * If null, undefined, or missing values, renders "estimate unavailable"
   * (no runtime prediction or no price for the chosen machine).
   */
  estimate?: CloudJobEstimate | null;
  className?: string;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  if (mins < 60) return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
  const hours = Math.floor(mins / 60);
  const remMins = mins % 60;
  return remMins > 0 ? `${hours}h ${remMins}m` : `${hours}h`;
}

export function EstimatePanel({ estimate, className = "" }: EstimatePanelProps) {
  const hasEstimateData =
    estimate != null &&
    (estimate.cost_usd != null ||
      estimate.eta_seconds != null ||
      estimate.memory_required_gb != null);

  if (!hasEstimateData) {
    return (
      <div
        data-testid="estimate-panel-unavailable"
        className={`card bg-canvas-elevated/40 border border-canvas-border rounded-xl p-4 ${className}`}
      >
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-canvas-surface border border-canvas-border text-canvas-muted">
            <Info size={18} />
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-semibold text-gray-200">
                Resource & Cost Estimation
              </h4>
              <span className="px-2 py-0.5 text-[11px] font-medium bg-canvas-surface text-canvas-muted rounded-full border border-canvas-border">
                Estimate unavailable
              </span>
            </div>
            <p className="text-xs text-canvas-muted leading-relaxed">
              Estimate unavailable: the runtime predictor could not read the cells (use
              sim.graph.* arguments or a rerun manifest) or no price is known for this machine.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      data-testid="estimate-panel-available"
      className={`card bg-canvas-elevated/60 border border-canvas-border rounded-xl p-4 space-y-3 ${className}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Coins size={16} className="text-accent-primary" />
          <h4 className="text-sm font-semibold text-gray-200">
            Resource & Cost Estimation
          </h4>
        </div>
        {estimate.source && (
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-accent-primary/10 text-accent-primary border border-accent-primary/20">
            {estimate.source}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="bg-canvas-surface/80 rounded-lg p-3 border border-canvas-border flex flex-col gap-0.5">
          <div className="flex items-center gap-1.5 text-xs text-canvas-muted">
            <Coins size={13} />
            <span>Estimated Cost</span>
          </div>
          <span className="text-lg font-bold font-mono text-gray-100">
            {estimate.cost_usd != null
              ? `$${estimate.cost_usd.toFixed(2)}`
              : "Unavailable"}
          </span>
        </div>

        <div className="bg-canvas-surface/80 rounded-lg p-3 border border-canvas-border flex flex-col gap-0.5">
          <div className="flex items-center gap-1.5 text-xs text-canvas-muted">
            <Clock size={13} />
            <span>Estimated Runtime (ETA)</span>
          </div>
          <span className="text-lg font-bold font-mono text-gray-100">
            {estimate.eta_seconds != null
              ? formatDuration(estimate.eta_seconds)
              : "Unavailable"}
          </span>
        </div>

        <div className="bg-canvas-surface/80 rounded-lg p-3 border border-canvas-border flex flex-col gap-0.5">
          <div className="flex items-center gap-1.5 text-xs text-canvas-muted">
            <Cpu size={13} />
            <span>Memory Peak</span>
          </div>
          <span className="text-lg font-bold font-mono text-gray-100">
            {estimate.memory_required_gb != null
              ? `${estimate.memory_required_gb} GB`
              : "Unavailable"}
          </span>
        </div>
      </div>
    </div>
  );
}
