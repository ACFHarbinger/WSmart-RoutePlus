import { formatObserved } from "../../utils/cloud/analysis";
/**
 * Prediction Accuracy Card — displays actual vs predicted task runtime metrics,
 * model calibration error, and variance for CR-11 / CR-16 (§J.1, Slice V5).
 */

import { Clock, Gauge, AlertCircle, CheckCircle2 } from "lucide-react";
import type { CloudJobAnalysis } from "../../types/cloud";
import { ProvenanceBadge } from "../ui/telemetry/gauges";

interface PredictionAccuracyCardProps {
  analysis: CloudJobAnalysis;
}

export function PredictionAccuracyCard({ analysis }: PredictionAccuracyCardProps) {
  const tasksWithPred = analysis.tasks.filter(
    (t) => t.state === "succeeded" && t.attempt === 1 && t.expected_wall_seconds != null && t.expected_wall_seconds > 0 && t.duration_seconds != null && t.duration_seconds > 0
  );

  const hasPredictionData = tasksWithPred.length > 0;

  let totalActualSec = 0;
  let totalPredSec = 0;
  let withinToleranceCount = 0;
  let overEstimateCount = 0;
  let underEstimateCount = 0;
  let absolutePercentageError = 0;

  if (hasPredictionData) {
    for (const t of tasksWithPred) {
      const actual = t.duration_seconds!;
      const pred = t.expected_wall_seconds!;
      totalActualSec += actual;
      totalPredSec += pred;

      absolutePercentageError += Math.abs(actual - pred) / actual;
      const relDiff = Math.abs(actual - pred) / pred;
      if (relDiff <= 0.15) {
        withinToleranceCount++;
      }
      if (pred > actual) {
        overEstimateCount++;
      } else if (pred < actual) {
        underEstimateCount++;
      }
    }
  }

  const overallRatio = totalPredSec > 0 ? totalActualSec / totalPredSec : 1.0;
  const overallErrorPct = tasksWithPred.length ? absolutePercentageError / tasksWithPred.length * 100 : 0;
  const accuracyPct = Math.max(0, 100 - overallErrorPct);

  return (
    <div className="bg-noc-panel border border-noc-border rounded-xl p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-noc-border">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-noc-indigo/15 text-noc-indigo border border-noc-indigo/30">
            <Gauge size={18} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-noc-textPrimary">Runtime Prediction Accuracy (CR-11)</h3>
              <ProvenanceBadge tier="neural" />
            </div>
            <p className="text-xs text-noc-textMuted mt-0.5">
              Evaluates execution time vs prior ML surrogate / heuristic predictions
            </p>
          </div>
        </div>

        <div>
          {hasPredictionData ? (
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium bg-noc-indigo/15 text-noc-indigo border border-noc-indigo/30">
              <CheckCircle2 size={12} />
              Calibrated ({tasksWithPred.length}/{analysis.tasks.length} tasks)
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium bg-noc-deep text-noc-textMuted border border-noc-border">
              <AlertCircle size={12} />
              Pending CR-11 Calibration
            </span>
          )}
        </div>
      </div>

      {!hasPredictionData ? (
        <div className="bg-noc-deep border border-noc-border rounded-lg p-6 text-center">
          <Clock className="w-8 h-8 text-noc-textMuted mx-auto mb-2 opacity-40" />
          <h4 className="text-sm font-medium text-noc-textPrimary">No Pre-Run Runtime Predictions Available</h4>
          <p className="text-xs text-noc-textMuted max-w-md mx-auto mt-1">
            Tasks in this job manifest were executed without pre-computed runtime surrogates (`expected_wall_seconds`).
            Once CR-11 model calibration is enabled during manifest generation, actual vs predicted runtimes will display here.
          </p>
          <div className="mt-4 flex items-center justify-center gap-4 text-xs font-mono text-noc-textSecondary">
            <span>Observed Compute: <strong>{formatObserved(analysis.total_compute_seconds, 1)}s</strong></span>
            <span>•</span>
            <span>Observed Wall Time: <strong>{formatObserved(analysis.total_wall_seconds, 1)}s</strong></span>
            <span>•</span>
            <span>Total Tasks: <strong>{analysis.total_tasks}</strong></span>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Overall Accuracy */}
          <div className="bg-noc-deep border border-noc-border rounded-lg p-3 space-y-1">
            <span className="text-xs text-noc-textMuted font-medium">Model Accuracy</span>
            <div className="text-xl font-bold font-mono text-noc-textPrimary tabular-nums">
              {accuracyPct.toFixed(1)}%
            </div>
            <span className="text-[11px] text-noc-textMuted tabular-nums">
              Mean absolute error: {overallErrorPct.toFixed(1)}%
            </span>
          </div>

          {/* Actual vs Predicted Total */}
          <div className="bg-noc-deep border border-noc-border rounded-lg p-3 space-y-1">
            <span className="text-xs text-noc-textMuted font-medium">Observed vs Predicted</span>
            <div className="text-xl font-bold font-mono text-noc-textSecondary tabular-nums">
              {totalActualSec.toFixed(1)}s / {totalPredSec.toFixed(1)}s
            </div>
            <span className="text-[11px] text-noc-textMuted tabular-nums">
              Ratio: <strong className="text-noc-textPrimary">{overallRatio.toFixed(2)}x</strong>
            </span>
          </div>

          {/* Within ±15% Tolerance */}
          <div className="bg-noc-deep border border-noc-border rounded-lg p-3 space-y-1">
            <span className="text-xs text-noc-textMuted font-medium">Within ±15% Tolerance</span>
            <div className="text-xl font-bold font-mono text-noc-ok tabular-nums">
              {withinToleranceCount} / {tasksWithPred.length}
            </div>
            <span className="text-[11px] text-noc-ok font-medium tabular-nums">
              {((withinToleranceCount / tasksWithPred.length) * 100).toFixed(0)}% within margin
            </span>
          </div>

          {/* Over / Under Bias */}
          <div className="bg-noc-deep border border-noc-border rounded-lg p-3 space-y-1">
            <span className="text-xs text-noc-textMuted font-medium">Prediction Bias</span>
            <div className="text-sm font-mono text-noc-textSecondary mt-2 flex items-center gap-2 tabular-nums">
              <span className="text-noc-warn">{overEstimateCount} over</span>
              <span>/</span>
              <span className="text-noc-cyan">{underEstimateCount} under</span>
            </div>
            <span className="text-[11px] text-noc-textMuted">
              {overEstimateCount >= underEstimateCount ? "Conservative estimates" : "Optimistic estimates"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
