import { formatObserved } from "../../utils/cloud/analysis";
/**
 * Prediction Accuracy Card — displays actual vs predicted task runtime metrics,
 * model calibration error, and variance for CR-11 / CR-16 (§J.1).
 */

import { Clock, Gauge, AlertCircle, CheckCircle2 } from "lucide-react";
import type { CloudJobAnalysis } from "../../types/cloud";

interface PredictionAccuracyCardProps {
  analysis: CloudJobAnalysis;
}

export function PredictionAccuracyCard({ analysis }: PredictionAccuracyCardProps) {
  const tasksWithPred = analysis.tasks.filter(
    (t) => t.expected_wall_seconds != null && t.expected_wall_seconds > 0 && t.duration_seconds != null && t.duration_seconds > 0
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
    <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-5">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-neutral-800">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded bg-indigo-500/10 text-indigo-400">
            <Gauge size={18} />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-neutral-200">Runtime Prediction Accuracy (CR-11)</h3>
            <p className="text-xs text-neutral-500">
              Evaluates execution time vs prior ML surrogate / heuristic predictions
            </p>
          </div>
        </div>

        <div>
          {hasPredictionData ? (
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <CheckCircle2 size={12} />
              Calibrated ({tasksWithPred.length}/{analysis.tasks.length} tasks)
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium bg-neutral-800 text-neutral-400 border border-neutral-700">
              <AlertCircle size={12} />
              Pending CR-11 Calibration
            </span>
          )}
        </div>
      </div>

      {!hasPredictionData ? (
        <div className="bg-neutral-950/40 border border-neutral-800/60 rounded-lg p-6 text-center">
          <Clock className="w-8 h-8 text-neutral-600 mx-auto mb-2" />
          <h4 className="text-sm font-medium text-neutral-300">No Pre-Run Runtime Predictions Available</h4>
          <p className="text-xs text-neutral-500 max-w-md mx-auto mt-1">
            Tasks in this job manifest were executed without pre-computed runtime surrogates (`expected_wall_seconds`).
            Once CR-11 model calibration is enabled during manifest generation, actual vs predicted runtimes will display here.
          </p>
          <div className="mt-4 flex items-center justify-center gap-4 text-xs font-mono text-neutral-400">
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
          <div className="bg-neutral-950/60 border border-neutral-800/80 rounded p-3">
            <span className="text-xs text-neutral-400">Model Accuracy</span>
            <div className="text-xl font-bold font-mono text-neutral-100 mt-1">
              {accuracyPct.toFixed(1)}%
            </div>
            <span className="text-[11px] text-neutral-500">
              Mean absolute error: {overallErrorPct.toFixed(1)}%
            </span>
          </div>

          {/* Actual vs Predicted Total */}
          <div className="bg-neutral-950/60 border border-neutral-800/80 rounded p-3">
            <span className="text-xs text-neutral-400">Observed vs Predicted</span>
            <div className="text-xl font-bold font-mono text-neutral-200 mt-1">
              {totalActualSec.toFixed(1)}s / {totalPredSec.toFixed(1)}s
            </div>
            <span className="text-[11px] text-neutral-500">
              Ratio: <strong className="text-neutral-300">{overallRatio.toFixed(2)}x</strong>
            </span>
          </div>

          {/* Within ±15% Tolerance */}
          <div className="bg-neutral-950/60 border border-neutral-800/80 rounded p-3">
            <span className="text-xs text-neutral-400">Within ±15% Tolerance</span>
            <div className="text-xl font-bold font-mono text-emerald-400 mt-1">
              {withinToleranceCount} / {tasksWithPred.length}
            </div>
            <span className="text-[11px] text-emerald-500/90 font-medium">
              {((withinToleranceCount / tasksWithPred.length) * 100).toFixed(0)}% within margin
            </span>
          </div>

          {/* Over / Under Bias */}
          <div className="bg-neutral-950/60 border border-neutral-800/80 rounded p-3">
            <span className="text-xs text-neutral-400">Prediction Bias</span>
            <div className="text-sm font-mono text-neutral-200 mt-2 flex items-center gap-2">
              <span className="text-amber-400">{overEstimateCount} over</span>
              <span>/</span>
              <span className="text-blue-400">{underEstimateCount} under</span>
            </div>
            <span className="text-[11px] text-neutral-500">
              {overEstimateCount >= underEstimateCount ? "Conservative estimates" : "Optimistic estimates"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
