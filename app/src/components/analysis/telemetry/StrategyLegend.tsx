/**
 * Packing algorithm colour legend (LA · LM-CF70 · …) for §G.1.4 charts.
 */
import { SELECTION_STRATEGY_LEGEND, selectionStrategyColor } from "../../../utils/sim/simMetadata";

export function StrategyLegend() {
  return (
    <div className="flex flex-wrap gap-2">
      {SELECTION_STRATEGY_LEGEND.map((strategy) => (
        <span key={strategy} className="flex items-center gap-1 text-[10px] text-canvas-muted">
          <span
            className="inline-block w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: selectionStrategyColor(strategy) }}
          />
          {strategy}
        </span>
      ))}
    </div>
  );
}
