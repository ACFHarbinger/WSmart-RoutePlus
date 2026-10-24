/**
 * Simulation analysis chart library (§H.2) — native ECharts ports of every
 * figure generator in `logic/gen/gen_simulation_analysis.py`.
 *
 * Each builder returns a ChartSpec (option + pixel size) used identically for
 * in-app preview and PNG export.
 */
export { groupSpans } from "../data/simulation";

export {
  type Scale,
  strategyColor,
  regionMarkers,
  buildParetoScatter,
  buildStrategyBubble,
  buildImproverBubble,
} from "./simulationScatter";

export {
  buildPolicyScenarioHeatmap,
  buildScenarioConstructorHeatmap,
  buildImproverDelta,
} from "./simulationHeatmaps";

export {
  buildKpiBar,
  buildKpiCombined,
  buildKmViolin,
} from "./simulationKpi";

export {
  RANK_METRICS,
  averageRanks,
  buildConstructorRanking,
  buildRadar,
  buildHorizonComparison,
  buildHorizonDelta,
  buildHorizonConstructorRanking,
} from "./simulationRanking";
