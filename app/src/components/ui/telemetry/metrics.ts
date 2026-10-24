/**
 * Shared Operations Research Metric Contract (Slice V2).
 *
 * Defines canonical units, polarity (`higherIsBetter`), formatting, and
 * direction-aware cohort normalization (`1.0 = best in cohort`) for all
 * eight `SimDayData` KPIs.
 */

export type SimKpiKey =
  | "profit"
  | "km"
  | "kg"
  | "kg/km"
  | "overflows"
  | "cost"
  | "kg_lost"
  | "ncol";

export interface MetricContract {
  key: SimKpiKey;
  label: string;
  shortLabel: string;
  unit: string;
  higherIsBetter: boolean;
  decimals: number;
  directionArrow: "↑" | "↓";
  format: (val: number | null | undefined) => string;
}

function makeFormatter(decimals: number): (val: number | null | undefined) => string {
  return (val) => {
    if (val === null || val === undefined || Number.isNaN(val)) return "—";
    return val.toLocaleString(undefined, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };
}

export const SIM_METRICS: Record<SimKpiKey, MetricContract> = {
  profit: {
    key: "profit",
    label: "Net Profit",
    shortLabel: "Profit",
    unit: "€",
    higherIsBetter: true,
    decimals: 2,
    directionArrow: "↑",
    format: makeFormatter(2),
  },
  km: {
    key: "km",
    label: "Travel Distance",
    shortLabel: "Distance",
    unit: "km",
    higherIsBetter: false,
    decimals: 2,
    directionArrow: "↓",
    format: makeFormatter(2),
  },
  kg: {
    key: "kg",
    label: "Waste Collected",
    shortLabel: "Collected",
    unit: "kg",
    higherIsBetter: true,
    decimals: 1,
    directionArrow: "↑",
    format: makeFormatter(1),
  },
  "kg/km": {
    key: "kg/km",
    label: "Collection Efficiency",
    shortLabel: "Efficiency",
    unit: "kg/km",
    higherIsBetter: true,
    decimals: 2,
    directionArrow: "↑",
    format: makeFormatter(2),
  },
  overflows: {
    key: "overflows",
    label: "Bin Overflows",
    shortLabel: "Overflows",
    unit: "bins",
    higherIsBetter: false,
    decimals: 1,
    directionArrow: "↓",
    format: makeFormatter(1),
  },
  cost: {
    key: "cost",
    label: "Operational Cost",
    shortLabel: "Cost",
    unit: "€",
    higherIsBetter: false,
    decimals: 2,
    directionArrow: "↓",
    format: makeFormatter(2),
  },
  kg_lost: {
    key: "kg_lost",
    label: "Overflow Waste Lost",
    shortLabel: "Waste Lost",
    unit: "kg",
    higherIsBetter: false,
    decimals: 1,
    directionArrow: "↓",
    format: makeFormatter(1),
  },
  ncol: {
    key: "ncol",
    label: "Bins Visited",
    shortLabel: "Collections",
    unit: "stops",
    higherIsBetter: true,
    decimals: 0,
    directionArrow: "↑",
    format: makeFormatter(0),
  },
};

/**
 * Normalizes a vector of raw metric values across a cohort of policies onto `[0.15, 1.0]`
 * where `1.0` strictly represents the best-performing policy in the cohort and `0.15`
 * represents the worst-performing policy (leaving a visible inner polygon core).
 *
 * When all values in the cohort are identical, returns `1.0` for all entries.
 */
export function normalizeCohortMetric(
  values: readonly number[],
  higherIsBetter: boolean,
  minFloor = 0.15
): number[] {
  if (values.length === 0) return [];
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return values.map(() => minFloor);

  const minVal = Math.min(...finite);
  const maxVal = Math.max(...finite);
  const span = maxVal - minVal;

  if (span < 1e-9) {
    return values.map(() => 1.0);
  }

  return values.map((v) => {
    if (!Number.isFinite(v)) return minFloor;
    const unit = higherIsBetter ? (v - minVal) / span : (maxVal - v) / span;
    return minFloor + unit * (1.0 - minFloor);
  });
}
