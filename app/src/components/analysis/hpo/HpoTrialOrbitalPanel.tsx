import React, { useMemo } from "react";
import { OrionOrbitalBubbleChart, type OrbitalBubble } from "../../ui/orion/OrionOrbitalBubbleChart";
import type { OptunaStudyData } from "../../../types";

export interface HpoTrialOrbitalPanelProps {
  studyData: OptunaStudyData;
  height?: number;
}

export const HpoTrialOrbitalPanel: React.FC<HpoTrialOrbitalPanelProps> = ({
  studyData,
  height = 360,
}) => {
  const bubbles: OrbitalBubble[] = useMemo(() => {
    const trials = studyData.trials.filter(
      (t) => t.state === "COMPLETE" && t.value != null && Number.isFinite(t.value)
    );
    if (trials.length === 0) return [];

    const values = trials.map((t) => t.value as number);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const span = maxVal - minVal || 1;

    // Sort by value to assign orbits: top 20% -> orbit 0, middle 40% -> orbit 1, rest -> orbit 2
    const sorted = [...trials].sort((a, b) => (b.value ?? 0) - (a.value ?? 0));

    return sorted.slice(0, 18).map((t, idx) => {
      const norm = ((t.value as number) - minVal) / span;
      const orbitIdx = idx < 4 ? 0 : idx < 10 ? 1 : 2;
      const angleDeg = (idx * (360 / Math.min(18, sorted.length))) % 360;
      const radius = Math.max(12, Math.min(28, 12 + norm * 16));

      const paramEntries = Object.entries(t.params).slice(0, 4);
      const trialMeta: Record<string, string | number> = {};
      paramEntries.forEach(([k, v]) => {
        trialMeta[k] = typeof v === "number" ? Number(v.toFixed(4)) : String(v);
      });

      return {
        id: `trial-${t.number}`,
        label: `Trial #${t.number}`,
        orbitIdx,
        angleDeg,
        radius,
        value: Number((t.value as number).toFixed(2)),
        unit: "Reward",
        tone: orbitIdx === 0 ? "ok" : orbitIdx === 1 ? "cyan" : "indigo",
        metrics: {
          Objective: Number((t.value as number).toFixed(4)),
          State: t.state,
          Params: Object.keys(t.params).length,
        },
        trialMeta,
      };
    });
  }, [studyData]);

  if (bubbles.length === 0) return null;

  return (
    <OrionOrbitalBubbleChart
      bubbles={bubbles}
      title="Hyperparameter Trial Orbital Cluster"
      subtitle="Concentric trial distribution: Inner orbit represents top objective performance"
      centerLabel="Best Value"
      centerValue={studyData.best_value != null ? studyData.best_value.toFixed(2) : "Opt"}
      height={height}
    />
  );
};
