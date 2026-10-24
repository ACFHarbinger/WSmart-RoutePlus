import React, { useState } from "react";
import {
  KpiCardGrid,
  HudStatusPill,
  StalenessDot,
  MonitorTileStrip,
  EmptyStateChip,
} from "./status";
import { ProvenanceBadge, Meter, ConcentricArcGauge } from "./gauges";
import { SubtabBar } from "./SubtabBar";
import { TelemetryTimeSeriesCard } from "./charts";
import { nocPanelStyle, nocSubsurfaceStyle } from "./panel";
import {
  OrionRadarChart,
  OrionConstellationGraph,
  OrionOrbitalBubbleChart,
  OrionHoneycombMap,
  OrionRadialRingMeter,
  OrionDotMatrixHeatmap,
  getSampleRoutingOrionData,
} from "../orion";

export interface KitGalleryProps {
  className?: string;
}

export const KitGallery: React.FC<KitGalleryProps> = ({ className = "" }) => {
  const [activeTab, setActiveTab] = useState<string>("table");
  const now = Date.now();
  const sample = getSampleRoutingOrionData();

  return (
    <div
      className={`space-y-6 p-4 text-canvas-text ${className}`}
      style={nocPanelStyle}
      data-testid="kit-gallery"
    >
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-canvas-border pb-3">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider">
            NOC Telemetry Kit Gallery
          </h2>
          <p className="text-xs text-canvas-muted">
            WCAG AA Dual-Theme Operations Research Primitives & Orion Suite
          </p>
        </div>
        <div className="flex items-center gap-2">
          <HudStatusPill tone="ok" label="STREAM ONLINE" pulse />
          <HudStatusPill tone="amber" label="STALE CHECK" />
          <HudStatusPill tone="crit" label="OVERFLOW ALERT" />
          <StalenessDot lastEventAt={now - 1000} />
        </div>
      </div>

      <SubtabBar
        tabs={[
          { id: "table", label: "Table", count: 8 },
          { id: "trajectories", label: "Trajectories" },
          { id: "map", label: "Map" },
          { id: "compare", label: "Compare" },
          { id: "orion", label: "Orion Visualizations" },
        ]}
        activeId={activeTab}
        onChange={setActiveTab}
      />

      {activeTab === "orion" ? (
        <div className="space-y-6" data-testid="orion-showcase">
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <OrionRadarChart cohort={sample.radarCohort} height={320} />
            <OrionRadialRingMeter
              rings={sample.radialRings}
              centerValue="92.4%"
              centerLabel="FLEET UTILIZATION"
            />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <OrionConstellationGraph
              nodes={sample.constellationNodes}
              links={sample.constellationLinks}
              height={360}
            />
            <OrionOrbitalBubbleChart
              bubbles={sample.orbitalBubbles}
              height={360}
            />
          </div>

          <OrionHoneycombMap
            zones={sample.honeycombZones}
            trajectories={sample.zoneTrajectories}
            height={360}
          />

          <OrionDotMatrixHeatmap
            dayLabels={sample.dayLabels}
            binLabels={sample.binLabels}
            cells={sample.dotMatrixCells}
            height={300}
          />
        </div>
      ) : (
        <>
          <MonitorTileStrip
            tiles={[
              { id: "day", label: "Simulation Day", value: "30 / 31", tone: "cyan" },
              { id: "profit", label: "Net Profit", value: "€1,420.50", tone: "ok" },
              { id: "overflows", label: "Overflows", value: 2, tone: "crit" },
              { id: "efficiency", label: "Efficiency", value: "14.8 kg/km", tone: "indigo" },
            ]}
          />

          <KpiCardGrid
            columns={4}
            items={[
              {
                label: "Mean Profit",
                value: "1,284.40",
                unit: "EUR",
                tone: "ok",
                provenance: "exact",
                delta: 8.4,
                deltaLabel: "vs regular",
                footer: "95% CI: ±42.10",
              },
              {
                label: "Total Distance",
                value: "86.2",
                unit: "km",
                tone: "cyan",
                provenance: "metaheuristic",
                delta: -4.1,
                deltaLabel: "improved",
              },
              {
                label: "Bin Overflows",
                value: 3,
                tone: "crit",
                provenance: "simulated",
                footer: "Threshold: 0",
              },
              {
                label: "Service Time",
                value: "4.25",
                unit: "h",
                tone: "amber",
                provenance: "neural",
              },
            ]}
          />

          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="p-3 rounded-xl border" style={nocSubsurfaceStyle}>
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-canvas-muted">
                  Concentric Fleet Gauge
                </span>
                <ProvenanceBadge tier="simulated" />
              </div>
              <div className="flex justify-center py-2">
                <ConcentricArcGauge
                  centerValue="82%"
                  centerLabel="FLEET UTIL"
                  rings={[
                    { label: "Truck 1", value: 88, max: 100 },
                    { label: "Truck 2", value: 76, max: 100 },
                    { label: "Truck 3", value: 64, max: 100 },
                  ]}
                />
              </div>
            </div>

            <div className="space-y-3 p-3 rounded-xl border" style={nocSubsurfaceStyle}>
              <span className="text-[11px] font-semibold uppercase tracking-wider text-canvas-muted">
                Capacity Meters
              </span>
              <Meter value={42} max={100} label="Zone North Fill" unit="%" />
              <Meter value={84} max={100} label="Zone Central Fill" unit="%" />
              <Meter value={97} max={100} label="Zone South Fill" unit="%" />
            </div>

            <div className="flex flex-col justify-between p-3 rounded-xl border" style={nocSubsurfaceStyle}>
              <span className="text-[11px] font-semibold uppercase tracking-wider text-canvas-muted">
                Empty State & Diagnostics
              </span>
              <EmptyStateChip
                title="No HPO Sweep Active"
                description="Launch an Optuna study from Training Hub to populate parameter surfaces."
              />
            </div>
          </div>

          <TelemetryTimeSeriesCard
            title="Cumulative Fleet Profit Trajectory"
            provenance="exact"
            unit="EUR"
            height={160}
            points={[
              { x: "D1", y: 120 },
              { x: "D5", y: 340 },
              { x: "D10", y: 610 },
              { x: "D15", y: 890 },
              { x: "D20", y: 1120 },
              { x: "D25", y: 1310 },
              { x: "D30", y: 1490 },
            ]}
          />
        </>
      )}
    </div>
  );
};

export default KitGallery;
