import type { RadarPolicyCohortEntry } from "./OrionRadarChart";
import type { ConstellationNode, ConstellationLink } from "./OrionConstellationGraph";
import type { OrbitalBubble } from "./OrionOrbitalBubbleChart";
import type { HoneycombZone, ZoneTrajectory } from "./OrionHoneycombMap";
import type { RadialRingSpec } from "./OrionRadialRingMeter";
import type { BinFillMatrixCell } from "./OrionDotMatrixHeatmap";

export { OrionRadarChart } from "./OrionRadarChart";
export type { OrionRadarChartProps, RadarPolicyCohortEntry } from "./OrionRadarChart";

export { OrionConstellationGraph } from "./OrionConstellationGraph";
export type {
  OrionConstellationGraphProps,
  ConstellationNode,
  ConstellationLink,
} from "./OrionConstellationGraph";

export { OrionOrbitalBubbleChart } from "./OrionOrbitalBubbleChart";
export type {
  OrionOrbitalBubbleChartProps,
  OrbitalBubble,
} from "./OrionOrbitalBubbleChart";

export { OrionHoneycombMap } from "./OrionHoneycombMap";
export type {
  OrionHoneycombMapProps,
  HoneycombZone,
  ZoneTrajectory,
} from "./OrionHoneycombMap";

export { OrionHoneycombMatrix } from "./OrionHoneycombMatrix";
export type {
  OrionHoneycombMatrixProps,
  HoneycombCellData,
} from "./OrionHoneycombMatrix";

export { OrionDotMatrixHeatmap } from "./OrionDotMatrixHeatmap";
export type {
  OrionDotMatrixHeatmapProps,
  BinFillMatrixCell,
} from "./OrionDotMatrixHeatmap";

export { OrionRadialRingMeter } from "./OrionRadialRingMeter";
export type {
  OrionRadialRingMeterProps,
  RadialRingSpec,
} from "./OrionRadialRingMeter";

export { OrionGlowDefs } from "./OrionGlowDefs";

/**
 * Returns realistic WSmart-Route Operations Research & DRL telemetry fixtures
 * for all 6 signature SetProduct Orion visualization components.
 */
export function getSampleRoutingOrionData() {
  const radarCohort: RadarPolicyCohortEntry[] = [
    {
      policy: "gurobi",
      means: {
        profit: 1850.5,
        km: 72.4,
        kg: 2840,
        "kg/km": 39.2,
        overflows: 0,
        cost: 165.2,
      },
    },
    {
      policy: "hgs",
      means: {
        profit: 1780.2,
        km: 75.8,
        kg: 2790,
        "kg/km": 36.8,
        overflows: 1,
        cost: 180.4,
      },
    },
    {
      policy: "alns",
      means: {
        profit: 1710.0,
        km: 79.1,
        kg: 2650,
        "kg/km": 33.5,
        overflows: 2,
        cost: 195.0,
      },
    },
    {
      policy: "am",
      means: {
        profit: 1640.8,
        km: 84.5,
        kg: 2580,
        "kg/km": 30.5,
        overflows: 3,
        cost: 215.6,
      },
    },
    {
      policy: "tam",
      means: {
        profit: 1690.4,
        km: 81.2,
        kg: 2620,
        "kg/km": 32.3,
        overflows: 2,
        cost: 202.1,
      },
    },
    {
      policy: "ddam",
      means: {
        profit: 1735.6,
        km: 78.0,
        kg: 2710,
        "kg/km": 34.7,
        overflows: 1,
        cost: 188.5,
      },
    },
    {
      policy: "metarnn",
      means: {
        profit: 1675.0,
        km: 82.6,
        kg: 2600,
        "kg/km": 31.5,
        overflows: 2,
        cost: 208.3,
      },
    },
  ];

  const constellationNodes: ConstellationNode[] = [
    {
      id: "hub-manager",
      label: "Manager HRL",
      tier: 0,
      group: "Hierarchical Core",
      status: "active",
      tone: "cyan",
      details: "Top-level MDP manager dispatching regional cluster sub-goals",
      metrics: { "Horizon (Days)": 31, "Clusters": 6, "Val Reward": "1,842.1" },
    },
    {
      id: "worker-vrpp",
      label: "Worker VRPP",
      tier: 1,
      group: "Constructive Neural",
      status: "nominal",
      tone: "indigo",
      details: "DDAM Attention Worker routing profitable node selections",
      metrics: { "Solve Speed": "48 inst/s", "Gap vs Exact": "2.4%" },
    },
    {
      id: "worker-wcvrp",
      label: "Worker WCVRP",
      tier: 1,
      group: "Dynamic Waste",
      status: "active",
      tone: "indigo",
      details: "Capacitated dynamic waste tour generator with split deliveries",
      metrics: { "Fleet Fill": "91.5%", "Load": "4,200 kg" },
    },
    {
      id: "worker-exact",
      label: "Subproblem BPC",
      tier: 1,
      group: "Operations Research",
      status: "nominal",
      tone: "ok",
      details: "Barnhart-Hane-Vance branch-and-price-and-cut exact bound",
      metrics: { "MIP Gap": "0.0%", "Lagrangian z_UB": "1,902.4" },
    },
    {
      id: "leaf-am",
      label: "AM Head #1",
      tier: 2,
      group: "Attention",
      status: "nominal",
      tone: "purple",
      details: "Multi-head attention query-key projection layer 1",
      metrics: { "Heads": 8, "Dim": 128 },
    },
    {
      id: "leaf-tam",
      label: "TAM Temporal",
      tier: 2,
      group: "Temporal Context",
      status: "nominal",
      tone: "purple",
      details: "Temporal attention RNN capturing fill-rate stochastics",
      metrics: { "State Dim": 64 },
    },
    {
      id: "leaf-alns",
      label: "ALNS Refine",
      tier: 2,
      group: "Metaheuristic",
      status: "nominal",
      tone: "amber",
      details: "Adaptive large neighborhood search local tour repair",
      metrics: { "Destroy Ops": 4, "Repair Ops": 4 },
    },
  ];

  const constellationLinks: ConstellationLink[] = [
    { source: "hub-manager", target: "worker-vrpp", label: "dispatch", active: true, tone: "cyan" },
    { source: "hub-manager", target: "worker-wcvrp", label: "dispatch", active: true, tone: "cyan" },
    { source: "hub-manager", target: "worker-exact", label: "bound check", active: false, tone: "ok" },
    { source: "worker-vrpp", target: "leaf-am", label: "attend", active: true, tone: "indigo" },
    { source: "worker-vrpp", target: "leaf-tam", label: "sequence", active: true, tone: "indigo" },
    { source: "worker-wcvrp", target: "leaf-alns", label: "repair", active: false, tone: "amber" },
  ];

  const orbitalBubbles: OrbitalBubble[] = [
    {
      id: "trial-104",
      label: "Trial #104 (BPC)",
      orbitIdx: 0,
      angleDeg: 35,
      radius: 26,
      value: 1890,
      unit: "EUR",
      tone: "ok",
      metrics: { Profit: "€1,890.50", "Collection Eff": "39.4 kg/km", "Gap": "0.0%" },
      trialMeta: { LR: "Exact (BPC)", TimeCap: "300s", Cuts: "Edge Clique" },
    },
    {
      id: "trial-088",
      label: "Trial #88 (DDAM)",
      orbitIdx: 0,
      angleDeg: 140,
      radius: 24,
      value: 1740,
      unit: "EUR",
      tone: "cyan",
      metrics: { Profit: "€1,740.20", "Solve Speed": "52 inst/s" },
      trialMeta: { LR: "2.5e-4", Layers: 6, Heads: 8, EmbedDim: 128 },
    },
    {
      id: "trial-062",
      label: "Trial #62 (TAM)",
      orbitIdx: 1,
      angleDeg: 210,
      radius: 20,
      value: 1695,
      unit: "EUR",
      tone: "indigo",
      metrics: { Profit: "€1,695.10", "Overflows": 1 },
      trialMeta: { LR: "5e-4", HiddenDim: 64, RewardScale: "Normalized" },
    },
    {
      id: "trial-041",
      label: "Trial #41 (HGS)",
      orbitIdx: 1,
      angleDeg: 310,
      radius: 22,
      value: 1775,
      unit: "EUR",
      tone: "amber",
      metrics: { Profit: "€1,775.00", "Distance": "76.2 km" },
      trialMeta: { PopSize: 40, MutRate: 0.15, EliteRatio: 0.25 },
    },
    {
      id: "trial-012",
      label: "Trial #12 (AM Explore)",
      orbitIdx: 2,
      angleDeg: 95,
      radius: 16,
      value: 1580,
      unit: "EUR",
      tone: "purple",
      metrics: { Profit: "€1,580.40", "Exploration": "High" },
      trialMeta: { LR: "1e-3", EntropyCoeff: 0.05 },
    },
  ];

  const honeycombZones: HoneycombZone[] = [
    { id: "z-depot", row: 1, col: 1, label: "DEPOT-01", fillRatio: 0.15, binCount: 12, overflowRisk: false, status: "ok", depot: true },
    { id: "z-centro", row: 0, col: 2, label: "Centro", fillRatio: 0.88, binCount: 42, overflowRisk: true, status: "warn" },
    { id: "z-norte", row: 0, col: 0, label: "Norte", fillRatio: 0.52, binCount: 28, overflowRisk: false, status: "ok" },
    { id: "z-sul", row: 2, col: 0, label: "Sul", fillRatio: 1.12, binCount: 35, overflowRisk: true, status: "crit" },
    { id: "z-oriente", row: 1, col: 3, label: "Oriente", fillRatio: 0.74, binCount: 31, overflowRisk: false, status: "active" },
    { id: "z-porto", row: 2, col: 2, label: "Porto", fillRatio: 0.38, binCount: 19, overflowRisk: false, status: "idle" },
    { id: "z-oeste", row: 1, col: 0, label: "Oeste", fillRatio: 0.65, binCount: 24, overflowRisk: false, status: "ok" },
  ];

  const zoneTrajectories: ZoneTrajectory[] = [
    { id: "traj-1", fromZoneId: "z-depot", toZoneId: "z-sul", vehicleId: "Truck-01", loadKg: 2800, active: true, tone: "crit" },
    { id: "traj-2", fromZoneId: "z-depot", toZoneId: "z-centro", vehicleId: "Truck-02", loadKg: 2100, active: true, tone: "ok" },
    { id: "traj-3", fromZoneId: "z-centro", toZoneId: "z-oriente", vehicleId: "Truck-02", loadKg: 3400, active: false, tone: "cyan" },
  ];

  const radialRings: RadialRingSpec[] = [
    { id: "fleet-cap", label: "Fleet Capacity", value: 87.5, max: 100, unit: "%", tone: "cyan", warnAt: 80, critAt: 95 },
    { id: "gpu-load", label: "GPU VRAM Load", value: 10.4, max: 12.0, unit: "GB", tone: "indigo", warnAt: 10.5, critAt: 11.5 },
    { id: "day-progress", label: "Route Completion", value: 24, max: 31, unit: "Days", tone: "ok" },
  ];

  const dotMatrixCells: BinFillMatrixCell[] = [];
  for (let d = 0; d < 14; d++) {
    for (let b = 0; b < 10; b++) {
      const baseFill = 0.3 + (d * 0.05) + ((b % 3) * 0.15);
      const isOverflow = (d === 6 && b === 2) || (d === 11 && b === 7);
      dotMatrixCells.push({
        dayIdx: d,
        binIdx: b,
        fillRatio: isOverflow ? 1.15 : Math.min(0.95, baseFill),
      });
    }
  }

  return {
    radarCohort,
    constellationNodes,
    constellationLinks,
    orbitalBubbles,
    honeycombZones,
    zoneTrajectories,
    radialRings,
    dotMatrixCells,
    dayLabels: Array.from({ length: 14 }, (_, i) => `Day ${i + 1}`),
    binLabels: Array.from({ length: 10 }, (_, i) => `Bin #${i + 1}`),
  };
}
