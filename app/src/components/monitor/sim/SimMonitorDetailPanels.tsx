import { KpiCard } from "../../ui/KpiCard";
import { KpiCardGrid } from "../../ui/telemetry/status";
import { OrionRadialRingMeter } from "../../ui/orion/OrionRadialRingMeter";
import { BinFillStrip, computeDelta, MetricTimeseries, PRIMARY_KPIS, SECONDARY_KPIS, TourTable } from "./SimMonitorPanels";
import { FailureAnalysisPanel } from "../../analysis/routes/FailureAnalysisPanel";
import { PolicyTelemetryPanel } from "../../analysis/telemetry/PolicyTelemetryPanel";
import { PolicyTelemetryTrendsPanel } from "../../analysis/telemetry/PolicyTelemetryTrendsPanel";
import { GraphTopologyPanel } from "../../analysis/topology/GraphTopologyPanel";
import { SqlQueryPanel } from "../../analysis/explorer/SqlQueryPanel";
import type { DayLogEntry, PolicyVizEntry, SimFailureEntry } from "../../../types";

export function SimMonitorKpiSection({
  displayEntry,
  displayDay,
  filteredEntries,
  showSecondary,
  setShowSecondary,
  policySeries,
  logScale,
}: {
  displayEntry: DayLogEntry;
  displayDay: number;
  filteredEntries: DayLogEntry[];
  showSecondary: boolean;
  setShowSecondary: (fn: (v: boolean) => boolean) => void;
  policySeries: Array<{ policy: string; entries: DayLogEntry[]; color: string }>;
  logScale: boolean;
}) {
  const data = displayEntry.data as Record<string, number>;
  const totalDays = Math.max(1, filteredEntries.length);
  const dayProgressPct = Math.min(100, Math.round((displayDay / totalDays) * 100));
  const capacityPct = Math.min(100, Math.round(((data.kg ?? 0) / 3000) * 100));
  const overflowVal = data.overflows ?? 0;

  const kpiItems = PRIMARY_KPIS.map(({ key, label, lowerIsBetter }) => ({
    label,
    value: data[key],
    delta: computeDelta(filteredEntries, displayDay, key),
    colorize: true,
    lowerIsBetter,
    provenance: "simulated" as const,
    tone: (key === "overflows" && overflowVal > 0 ? "crit" : key === "profit" ? "ok" : "cyan") as any,
  }));

  return (
    <>
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 items-center">
        <div className="xl:col-span-2">
          <KpiCardGrid items={kpiItems} columns={4} />
        </div>
        <div className="flex justify-center">
          <OrionRadialRingMeter
            title="Real-Time Fleet & Route Rings"
            subtitle={`Day ${displayDay} telemetry telemetry summary`}
            centerValue={`${dayProgressPct}%`}
            centerLabel="DAY PROGRESS"
            size={220}
            rings={[
              {
                id: "progress",
                label: "Route Progress",
                value: dayProgressPct,
                max: 100,
                unit: "%",
                tone: "cyan",
              },
              {
                id: "capacity",
                label: "Fleet Capacity",
                value: capacityPct,
                max: 100,
                unit: "%",
                tone: capacityPct >= 90 ? "warn" : "ok",
              },
              {
                id: "overflow",
                label: "Overflow Risk",
                value: overflowVal,
                max: 10,
                unit: "bins",
                tone: overflowVal > 0 ? "crit" : "ok",
                critAt: 1,
              },
            ]}
          />
        </div>
      </div>

      <button
        className="btn-ghost text-xs"
        onClick={() => setShowSecondary((v) => !v)}
      >
        {showSecondary ? "Hide" : "Show"} secondary metrics
      </button>
      {showSecondary && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {SECONDARY_KPIS.map(({ key, label, lowerIsBetter }) => (
            <KpiCard
              key={key}
              label={label}
              value={(displayEntry.data as Record<string, number>)[key]}
              delta={computeDelta(filteredEntries, displayDay, key)}
              colorize
              lowerIsBetter={lowerIsBetter}
            />
          ))}
        </div>
      )}

      <p className="text-[10px] text-canvas-muted">
        {logScale
          ? "Daily KPI timeseries · log-scale (symlog overflows, log profit/km/kg)"
          : "Daily KPI timeseries · linear scale"}
      </p>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {PRIMARY_KPIS.map(({ key, label }) => (
          <MetricTimeseries
            key={key}
            policySeries={policySeries}
            metricKey={key}
            label={label}
            logScale={logScale}
          />
        ))}
      </div>
    </>
  );
}

export function SimMonitorAdvancedPanels({
  showBinFill,
  showTourTable,
  displayEntry,
  displayDay,
  displayFailureEntry,
  policyVizEntries,
  selectedPolicy,
  selectedSample,
  theme,
  logScale,
  policyVizLive,
  telemetryTrendsKey,
  sourceRunLabel,
  activeLogPath,
  projectRoot,
  duckdbProfitRange,
  filteredEntries,
  dayRange,
  setSelectedDay,
  duckdbReady,
  lastPipeline,
  setDuckdbProfitRange,
}: {
  showBinFill: boolean;
  showTourTable: boolean;
  displayEntry: DayLogEntry;
  displayDay: number;
  displayFailureEntry: SimFailureEntry | null;
  policyVizEntries: PolicyVizEntry[];
  selectedPolicy: string | null;
  selectedSample: number | null;
  theme: "dark" | "light";
  logScale: boolean;
  policyVizLive: boolean;
  telemetryTrendsKey: number;
  sourceRunLabel: string | null;
  activeLogPath: string | null;
  projectRoot: string | null;
  duckdbProfitRange: [number, number] | null;
  filteredEntries: DayLogEntry[];
  dayRange: { min: number; max: number };
  setSelectedDay: (d: number | null) => void;
  duckdbReady: boolean;
  lastPipeline: any;
  setDuckdbProfitRange: (r: [number, number] | null) => void;
}) {
  return (
    <>
      {showBinFill && <BinFillStrip data={displayEntry.data} />}
      {showTourTable && (
        <TourTable
          data={displayEntry.data}
          day={displayDay}
          policy={displayEntry.policy}
        />
      )}

      <FailureAnalysisPanel
        entry={displayFailureEntry}
        embedded={displayEntry?.data.failure_analysis ?? null}
      />

      <PolicyTelemetryPanel
        entries={policyVizEntries}
        policy={selectedPolicy}
        sampleId={selectedSample}
        day={displayDay}
        theme={theme}
        logScale={logScale}
        live={policyVizLive}
      />

      <PolicyTelemetryTrendsPanel
        theme={theme}
        logScale={logScale}
        refreshKey={telemetryTrendsKey}
        initialPolicy={selectedPolicy}
        initialRunLabel={sourceRunLabel}
      />

      <GraphTopologyPanel
        logPath={activeLogPath}
        projectRoot={projectRoot}
        simData={displayEntry?.data ?? null}
        theme={theme}
        duckdbProfitRange={duckdbProfitRange}
        entries={filteredEntries}
        displayDay={displayDay}
        dayRange={dayRange}
        onDaySelect={setSelectedDay}
        logScale={logScale}
      />

      {duckdbReady && lastPipeline?.tableName === "monitor_sim" && (
        <SqlQueryPanel
          tableName={lastPipeline.tableName}
          theme={theme}
          onDaySelect={(day) => setSelectedDay(day)}
          onProfitRange={(min, max) => setDuckdbProfitRange([min, max])}
          highlightPolicies={selectedPolicy ? [selectedPolicy] : null}
          highlightRunLabels={sourceRunLabel ? [sourceRunLabel] : null}
          brushSqlSync
          autoRunOnBrushSync
          portfolioMode
          portfolioRunLabels={sourceRunLabel ? [sourceRunLabel] : []}
        />
      )}
    </>
  );
}
