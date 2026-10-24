/**
 * Simulation Launcher — full-featured replacement for `just test-sim` (§G.9).
 *
 * Form parameters mirror the controller justfile recipe:
 *   sim.policies=[...], sim.days, sim.n_samples, sim.graph.area, sim.graph.num_loc,
 *   sim.data_distribution, sim.cpu_cores, seed
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Play, ChevronDown, ChevronUp, Terminal, Database } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useLaunchTriggerStore } from "../../store/launchTrigger";
import {
  PAPER_BENCHMARK_POLICIES,
  useSimLauncherStore,
  type PolicyViewMode,
} from "../../store/launchers";
import { useLoadedRunStore } from "../../store/loadedRun";
import { useProcessStore } from "../../store/process";
import { useSpawnProcess } from "../../hooks/process/useSpawnProcess";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import {
  collectPolicyVizFromLogLines,
  uniquePolicyVizPolicies,
} from "../../utils/benchmark/policyTelemetry";
import { extractJsonlPathFromLogLines } from "../../utils/benchmark/policyTelemetryTrends";
import { useProcessRunLabelBrush } from "../../hooks/brush/useProcessRunLabelBrush";
import { brushLogPathFromProcessLines, outputRunPathFromLogLines } from "../../utils/runs/outputRunPath";
import { collectLatestDayLogsByPolicy } from "../../utils/sim/dayLog";
import { findRecentLauncherProcessId } from "../../utils/process/launcherProcess";
import {
  buildSimHydraArgs,
  resolveSimDatasetPreviewPath,
  SimLiveMonitorSection,
  SimPolicySelectorCard,
} from "../../components/launch/SimPolicySelector";
import type { SimPolicyEntry } from "../../types";

const DISTRIBUTIONS = [
  { value: "emp", label: "Empirical" },
  { value: "gamma3", label: "Gamma-3" },
];

function NumberField({
  label,
  value,
  onChange,
  min = 1,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-canvas-muted">{label}</label>
      <input
        type="number"
        className="input-base font-mono text-sm w-28"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export function SimulationLauncher() {
  const { projectRoot, effectiveTheme: theme } = useAppStore();
  const { setMode, handoff } = useRecentHandoff();
  const setLoadedRun = useLoadedRunStore((s) => s.setLoadedRun);
  const {
    policy: activePolicy,
    runLabel: activeRunLabel,
    logScale,
    setPolicy,
  } = useGlobalFiltersStore();
  const { spawn, launching } = useSpawnProcess();

  const {
    selectedPolicies,
    policyViewMode,
    area,
    numLoc,
    days,
    samples,
    nCores,
    distribution,
    seed,
    extraOverrides,
    patch,
  } = useSimLauncherStore();

  const setSelectedPolicies = useCallback(
    (v: string[]) => patch({ selectedPolicies: v }),
    [patch]
  );
  const setPolicyViewMode = (v: PolicyViewMode) => patch({ policyViewMode: v });
  const setArea = (v: string) => patch({ area: v });
  const setNumLoc = (v: number) => patch({ numLoc: v });
  const setDays = (v: number) => patch({ days: v });
  const setSamples = (v: number) => patch({ samples: v });
  const setNCores = (v: number) => patch({ nCores: v });
  const setDistribution = (v: string) => patch({ distribution: v });
  const setSeed = (v: number) => patch({ seed: v });
  const setExtraOverrides = (v: string) => patch({ extraOverrides: v });

  const [availablePolicies, setAvailablePolicies] = useState<SimPolicyEntry[]>(
    PAPER_BENCHMARK_POLICIES.map((id) => ({ id, config_key: `policy_${id}` }))
  );
  const [policiesLoading, setPoliciesLoading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [liveProcessId, setLiveProcessId] = useState<string | null>(null);
  const processes = useProcessStore((s) => s.processes);
  const displayProcessId = useMemo(
    () => liveProcessId ?? findRecentLauncherProcessId(processes, "sim"),
    [liveProcessId, processes]
  );
  const displayProc = displayProcessId ? processes[displayProcessId] : null;
  const simStatus = displayProc?.status ?? null;

  const togglePolicy = (policy: string) => {
    const next = selectedPolicies.includes(policy)
      ? selectedPolicies.filter((p) => p !== policy)
      : [...selectedPolicies, policy];
    setSelectedPolicies(next);
  };

  const loadPolicies = useCallback(async () => {
    if (!projectRoot) return;
    setPoliciesLoading(true);
    try {
      const entries = await invoke<SimPolicyEntry[]>("list_sim_policies", {
        projectRoot,
      });
      if (entries.length > 0) {
        setAvailablePolicies(entries);
        const valid = new Set(entries.map((e) => e.id));
        const filtered = selectedPolicies.filter((p) => valid.has(p));
        if (filtered.length !== selectedPolicies.length) {
          setSelectedPolicies(filtered);
        }
      }
    } catch (err) {
      console.error("Failed to load policy registry:", err);
    } finally {
      setPoliciesLoading(false);
    }
  }, [projectRoot, selectedPolicies, setSelectedPolicies]);

  useEffect(() => {
    void loadPolicies();
  }, [projectRoot]); // eslint-disable-line react-hooks/exhaustive-deps

  const hydraArgs = useMemo(
    () =>
      buildSimHydraArgs({
        selectedPolicies,
        area,
        numLoc,
        days,
        samples,
        nCores,
        distribution,
        seed,
        extraOverrides,
      }),
    [selectedPolicies, area, numLoc, days, samples, nCores, distribution, seed, extraOverrides]
  );

  const datasetPreviewPath = useMemo(
    () => resolveSimDatasetPreviewPath(area, numLoc, distribution, days, samples, seed),
    [area, numLoc, distribution, days, samples, seed]
  );

  const commandPreview = `python main.py test_sim \\\n  ${hydraArgs.join(" \\\n  ")}`;

  const liveLogLines = displayProc?.logLines ?? [];
  const simJsonlPath = useMemo(
    () => extractJsonlPathFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const liveRunLabel = useProcessRunLabelBrush(displayProcessId, liveLogLines);

  const openCompletedInSummary = useCallback(() => {
    const targetPath = simJsonlPath;
    if (targetPath) {
      setLoadedRun({
        path: targetPath,
        label: liveRunLabel ?? `${area} N=${numLoc} (${distribution})`,
        source: "viewing",
        area,
        numLoc,
        distribution,
        days,
      });
      handoff(targetPath, "log");
    } else {
      setMode("simulation_summary");
    }
  }, [
    simJsonlPath,
    setLoadedRun,
    liveRunLabel,
    area,
    numLoc,
    distribution,
    days,
    handoff,
    setMode,
  ]);

  const launch = useCallback(async () => {
    if (!projectRoot || selectedPolicies.length === 0) return;
    const procId = `sim_${area}_${Date.now()}`;
    setLiveProcessId(procId);
    await spawn({
      id: procId,
      pythonArgs: ["main.py", "test_sim", ...hydraArgs],
      workingDir: projectRoot,
    });
  }, [projectRoot, selectedPolicies, hydraArgs, area, spawn]);

  const simNonce = useLaunchTriggerStore((s) => s.simNonce);
  useEffect(() => {
    if (simNonce > 0) void launch();
  }, [simNonce, launch]);

  const liveProcStatus = displayProc?.status ?? null;
  const latestByPolicy = useMemo(
    () => collectLatestDayLogsByPolicy(liveLogLines),
    [liveLogLines]
  );
  const isDone =
    simStatus === "completed" ||
    simStatus === "failed" ||
    simStatus === "cancelled";
  const liveEntries = Object.values(latestByPolicy);

  const policyVizEntries = useMemo(
    () => collectPolicyVizFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const vizPolicies = useMemo(
    () => uniquePolicyVizPolicies(policyVizEntries),
    [policyVizEntries]
  );
  const selectedPolicy = activePolicy ?? vizPolicies[0] ?? null;
  const outputRunPath = useMemo(
    () => outputRunPathFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const liveLogPath = useMemo(
    () => brushLogPathFromProcessLines(liveLogLines, "sim"),
    [liveLogLines]
  );
  const policyVizLive = liveProcStatus === "running";
  const [telemetryTrendsKey, setTelemetryTrendsKey] = useState(0);

  useEffect(() => {
    if (policyVizEntries.length > 0) {
      setTelemetryTrendsKey((k) => k + 1);
    }
  }, [policyVizEntries.length, liveLogLines.length]);

  return (
    <div className="space-y-4 max-w-2xl">
      <SimPolicySelectorCard
        availablePolicies={availablePolicies}
        selectedPolicies={selectedPolicies}
        onSelectPolicies={setSelectedPolicies}
        onTogglePolicy={togglePolicy}
        policyViewMode={policyViewMode}
        onChangeViewMode={setPolicyViewMode}
        policiesLoading={policiesLoading}
        projectRoot={projectRoot}
        onReloadPolicies={() => void loadPolicies()}
      />

      {/* Dataset & simulation params */}
      <div className="card space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <h2 className="text-sm font-semibold text-gray-200">
            Dataset & Parameters
          </h2>
          <span
            className="inline-flex items-center gap-1.5 text-[11px] font-mono px-2.5 py-1 rounded-md bg-canvas-bg border border-canvas-border text-accent-secondary"
            title="Resolved simulator test dataset path"
          >
            <Database size={11} />
            {datasetPreviewPath}
          </span>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-canvas-muted">
            Area (<code className="font-mono">sim.graph.area</code>)
          </label>
          <input
            type="text"
            className="input-base font-mono text-sm w-full"
            value={area}
            onChange={(e) => setArea(e.target.value)}
            placeholder="figueiradafoz"
          />
        </div>

        <div className="flex flex-wrap gap-4">
          <NumberField
            label="Num Locations"
            value={numLoc}
            onChange={setNumLoc}
            min={10}
            max={10000}
            step={50}
          />
          <NumberField
            label="Days (sim.days)"
            value={days}
            onChange={setDays}
            min={1}
            max={365}
          />
          <NumberField
            label="Samples"
            value={samples}
            onChange={setSamples}
            min={1}
            max={100}
          />
          <NumberField
            label="CPU Cores"
            value={nCores}
            onChange={setNCores}
            min={1}
            max={64}
          />
          <NumberField
            label="Seed"
            value={seed}
            onChange={setSeed}
            min={0}
            max={99999}
          />
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-canvas-muted">
            Distribution (<code className="font-mono">sim.data_distribution</code>)
          </label>
          <div className="flex gap-2">
            {DISTRIBUTIONS.map((d) => (
              <label
                key={d.value}
                className={`flex items-center gap-2 py-1.5 px-3 rounded-lg border cursor-pointer text-xs transition-colors ${
                  distribution === d.value
                    ? "border-accent-primary bg-accent-primary/15 text-accent-secondary"
                    : "border-canvas-border text-gray-400 hover:border-canvas-muted"
                }`}
              >
                <input
                  type="radio"
                  name="dist"
                  value={d.value}
                  checked={distribution === d.value}
                  onChange={() => setDistribution(d.value)}
                  className="sr-only"
                />
                {d.label}
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Advanced overrides */}
      <div className="card">
        <button
          className="w-full flex items-center justify-between text-sm font-medium text-gray-300"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          <span>Advanced Overrides</span>
          {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        {showAdvanced && (
          <div className="mt-3 space-y-1">
            <p className="text-xs text-canvas-muted">
              Extra Hydra overrides, one per line.
            </p>
            <textarea
              className="input-base w-full font-mono text-xs h-20 resize-y mt-1"
              value={extraOverrides}
              onChange={(e) => setExtraOverrides(e.target.value)}
              placeholder="sim.something=value"
              spellCheck={false}
            />
          </div>
        )}
      </div>

      {/* Command preview */}
      <div className="card space-y-2">
        <div className="flex items-center gap-2 text-xs text-canvas-muted">
          <Terminal size={12} />
          Command preview
        </div>
        <pre className="font-mono text-xs text-accent-secondary whitespace-pre-wrap bg-canvas-bg rounded-lg p-3">
          {commandPreview}
        </pre>
      </div>

      {/* Launch */}
      <div className="flex items-center gap-3">
        {!projectRoot && (
          <p className="text-xs text-accent-warning">
            Configure Project Root in Settings first.
          </p>
        )}
        <button
          onClick={() => void launch()}
          disabled={launching || !projectRoot || selectedPolicies.length === 0}
          className="btn-primary flex items-center gap-2"
        >
          <Play size={14} />
          {launching ? "Launching…" : "Launch Simulation"}
        </button>
      </div>

      {/* Live status panel */}
      {displayProcessId && (
        <SimLiveMonitorSection
          displayProcessId={displayProcessId}
          vizPolicies={vizPolicies}
          liveRunLabel={liveRunLabel}
          activeRunLabel={activeRunLabel}
          isDone={isDone}
          simStatus={simStatus}
          liveLogPath={liveLogPath}
          projectRoot={projectRoot}
          outputRunPath={outputRunPath}
          simJsonlPath={simJsonlPath}
          openCompletedInSummary={openCompletedInSummary}
          liveLogLines={liveLogLines}
          liveEntries={liveEntries}
          activePolicy={activePolicy}
          setPolicy={setPolicy}
          policyVizEntries={policyVizEntries}
          selectedPolicy={selectedPolicy}
          theme={theme}
          logScale={logScale}
          policyVizLive={policyVizLive}
          telemetryTrendsKey={telemetryTrendsKey}
        />
      )}
    </div>
  );
}
