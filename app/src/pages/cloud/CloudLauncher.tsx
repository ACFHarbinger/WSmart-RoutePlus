/**
 * Cloud Launcher — Studio Launcher for distributed simulations and batch manifests (CR-14 / §J.1).
 *
 * Provides GUI form controls for:
 * - Single simulation command or multi-cell experiment manifest.
 * - Target cloud provider (Local, GCP, AWS, Azure) and compute topology (Process, Batch, Serverless, VM).
 * - Resource sizing (parallel tasks, CPUs, RAM, spot capacity, retries, timeouts).
 * - Gurobi WLS secret management.
 * - Live CLI command preview.
 * - Machine type, region and budget cap, with the CR-10 cost and ETA estimate.
 */

import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import {
  Cloud,
  Play,
  Terminal,
  Copy,
  Check,
  Layers,
  Settings2,
  KeyRound,
  RefreshCw,
} from "lucide-react";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";
import { useCloudLauncherStore } from "../../store/launchers";
import type { CloudBackendEntry, CloudSubmitRequest } from "../../types/cloud";
import { listCloudBackends, submitCloudJob } from "../../utils/cloud";
import { CostLimitsFields } from "../../components/cloud/CostLimitsFields";
import { PlacementFields, parseTargets } from "../../components/cloud/PlacementFields";
import { EstimatePanel } from "../../components/cloud/EstimatePanel";
import { useCloudEstimate } from "../../hooks/cloud/useCloudEstimate";
import { ProviderSelector } from "../../components/cloud/ProviderSelector";

const PRESET_COMMANDS = [
  {
    name: "Rio Maior 100loc Quick Smoke (1 day)",
    cmd: "test_sim sim.graph.area=riomaior sim.graph.num_loc=100 sim.n_days=1 sim.policies=[alns]",
  },
  {
    name: "Figueira da Foz Multi-Policy (30 days)",
    cmd: "test_sim sim.graph.area=figueiradafoz sim.graph.num_loc=350 sim.n_days=30 sim.policies=[aco_hh,alns,hgs]",
  },
  {
    name: "ACO-HH Paper Rerun Cell (60 days)",
    cmd: "test_sim sim.graph.area=porto sim.graph.num_loc=500 sim.n_days=60 sim.policies=[aco_hh]",
  },
];

export function CloudLauncher() {
  const { projectRoot, setMode } = useAppStore();
  const {
    jobType,
    command,
    manifestPath,
    policyFilter,
    jobName,
    provider,
    compute,
    cloudProfile,
    maxParallel,
    cpus,
    memoryGb,
    spot,
    retries,
    maxWallSeconds,
    idleSeconds,
    gurobiMode,
    gurobiSecret,
    machineType,
    region,
    budgetUsd,
    priority,
    targets,
    pool,
    root,
    patch,
  } = useCloudLauncherStore();

  const [backends, setBackends] = useState<CloudBackendEntry[]>([]);
  const [loadingBackends, setLoadingBackends] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [copied, setCopied] = useState(false);

  const refreshBackends = useCallback(async () => {
    setLoadingBackends(true);
    try {
      const result = await listCloudBackends({ projectRoot, root });
      setBackends(result);
    } catch {
      // Fallback: local backend always available
      setBackends([
        {
          provider: "local",
          compute: "process",
          capabilities: {
            provider: "local",
            kinds: ["process"],
            supports_spot: false,
            regions: ["local"],
            machine_types: {},
          },
        },
      ]);
    } finally {
      setLoadingBackends(false);
    }
  }, [projectRoot, root]);

  useEffect(() => {
    refreshBackends();
  }, [refreshBackends]);

  // Construct CLI preview string mirroring `logic/controllers/cli/cloud_parser.py`
  const cliPreview = useMemo(() => {
    const parts = ["python", "main.py", "cloud"];
    if (root && root !== "outputs/cloud") {
      parts.push("--root", root);
    }
    parts.push("submit");

    if (jobType === "manifest") {
      parts.push("--manifest", manifestPath ? `"${manifestPath}"` : '"manifest.yaml"');
      if (policyFilter) {
        parts.push("--policy", policyFilter);
      }
    } else {
      parts.push("--command", `"${command || 'test_sim'}"`);
    }

    if (jobName) {
      parts.push("--name", `"${jobName}"`);
    }

    if (provider) {
      parts.push("--provider", provider);
    }

    if (compute) {
      parts.push("--compute", compute);
    }

    if (maxParallel > 0) {
      parts.push("--max-parallel", String(maxParallel));
    }

    if (cpus > 0) {
      parts.push("--cpus", String(cpus));
    }

    if (memoryGb > 0) {
      parts.push("--memory-gb", String(memoryGb));
    }

    if (spot) {
      parts.push("--spot");
    }

    if (retries > 0) {
      parts.push("--retries", String(retries));
    }

    if (maxWallSeconds > 0) {
      parts.push("--max-wall", String(maxWallSeconds));
    }

    if (idleSeconds > 0) {
      parts.push("--idle", String(idleSeconds));
    }

    if (gurobiMode !== "none") {
      parts.push("--gurobi", gurobiMode);
      if (gurobiSecret) {
        parts.push("--gurobi-secret", gurobiSecret);
      }
    }

    if (machineType.trim()) parts.push("--machine-type", machineType.trim());
    if (region.trim()) parts.push("--region", region.trim());
    if (budgetUsd > 0) parts.push("--budget", String(budgetUsd));
    if (priority !== 0) parts.push("--priority", String(priority));
    for (const target of parseTargets(targets)) parts.push("--target", target);
    if (pool) parts.push("--pool");

    return parts.join(" ");
  }, [
    root,
    jobType,
    manifestPath,
    policyFilter,
    command,
    jobName,
    provider,
    compute,
    maxParallel,
    cpus,
    memoryGb,
    spot,
    retries,
    maxWallSeconds,
    idleSeconds,
    gurobiMode,
    gurobiSecret,
    machineType,
    region,
    budgetUsd,
    priority,
    targets,
    pool,
  ]);

  // CR-10: what the job would cost and how long it would take, before submitting.
  const estimateRequest = useMemo<CloudSubmitRequest | null>(() => {
    const source = jobType === "manifest" ? manifestPath.trim() : command.trim();
    if (!source) return null;
    return {
      ...(jobType === "manifest"
        ? { manifest: source, policy: policyFilter.trim() || undefined }
        : { command: source }),
      provider,
      compute,
      max_parallel: maxParallel,
      spot,
      machine_type: machineType.trim() || undefined,
      region: region.trim() || undefined,
    };
  }, [jobType, manifestPath, command, policyFilter, provider, compute, maxParallel, spot, machineType, region]);
  const estimate = useCloudEstimate(estimateRequest, { projectRoot, root });

  const handleCopyCli = () => {
    navigator.clipboard.writeText(cliPreview);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast.success("CLI command copied to clipboard");
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (jobType === "command" && !command.trim()) {
      toast.error("Simulation command cannot be empty");
      return;
    }

    if (jobType === "manifest" && !manifestPath.trim()) {
      toast.error("Manifest path cannot be empty");
      return;
    }

    if (gurobiMode !== "none" && !gurobiSecret.trim()) {
      toast.error("Please specify a Gurobi secret name");
      return;
    }

    setSubmitting(true);
    try {
      const req: CloudSubmitRequest = {
        name: jobName.trim() || undefined,
        provider,
        compute,
        max_parallel: maxParallel,
        cpus,
        memory_gb: memoryGb,
        spot,
        retries,
        max_wall: maxWallSeconds,
        idle: idleSeconds,
        gurobi: gurobiMode,
        gurobi_secret: gurobiSecret.trim() || undefined,
        machine_type: machineType.trim() || undefined,
        region: region.trim() || undefined,
        budget: budgetUsd > 0 ? budgetUsd : undefined,
        priority: priority !== 0 ? priority : undefined,
        targets: parseTargets(targets).length ? parseTargets(targets) : undefined,
        pool: pool || undefined,
      };

      if (jobType === "manifest") {
        req.manifest = manifestPath.trim();
        if (policyFilter.trim()) {
          req.policy = policyFilter.trim();
        }
      } else {
        req.command = command.trim();
      }

      const res = await submitCloudJob(req, { projectRoot, root });

      toast.success(`Cloud Job Submitted: ${res.job_id}`, {
        description: `Created ${res.tasks} task(s) in state '${res.state ?? "queued"}'.`,
        action: {
          label: "View Processes",
          onClick: () => setMode("process_monitor"),
        },
        duration: 8000,
      });
    } catch (err: unknown) {
      toast.error("Cloud Job Submission Failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-canvas-border pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Cloud size={24} className="text-accent-primary" />
            <h1 className="text-xl font-bold tracking-tight text-white">
              Cloud Launcher
            </h1>
          </div>
          <p className="text-xs text-canvas-muted mt-1">
            Dispatch single simulations, full paper reruns, or multi-cell manifests
            across local processes, GCP, AWS, or Azure.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() =>
              patch({
                jobType: "command",
                command: "test_sim --policies aco_hh alns --days 30 --area figueiradafoz",
                manifestPath: "",
                policyFilter: "",
                jobName: "",
                provider: "local",
                compute: "process",
                cloudProfile: "default",
                maxParallel: 4,
                cpus: 2,
                memoryGb: 4,
                spot: false,
                retries: 2,
                maxWallSeconds: 7200,
                idleSeconds: 60,
                gurobiMode: "none",
                gurobiSecret: "",
                root: "outputs/cloud",
              })
            }
            className="btn-ghost text-xs px-3 py-1.5 flex items-center gap-1.5 border border-canvas-border"
          >
            <RefreshCw size={12} />
            <span>Reset Defaults</span>
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Section 1: Job Specification */}
        <div className="card bg-canvas-surface border border-canvas-border rounded-xl p-4 space-y-4">
          <div className="flex items-center justify-between border-b border-canvas-border pb-3">
            <div className="flex items-center gap-2">
              <Layers size={16} className="text-accent-primary" />
              <h3 className="text-sm font-semibold text-gray-200">
                1. Job Specification
              </h3>
            </div>

            {/* Type selector */}
            <div className="flex items-center bg-canvas-elevated rounded-lg p-0.5 border border-canvas-border text-xs">
              <button
                type="button"
                onClick={() => patch({ jobType: "command" })}
                className={`px-3 py-1 rounded-md font-medium transition-all ${
                  jobType === "command"
                    ? "bg-accent-primary text-white shadow-sm"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                Single Command
              </button>
              <button
                type="button"
                onClick={() => patch({ jobType: "manifest" })}
                className={`px-3 py-1 rounded-md font-medium transition-all ${
                  jobType === "manifest"
                    ? "bg-accent-primary text-white shadow-sm"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                Experiment Manifest
              </button>
            </div>
          </div>

          {/* Job Name */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="sm:col-span-2 space-y-1">
              <label className="text-xs font-medium text-canvas-muted">
                Job Label / Identifier (Optional)
              </label>
              <input
                type="text"
                className="input-base text-xs font-mono"
                placeholder="e.g. aco-hh-benchmark-figueira"
                value={jobName}
                onChange={(e) => patch({ jobName: e.target.value })}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-canvas-muted">
                Local State Directory (--root)
              </label>
              <input
                type="text"
                className="input-base text-xs font-mono"
                placeholder="outputs/cloud"
                value={root}
                onChange={(e) => patch({ root: e.target.value })}
              />
            </div>
          </div>

          {/* Single Command Mode */}
          {jobType === "command" ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-canvas-muted">
                  Simulation Command Line
                </label>
                <div className="flex items-center gap-1.5 text-xs">
                  <span className="text-canvas-muted">Presets:</span>
                  {PRESET_COMMANDS.map((p) => (
                    <button
                      key={p.name}
                      type="button"
                      onClick={() => patch({ command: p.cmd })}
                      className="text-[11px] text-accent-primary hover:underline px-1 py-0.5"
                    >
                      {p.name.split(" ")[0]}
                    </button>
                  ))}
                </div>
              </div>

              <textarea
                rows={3}
                className="input-base font-mono text-xs resize-y"
                placeholder="test_sim sim.graph.area=figueiradafoz sim.graph.num_loc=350 sim.n_days=30 sim.policies=[aco_hh,alns]"
                value={command}
                onChange={(e) => patch({ command: e.target.value })}
              />
              <p className="text-[11px] text-canvas-muted">
                One simulation command dispatched as a single cloud task. Checkpoints
                and day logs will be saved to the cloud object store daily.
              </p>
            </div>
          ) : (
            /* Manifest Mode */
            <div className="space-y-3">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="sm:col-span-2 space-y-1">
                  <label className="text-xs font-medium text-canvas-muted">
                    Manifest Path (.yaml or .json)
                  </label>
                  <input
                    type="text"
                    className="input-base font-mono text-xs"
                    placeholder="assets/manifests/rerun_figueira.yaml"
                    value={manifestPath}
                    onChange={(e) => patch({ manifestPath: e.target.value })}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-canvas-muted">
                    Policy Filter Regex (--policy)
                  </label>
                  <input
                    type="text"
                    className="input-base font-mono text-xs"
                    placeholder="e.g. aco_hh or alns"
                    value={policyFilter}
                    onChange={(e) => patch({ policyFilter: e.target.value })}
                  />
                </div>
              </div>
              <p className="text-[11px] text-canvas-muted">
                Each cell in the manifest generates an independent task executed in
                parallel across the cloud compute cluster.
              </p>
            </div>
          )}
        </div>

        {/* Section 2: Provider & Compute Environment */}
        <ProviderSelector
          provider={provider}
          compute={compute}
          cloudProfile={cloudProfile}
          onProviderChange={(p) => patch({ provider: p })}
          onComputeChange={(c) => patch({ compute: c })}
          onProfileChange={(prof) => patch({ cloudProfile: prof })}
          backends={backends}
          loadingBackends={loadingBackends}
          onRefreshBackends={refreshBackends}
        />

        {/* Section 3: Resource Allocation & Resilience */}
        <div className="card bg-canvas-surface border border-canvas-border rounded-xl p-4 space-y-4">
          <div className="flex items-center gap-2 border-b border-canvas-border pb-3">
            <Settings2 size={16} className="text-accent-primary" />
            <h3 className="text-sm font-semibold text-gray-200">
              2. Compute Sizing & Fault Tolerance
            </h3>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="space-y-1">
              <label className="text-xs font-medium text-canvas-muted">
                Max Parallel Tasks
              </label>
              <input
                type="number"
                min={1}
                max={512}
                className="input-base font-mono text-xs"
                value={maxParallel}
                onChange={(e) => patch({ maxParallel: Number(e.target.value) || 1 })}
              />
            </div>

            <div className="space-y-1">
              <label className="text-xs font-medium text-canvas-muted">
                CPUs per Task
              </label>
              <input
                type="number"
                min={1}
                max={128}
                step={0.5}
                className="input-base font-mono text-xs"
                value={cpus}
                onChange={(e) => patch({ cpus: Number(e.target.value) || 1 })}
              />
            </div>

            <div className="space-y-1">
              <label className="text-xs font-medium text-canvas-muted">
                RAM per Task (GB)
              </label>
              <input
                type="number"
                min={1}
                max={512}
                step={1}
                className="input-base font-mono text-xs"
                value={memoryGb}
                onChange={(e) => patch({ memoryGb: Number(e.target.value) || 1 })}
              />
            </div>

            <div className="space-y-1">
              <label className="text-xs font-medium text-canvas-muted">
                Automatic Retries
              </label>
              <input
                type="number"
                min={0}
                max={10}
                className="input-base font-mono text-xs"
                value={retries}
                onChange={(e) => patch({ retries: Number(e.target.value) || 0 })}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
            <div className="space-y-1">
              <label className="text-xs font-medium text-canvas-muted">
                Max Wall Time (Seconds)
              </label>
              <input
                type="number"
                min={60}
                step={300}
                className="input-base font-mono text-xs"
                value={maxWallSeconds}
                onChange={(e) =>
                  patch({ maxWallSeconds: Number(e.target.value) || 3600 })
                }
              />
            </div>

            <div className="space-y-1">
              <label className="text-xs font-medium text-canvas-muted">
                Idle Timeout (Seconds)
              </label>
              <input
                type="number"
                min={10}
                step={10}
                className="input-base font-mono text-xs"
                value={idleSeconds}
                onChange={(e) =>
                  patch({ idleSeconds: Number(e.target.value) || 60 })
                }
              />
            </div>

            <div className="flex items-center gap-2 pt-5">
              <input
                type="checkbox"
                id="spot-toggle"
                className="rounded border-canvas-border bg-canvas-elevated text-accent-primary focus:ring-0 w-4 h-4 cursor-pointer"
                checked={spot}
                onChange={(e) => patch({ spot: e.target.checked })}
              />
              <label
                htmlFor="spot-toggle"
                className="text-xs font-medium text-gray-200 cursor-pointer select-none"
              >
                Use Spot / Preemptible Instances
              </label>
            </div>
          </div>
        </div>

        {/* Section 4: Gurobi Licensing */}
        <div className="card bg-canvas-surface border border-canvas-border rounded-xl p-4 space-y-4">
          <div className="flex items-center gap-2 border-b border-canvas-border pb-3">
            <KeyRound size={16} className="text-accent-primary" />
            <h3 className="text-sm font-semibold text-gray-200">
              3. Gurobi Optimizer Licensing
            </h3>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="space-y-1">
              <label className="text-xs font-medium text-canvas-muted">
                License Mode (--gurobi)
              </label>
              <select
                className="select-base text-xs"
                value={gurobiMode}
                onChange={(e) =>
                  patch({
                    gurobiMode: e.target.value as "none" | "wls_secret" | "byo",
                  })
                }
              >
                <option value="none">None (Non-Gurobi Policies)</option>
                <option value="wls_secret">WLS License via Secret Manager</option>
                <option value="byo">BYO Node-locked / Floating License</option>
              </select>
            </div>

            {gurobiMode !== "none" && (
              <div className="sm:col-span-2 space-y-1">
                <label className="text-xs font-medium text-canvas-muted">
                  Gurobi Secret Reference Name (--gurobi-secret)
                </label>
                <input
                  type="text"
                  className="input-base font-mono text-xs"
                  placeholder="e.g. gurobi-wls-key or WLS_SECRET_NAME"
                  value={gurobiSecret}
                  onChange={(e) => patch({ gurobiSecret: e.target.value })}
                />
                <p className="text-[10px] text-canvas-muted">
                  Enter the key name in the cloud secret manager. The license key
                  itself is never transmitted through the launcher.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Section 5: Cost & ETA Estimation (CR-10) */}
        <div className="card bg-canvas-surface border border-canvas-border rounded-xl p-4">
          <CostLimitsFields machineType={machineType} region={region} budgetUsd={budgetUsd} onChange={patch} />
          <div className="pt-4">
            <PlacementFields priority={priority} targets={targets} pool={pool} onChange={patch} />
          </div>
        </div>
        <EstimatePanel estimate={estimate} />

        {/* Section 6: CLI Command Preview & Launch */}
        <div className="card bg-canvas-surface border border-canvas-border rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Terminal size={16} className="text-accent-primary" />
              <h3 className="text-sm font-semibold text-gray-200">
                CLI Command Preview
              </h3>
            </div>
            <button
              type="button"
              onClick={handleCopyCli}
              className="text-xs text-canvas-muted hover:text-gray-200 flex items-center gap-1 transition-colors px-2 py-1 rounded hover:bg-canvas-elevated"
            >
              {copied ? (
                <>
                  <Check size={12} className="text-accent-success" />
                  <span className="text-accent-success">Copied</span>
                </>
              ) : (
                <>
                  <Copy size={12} />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>

          <pre className="p-3 rounded-lg bg-canvas-elevated/70 border border-canvas-border text-xs font-mono text-gray-300 overflow-x-auto whitespace-pre-wrap break-all select-all">
            {cliPreview}
          </pre>
        </div>

        {/* Submit Button */}
        <div className="flex items-center justify-end gap-3 pt-2">
          <button
            type="submit"
            disabled={submitting}
            className="btn-primary flex items-center gap-2 px-6 py-2.5 text-sm font-semibold shadow-lg shadow-accent-primary/20"
          >
            {submitting ? (
              <>
                <RefreshCw size={16} className="animate-spin" />
                <span>Submitting to Cloud...</span>
              </>
            ) : (
              <>
                <Play size={16} />
                <span>Launch Cloud Job</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
