/**
 * Cloud Launcher — Studio Launcher for distributed simulations and batch manifests (CR-14 / §J.1).
 */
import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import {
  Cloud,
  Play,
  Terminal,
  Copy,
  Check,
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
import {
  CloudComputeAndGurobiSections,
  CloudJobSpecSection,
} from "../../components/cloud/CloudLauncherSections";

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
    void refreshBackends();
  }, [refreshBackends]);

  const cliPreview = useMemo(() => {
    const parts = ["python", "main.py", "cloud"];
    if (root && root !== "outputs/cloud") {
      parts.push("--root", root);
    }
    parts.push("submit");

    if (jobType === "manifest") {
      parts.push(
        "--manifest",
        manifestPath ? `"${manifestPath}"` : '"manifest.yaml"'
      );
      if (policyFilter) {
        parts.push("--policy", policyFilter);
      }
    } else {
      parts.push("--command", `"${command || "test_sim"}"`);
    }

    if (jobName) parts.push("--name", `"${jobName}"`);
    if (provider) parts.push("--provider", provider);
    if (compute) parts.push("--compute", compute);
    if (maxParallel > 0) parts.push("--max-parallel", String(maxParallel));
    if (cpus > 0) parts.push("--cpus", String(cpus));
    if (memoryGb > 0) parts.push("--memory-gb", String(memoryGb));
    if (spot) parts.push("--spot");
    if (retries > 0) parts.push("--retries", String(retries));
    if (maxWallSeconds > 0) parts.push("--max-wall", String(maxWallSeconds));
    if (idleSeconds > 0) parts.push("--idle", String(idleSeconds));

    if (gurobiMode !== "none") {
      parts.push("--gurobi", gurobiMode);
      if (gurobiSecret) parts.push("--gurobi-secret", gurobiSecret);
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

  const estimateRequest = useMemo<CloudSubmitRequest | null>(() => {
    const source =
      jobType === "manifest" ? manifestPath.trim() : command.trim();
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
  }, [
    jobType,
    manifestPath,
    command,
    policyFilter,
    provider,
    compute,
    maxParallel,
    spot,
    machineType,
    region,
  ]);
  const estimate = useCloudEstimate(estimateRequest, { projectRoot, root });

  const handleCopyCli = () => {
    void navigator.clipboard.writeText(cliPreview);
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
        targets: parseTargets(targets).length
          ? parseTargets(targets)
          : undefined,
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
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-canvas-border pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Cloud size={24} className="text-accent-primary" />
            <h1 className="text-xl font-bold tracking-tight text-white">
              Cloud Launcher
            </h1>
          </div>
          <p className="text-xs text-canvas-muted mt-1">
            Dispatch single simulations, full paper reruns, or multi-cell
            manifests across local processes, GCP, AWS, or Azure.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() =>
              patch({
                jobType: "command",
                command:
                  "test_sim --policies aco_hh alns --days 30 --area figueiradafoz",
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

      <form onSubmit={(e) => void handleSubmit(e)} className="space-y-6">
        <CloudJobSpecSection
          jobType={jobType}
          command={command}
          manifestPath={manifestPath}
          policyFilter={policyFilter}
          jobName={jobName}
          root={root}
          patch={patch}
        />

        <ProviderSelector
          provider={provider}
          compute={compute}
          cloudProfile={cloudProfile}
          onProviderChange={(p) => patch({ provider: p })}
          onComputeChange={(c) => patch({ compute: c })}
          onProfileChange={(prof) => patch({ cloudProfile: prof })}
          backends={backends}
          loadingBackends={loadingBackends}
          onRefreshBackends={() => void refreshBackends()}
        />

        <CloudComputeAndGurobiSections
          maxParallel={maxParallel}
          cpus={cpus}
          memoryGb={memoryGb}
          retries={retries}
          maxWallSeconds={maxWallSeconds}
          idleSeconds={idleSeconds}
          spot={spot}
          gurobiMode={gurobiMode}
          gurobiSecret={gurobiSecret}
          patch={patch}
        />

        <div className="card bg-canvas-surface border border-canvas-border rounded-xl p-4">
          <CostLimitsFields
            machineType={machineType}
            region={region}
            budgetUsd={budgetUsd}
            onChange={patch}
          />
          <div className="pt-4">
            <PlacementFields
              priority={priority}
              targets={targets}
              pool={pool}
              onChange={patch}
            />
          </div>
        </div>
        <EstimatePanel estimate={estimate} />

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
