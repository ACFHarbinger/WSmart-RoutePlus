import { KeyRound, Layers, Settings2 } from "lucide-react";
import type { CloudLauncherState } from "../../store/launchers";

export const PRESET_COMMANDS = [
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

export function CloudJobSpecSection({
  jobType,
  command,
  manifestPath,
  policyFilter,
  jobName,
  root,
  patch,
}: {
  jobType: "command" | "manifest";
  command: string;
  manifestPath: string;
  policyFilter: string;
  jobName: string;
  root: string;
  patch: CloudLauncherState["patch"];
}) {
  return (
    <div className="card bg-canvas-surface border border-canvas-border rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between border-b border-canvas-border pb-3">
        <div className="flex items-center gap-2">
          <Layers size={16} className="text-accent-primary" />
          <h3 className="text-sm font-semibold text-gray-200">
            1. Job Specification
          </h3>
        </div>

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
            One simulation command dispatched as a single cloud task.
            Checkpoints and day logs will be saved to the cloud object store
            daily.
          </p>
        </div>
      ) : (
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
  );
}

export function CloudComputeAndGurobiSections({
  maxParallel,
  cpus,
  memoryGb,
  retries,
  maxWallSeconds,
  idleSeconds,
  spot,
  gurobiMode,
  gurobiSecret,
  patch,
}: {
  maxParallel: number;
  cpus: number;
  memoryGb: number;
  retries: number;
  maxWallSeconds: number;
  idleSeconds: number;
  spot: boolean;
  gurobiMode: CloudLauncherState["gurobiMode"];
  gurobiSecret: string;
  patch: CloudLauncherState["patch"];
}) {
  return (
    <>
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
              onChange={(e) =>
                patch({ maxParallel: Number(e.target.value) || 1 })
              }
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
    </>
  );
}
