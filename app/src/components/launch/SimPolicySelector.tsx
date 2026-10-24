import { useState } from "react";
import { RefreshCw, AlertTriangle } from "lucide-react";
import {
  PAPER_BENCHMARK_POLICIES,
  type PolicyViewMode,
} from "../../store/launchers";
import type { SimPolicyEntry } from "../../types";
export { PolicyLiveCard, SimLiveMonitorSection } from "./SimLiveMonitorSection";

export interface PolicyFamilyGroup {
  id: string;
  label: string;
  description: string;
  policies: string[];
}

export function groupPoliciesByFamily(
  availablePolicies: SimPolicyEntry[]
): PolicyFamilyGroup[] {
  const allIds = availablePolicies.map((p) => p.id);
  const paperSet = new Set<string>(PAPER_BENCHMARK_POLICIES);
  const paper = allIds.filter((id) => paperSet.has(id));
  const exact = allIds.filter((id) =>
    /^(bpc|gurobi|hexaly|ortools|exact)/i.test(id)
  );
  const metaheuristic = allIds.filter((id) =>
    /^(alns|hgs|aco|pg_clns|psoma|sans|tabu|sa_|vns|ga_|ils)/i.test(id)
  );
  const neural = allIds.filter((id) =>
    /^(am|tam|deep_decoder|moe|ptr|symnco|drl|rl_|ppo|dqn|meta)/i.test(id)
  );
  const rule = allIds.filter(
    (id) =>
      !exact.includes(id) &&
      !metaheuristic.includes(id) &&
      !neural.includes(id)
  );

  return [
    {
      id: "paper",
      label: "Paper Benchmark (8 Core)",
      description: "Canonical 8-policy benchmark suite from the manuscript",
      policies: paper.length > 0 ? paper : [...PAPER_BENCHMARK_POLICIES],
    },
    {
      id: "metaheuristics",
      label: "Metaheuristics & Local Search",
      description: "ALNS, HGS, ACO-HH, PG-CLNS, PSOMA, SANS",
      policies: metaheuristic,
    },
    {
      id: "exact",
      label: "Exact & Mathematical Solvers",
      description: "Branch-and-Price-and-Cut (BPC), Gurobi, OR-Tools, Hexaly",
      policies: exact,
    },
    {
      id: "neural",
      label: "Neural & DRL Policies",
      description: "Constructive attention & reinforcement learning decoders",
      policies: neural,
    },
    {
      id: "constructive",
      label: "Constructive & Threshold Heuristics",
      description: "SWC-TCF, regular periodic, and threshold rules",
      policies: rule,
    },
  ].filter((g) => g.policies.length > 0);
}

export function buildSimHydraArgs(params: {
  selectedPolicies: string[];
  area: string;
  numLoc: number;
  days: number;
  samples: number;
  nCores: number;
  distribution: string;
  seed: number;
  extraOverrides: string;
}): string[] {
  const args = [
    `sim.policies=[${params.selectedPolicies.join(",")}]`,
    `sim.days=${params.days}`,
    `sim.n_samples=${params.samples}`,
    `sim.graph.area=${params.area}`,
    `sim.graph.num_loc=${params.numLoc}`,
    `sim.data_distribution=${params.distribution}`,
    `sim.cpu_cores=${params.nCores}`,
    `seed=${params.seed}`,
  ];
  const extra = params.extraOverrides
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  return [...args, ...extra];
}

export function resolveSimDatasetPreviewPath(
  area: string,
  numLoc: number,
  distribution: string
): string {
  return `data/${area}_${numLoc}_${distribution}_test.npz`;
}

export function SimPolicySelectorCard({
  availablePolicies,
  selectedPolicies,
  onSelectPolicies,
  onTogglePolicy,
  policyViewMode,
  onChangeViewMode,
  policiesLoading,
  projectRoot,
  onReloadPolicies,
}: {
  availablePolicies: SimPolicyEntry[];
  selectedPolicies: string[];
  onSelectPolicies: (policies: string[]) => void;
  onTogglePolicy: (policy: string) => void;
  policyViewMode: PolicyViewMode;
  onChangeViewMode: (mode: PolicyViewMode) => void;
  policiesLoading: boolean;
  projectRoot: string | null;
  onReloadPolicies: () => void;
}) {
  const [confirmSelectAll, setConfirmSelectAll] = useState(false);
  const families = groupPoliciesByFamily(availablePolicies);
  const allPolicyIds = availablePolicies.map((p) => p.id);

  const handleSelectAllClick = () => {
    if (allPolicyIds.length > 16) {
      setConfirmSelectAll(true);
    } else {
      onSelectPolicies([...allPolicyIds]);
    }
  };

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-gray-200">
            Policies
            <span className="ml-2 text-xs font-normal text-canvas-muted">
              ({selectedPolicies.length} / {availablePolicies.length} selected)
            </span>
          </h2>
          <div
            className="flex items-center bg-canvas-bg rounded-lg p-0.5 border border-canvas-border text-[11px]"
            role="group"
            aria-label="Policy selection view mode"
          >
            <button
              type="button"
              onClick={() => onChangeViewMode("standard")}
              className={`px-2.5 py-0.5 rounded-md font-medium transition-colors ${
                policyViewMode === "standard"
                  ? "bg-accent-primary text-white"
                  : "text-canvas-muted hover:text-gray-200"
              }`}
            >
              Standard View
            </button>
            <button
              type="button"
              onClick={() => onChangeViewMode("advanced")}
              className={`px-2.5 py-0.5 rounded-md font-medium transition-colors ${
                policyViewMode === "advanced"
                  ? "bg-accent-primary text-white"
                  : "text-canvas-muted hover:text-gray-200"
              }`}
            >
              Advanced Grid ({availablePolicies.length})
            </button>
          </div>
        </div>

        <div className="flex gap-1.5 text-xs items-center">
          <button
            type="button"
            onClick={() => onSelectPolicies([...PAPER_BENCHMARK_POLICIES])}
            className="btn-ghost py-0.5 px-2 text-accent-secondary"
            title="Select the 8 canonical paper benchmark policies"
          >
            Paper Preset (8)
          </button>
          <button
            type="button"
            onClick={onReloadPolicies}
            disabled={policiesLoading || !projectRoot}
            className="btn-ghost py-0.5 px-2 flex items-center gap-1"
            title="Reload from test_sim.yaml"
          >
            <RefreshCw size={11} className={policiesLoading ? "animate-spin" : ""} />
          </button>
          {policyViewMode === "advanced" && (
            <button
              type="button"
              onClick={handleSelectAllClick}
              className="btn-ghost py-0.5 px-2"
            >
              All
            </button>
          )}
          <button
            type="button"
            onClick={() => {
              setConfirmSelectAll(false);
              onSelectPolicies([]);
            }}
            className="btn-ghost py-0.5 px-2"
          >
            None
          </button>
        </div>
      </div>

      {confirmSelectAll && (
        <div
          role="alertdialog"
          aria-label="Confirm select all policies"
          className="flex items-center justify-between gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200"
        >
          <div className="flex items-center gap-2">
            <AlertTriangle size={14} className="text-amber-400 shrink-0" />
            <span>
              Selecting all {allPolicyIds.length} policies will spawn a heavy multi-hour
              simulation batch. Proceed?
            </span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => {
                onSelectPolicies([...allPolicyIds]);
                setConfirmSelectAll(false);
              }}
              className="px-2.5 py-1 rounded bg-amber-500 text-slate-950 font-semibold hover:bg-amber-400"
            >
              Confirm All ({allPolicyIds.length})
            </button>
            <button
              type="button"
              onClick={() => setConfirmSelectAll(false)}
              className="btn-ghost px-2 py-1"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {policyViewMode === "standard" ? (
        <div className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
          {families.map((fam) => {
            const selectedInFam = fam.policies.filter((p) =>
              selectedPolicies.includes(p)
            ).length;
            return (
              <div
                key={fam.id}
                className="rounded-lg border border-canvas-border bg-canvas-bg/50 p-2.5 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-xs font-semibold text-gray-200">
                      {fam.label}
                    </span>
                    <span className="ml-2 text-[10px] font-mono text-canvas-muted">
                      ({selectedInFam}/{fam.policies.length})
                    </span>
                    <p className="text-[10px] text-canvas-muted">{fam.description}</p>
                  </div>
                  <div className="flex gap-1 text-[10px]">
                    <button
                      type="button"
                      onClick={() =>
                        onSelectPolicies(
                          Array.from(new Set([...selectedPolicies, ...fam.policies]))
                        )
                      }
                      className="btn-ghost py-0.5 px-1.5"
                    >
                      + Family
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        onSelectPolicies(
                          selectedPolicies.filter((p) => !fam.policies.includes(p))
                        )
                      }
                      className="btn-ghost py-0.5 px-1.5"
                    >
                      Clear
                    </button>
                  </div>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {fam.policies.map((p) => {
                    const active = selectedPolicies.includes(p);
                    return (
                      <button
                        key={p}
                        type="button"
                        onClick={() => onTogglePolicy(p)}
                        className={`flex items-center gap-1.5 py-1 px-2.5 rounded-md border font-mono text-xs transition-colors ${
                          active
                            ? "border-accent-primary bg-accent-primary/15 text-accent-secondary"
                            : "border-canvas-border text-gray-400 hover:border-canvas-muted"
                        }`}
                      >
                        <span
                          className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                            active ? "bg-accent-primary" : "bg-canvas-border"
                          }`}
                        />
                        <span>{p}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="grid grid-cols-3 sm:grid-cols-4 gap-2 max-h-64 overflow-y-auto pr-1">
          {availablePolicies.map(({ id: p, config_key }) => {
            const active = selectedPolicies.includes(p);
            return (
              <label
                key={p}
                title={`config: ${config_key}`}
                className={`flex items-center gap-2 py-2 px-3 rounded-lg border cursor-pointer font-mono text-xs transition-colors ${
                  active
                    ? "border-accent-primary bg-accent-primary/15 text-accent-secondary"
                    : "border-canvas-border text-gray-400 hover:border-canvas-muted"
                }`}
              >
                <input
                  type="checkbox"
                  checked={active}
                  onChange={() => onTogglePolicy(p)}
                  className="sr-only"
                />
                <span
                  className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                    active ? "bg-accent-primary" : "bg-canvas-border"
                  }`}
                />
                <span className="truncate">{p}</span>
              </label>
            );
          })}
        </div>
      )}

      {selectedPolicies.length === 0 && (
        <p className="text-xs text-accent-warning">Select at least one policy.</p>
      )}
    </div>
  );
}
