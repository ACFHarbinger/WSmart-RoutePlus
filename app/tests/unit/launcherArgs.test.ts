import { describe, expect, it } from "vitest";
import {
  PAPER_BENCHMARK_POLICIES,
  TRAIN_MODELS,
  TRAIN_PROBLEMS,
} from "../../src/store/launchers";
import {
  buildSimHydraArgs,
  groupPoliciesByFamily,
  resolveSimDatasetPreviewPath,
} from "../../src/components/launch/SimPolicySelector";
import { buildTrainHubHydraArgs } from "../../src/components/launch/TrainHubSections";
import { buildReportStudioCliArgs } from "../../src/components/launch/ReportStudioHelpers";
import { useReportGenStore } from "../../src/store/launchers";

describe("launcherArgs — Slice V7 Hydra argument builders & domain nomenclature", () => {
  it("synchronizes TRAIN_PROBLEMS with problem_factory.py and TRAIN_MODELS with logic/configs/models/", () => {
    expect(TRAIN_PROBLEMS).toEqual([
      "ptp",
      "cptp",
      "cptptc",
      "cptptwtc",
      "cvrp",
    ]);
    expect(TRAIN_MODELS).toEqual([
      "am",
      "tam",
      "deep_decoder",
      "moe",
      "ptr",
      "symnco",
    ]);
    expect(TRAIN_MODELS).not.toContain("ddam");
  });

  it("defines the 8 canonical PAPER_BENCHMARK_POLICIES", () => {
    expect(PAPER_BENCHMARK_POLICIES).toEqual([
      "aco_hh",
      "alns",
      "bpc",
      "hgs",
      "pg_clns",
      "psoma",
      "sans",
      "swc_tcf",
    ]);
  });

  it("buildSimHydraArgs includes sim.days and policy list", () => {
    const args = buildSimHydraArgs({
      selectedPolicies: ["alns", "hgs"],
      area: "figueiradafoz",
      numLoc: 350,
      days: 30,
      samples: 10,
      nCores: 8,
      distribution: "emp",
      seed: 42,
      extraOverrides: "sim.verbose=true\n",
    });

    expect(args).toContain("sim.policies=[alns,hgs]");
    expect(args).toContain("sim.days=30");
    expect(args).toContain("sim.n_samples=10");
    expect(args).toContain("sim.graph.area=figueiradafoz");
    expect(args).toContain("sim.graph.num_loc=350");
    expect(args).toContain("sim.data_distribution=emp");
    expect(args).toContain("sim.cpu_cores=8");
    expect(args).toContain("seed=42");
    expect(args).toContain("sim.verbose=true");
  });

  it("resolveSimDatasetPreviewPath builds the expected .npz path", () => {
    expect(resolveSimDatasetPreviewPath("figueiradafoz", 350, "emp")).toBe(
      "data/figueiradafoz_350_emp_test.npz"
    );
    expect(resolveSimDatasetPreviewPath("riomaior", 100, "gamma3")).toBe(
      "data/riomaior_100_gamma3_test.npz"
    );
  });

  it("groupPoliciesByFamily groups policies into Paper Benchmark, Metaheuristics, Exact, Neural, and Constructive", () => {
    const entries = [
      ...PAPER_BENCHMARK_POLICIES.map((id) => ({ id, config_key: `policy_${id}` })),
      { id: "gurobi_vrpp", config_key: "policy_gurobi_vrpp" },
      { id: "deep_decoder_rl", config_key: "policy_deep_decoder_rl" },
      { id: "regular_3d", config_key: "policy_regular_3d" },
    ];
    const groups = groupPoliciesByFamily(entries);
    const ids = groups.map((g) => g.id);
    expect(ids).toContain("paper");
    expect(ids).toContain("metaheuristics");
    expect(ids).toContain("exact");
    expect(ids).toContain("neural");
    expect(ids).toContain("constructive");
  });

  it("buildTrainHubHydraArgs emits envs@train.env=<problem> and models=deep_decoder", () => {
    const trainArgs = buildTrainHubHydraArgs({
      mode: "train",
      seed: 42,
      wandb: false,
      problem: "cptptwtc",
      model: "deep_decoder",
      encoder: "gat",
      batchSize: 512,
      epochs: 100,
      hpoTrials: 50,
      hpoMethod: "nsgaii",
      hpoWorkers: 1,
      metaStrategy: "hrl",
      metaLr: 0.00001,
      metaHistoryLength: 10,
      mrlBatchSize: 32,
      mrlStep: 1,
      checkpointPath: "",
      evalDataset: "",
      evalSamples: 10,
      evalStrategy: "greedy",
      extraOverrides: "",
    });

    expect(trainArgs).toContain("envs@train.env=cptptwtc");
    expect(trainArgs).toContain("models=deep_decoder");
    expect(trainArgs).toContain("tracker.enabled=false");
  });

  it("buildReportStudioCliArgs reflects bound scenario filter in simulation report mode", () => {
    const state = {
      ...useReportGenStore.getState(),
      tab: "simulation" as const,
      simMode: "report" as const,
      simScenarios: "figueiradafoz:350:emp",
    };
    const cli = buildReportStudioCliArgs("simulation", state);
    expect(cli).toContain("logic/gen/gen_simulation_analysis.py");
    expect(cli).toContain("--scenarios");
    expect(cli).toContain("figueiradafoz:350:emp");
  });
});
