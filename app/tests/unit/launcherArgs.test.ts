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
      "cmptp",
      "cmptptc",
      "cmptptwtc",
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

  it("resolveSimDatasetPreviewPath matches test_sim.yaml load_dataset", () => {
    expect(resolveSimDatasetPreviewPath("figueiradafoz", 350, "emp", 30, 1, 42)).toBe(
      "data/simulator/datasets/figueiradafoz350_emp_wsr30_N1_seed42.npz"
    );
    expect(resolveSimDatasetPreviewPath("riomaior", 100, "gamma3", 90, 10, 7)).toBe(
      "data/simulator/datasets/riomaior100_gamma3_wsr90_N10_seed7.npz"
    );
  });

  it("groupPoliciesByFamily uses registry folders, not name prefixes", () => {
    const entries = [
      "aco_hh", "alns", "amphh", "bpc", "hgs", "hna", "na", "swc_tcf",
      "lbbd", "bb", "abpc_hg", "rl_alns",
    ].map((id) => ({ id, config_key: `policy_${id}` }));
    const groups = groupPoliciesByFamily(entries);
    const byFamily = Object.fromEntries(groups.map((g) => [g.id, g.policies]));
    expect(byFamily.neural).toEqual(["hna", "na"]);
    expect(byFamily.metaheuristics).toContain("amphh");
    expect(byFamily.neural).not.toContain("amphh");
    expect(byFamily.exact).toEqual(expect.arrayContaining(["lbbd", "bb"]));
    expect(byFamily.exact).not.toContain("swc_tcf");
    expect(byFamily.paper).toContain("swc_tcf");
    expect(byFamily.constructive).toEqual(expect.arrayContaining(["abpc_hg", "rl_alns"]));
  });

  it("buildTrainHubHydraArgs emits envs@train.env=<problem> and models=deep_decoder", () => {
    const trainArgs = buildTrainHubHydraArgs({
      mode: "train",
      seed: 42,
      wandb: false,
      problem: "cmptptwtc",
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

    expect(trainArgs).toContain("envs@train.env=cmptptwtc");
    expect(trainArgs).toContain("models=deep_decoder");
    expect(trainArgs).toContain("tracker.enabled=false");

    const metaArgs = buildTrainHubHydraArgs({
      mode: "meta",
      seed: 42,
      wandb: false,
      problem: "cmptp",
      model: "am",
      encoder: "gat",
      batchSize: 64,
      epochs: 10,
      hpoTrials: 1,
      hpoMethod: "tpe",
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
    expect(metaArgs).toContain("envs@meta_rl.env=cmptp");
    expect(metaArgs).toContain("meta_rl.env.name=cmptp");
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
