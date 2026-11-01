/**
 * Persisted form state for the three launcher pages (§D.4 session persistence).
 *
 * Each store uses a single `patch` action so components can update any subset
 * of fields without per-field setters.  Ephemeral runtime state (process ID,
 * live metrics, run status) stays in local component state.
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { CloudComputeKind, CloudGurobiMode, CloudProvider } from "../types/cloud";

// ── Simulation Launcher (§G.9)

export type PolicyViewMode = "standard" | "advanced";

export const PAPER_BENCHMARK_POLICIES = [
  "aco_hh",
  "alns",
  "bpc",
  "hgs",
  "pg_clns",
  "psoma",
  "sans",
  "swc_tcf",
] as const;

/** Problems `logic/configs/tasks/gen_data.yaml` accepts. */
export const DATA_GEN_PROBLEMS = ["ptp", "cmptp", "cmptptc", "all"] as const;

const RETIRED_PROBLEM_IDS = new Set(["vrpp", "wcvrp", "scwcvrp", "cwcvrp"]);

export const TRAIN_PROBLEMS = [
  "ptp",
  "cmptp",
  "cmptptc",
  "cmptptwtc",
  "cvrp",
] as const;

export const TRAIN_MODELS = [
  "am",
  "tam",
  "deep_decoder",
  "moe",
  "ptr",
  "symnco",
] as const;

interface SimLauncherState {
  selectedPolicies: string[];
  policyViewMode: PolicyViewMode;
  area: string;
  numLoc: number;
  days: number;
  samples: number;
  nCores: number;
  seed: number;
  distribution: string;
  extraOverrides: string;
  patch: (updates: Partial<Omit<SimLauncherState, "patch">>) => void;
}

export const useSimLauncherStore = create<SimLauncherState>()(
  persist(
    (set) => ({
      selectedPolicies: [...PAPER_BENCHMARK_POLICIES],
      policyViewMode: "standard",
      area: "figueiradafoz",
      numLoc: 350,
      days: 30,
      samples: 1,
      nCores: 4,
      seed: 42,
      distribution: "emp",
      extraOverrides: "",
      patch: (updates) => set(updates as Partial<SimLauncherState>),
    }),
    { name: "wsroute-sim-launcher" }
  )
);

// ── Training Hub (§G.10)

interface TrainHubState {
  trainMode: "train" | "hpo" | "meta" | "eval";
  problem: string;
  seed: number;
  wandb: boolean;
  extraOverrides: string;
  model: string;
  encoder: string;
  batchSize: number;
  epochs: number;
  hpoTrials: number;
  hpoMethod: string;
  hpoWorkers: number;
  metaStrategy: string;
  metaLr: number;
  metaHistoryLength: number;
  mrlBatchSize: number;
  mrlStep: number;
  checkpointPath: string;
  evalDataset: string;
  evalSamples: number;
  evalStrategy: string;
  patch: (updates: Partial<Omit<TrainHubState, "patch">>) => void;
}

export const useTrainHubStore = create<TrainHubState>()(
  persist(
    (set) => ({
      trainMode: "train",
      problem: "ptp",
      seed: 42,
      wandb: false,
      extraOverrides: "",
      model: "am",
      encoder: "gat",
      batchSize: 64,
      epochs: 100,
      hpoTrials: 50,
      hpoMethod: "nsgaii",
      hpoWorkers: 1,
      metaStrategy: "hrl",
      metaLr: 0.000005,
      metaHistoryLength: 10,
      mrlBatchSize: 256,
      mrlStep: 10,
      checkpointPath: "",
      evalDataset: "",
      evalSamples: 10,
      evalStrategy: "greedy",
      patch: (updates) => set(updates as Partial<TrainHubState>),
    }),
    { name: "wsroute-train-hub" }
  )
);

// ── System Tools (file_system + test_suite CLI parity with the retired PySide6 GUI)

export type SystemToolsTab = "update" | "delete" | "crypto" | "tests";

interface SystemToolsState {
  tab: SystemToolsTab;
  // file_system update
  targetEntry: string;
  outputKey: string;
  filenamePattern: string;
  updateOperation: string;
  updateValue: number;
  inputKeys: string;
  statsFunction: string;
  outputFilename: string;
  updatePreview: boolean;
  // file_system delete (checked = delete that directory)
  delWandb: boolean;
  delLog: boolean;
  delOutput: boolean;
  delData: boolean;
  delEval: boolean;
  delTest: boolean;
  delTestCheckpoint: boolean;
  delCache: boolean;
  deletePreview: boolean;
  // file_system cryptography
  envFile: string;
  symkeyName: string;
  cryptoInputPath: string;
  cryptoOutputPath: string;
  saltSize: number;
  keyLength: number;
  hashIterations: number;
  // test_suite
  testModules: string[];
  testKeyword: string;
  testMarkers: string;
  testVerbose: boolean;
  testCoverage: boolean;
  testParallel: boolean;
  testFailedFirst: boolean;
  testMaxfail: number;
  testDir: string;
  patch: (updates: Partial<Omit<SystemToolsState, "patch">>) => void;
}

export const useSystemToolsStore = create<SystemToolsState>()(
  persist(
    (set) => ({
      tab: "update" as SystemToolsTab,
      targetEntry: "",
      outputKey: "",
      filenamePattern: "",
      updateOperation: "",
      updateValue: 0,
      inputKeys: "",
      statsFunction: "",
      outputFilename: "",
      updatePreview: true,
      delWandb: false,
      delLog: false,
      delOutput: false,
      delData: false,
      delEval: false,
      delTest: false,
      delTestCheckpoint: false,
      delCache: false,
      deletePreview: true,
      envFile: "vars.env",
      symkeyName: "",
      cryptoInputPath: "",
      cryptoOutputPath: "",
      saltSize: 16,
      keyLength: 32,
      hashIterations: 100000,
      testModules: [],
      testKeyword: "",
      testMarkers: "",
      testVerbose: false,
      testCoverage: false,
      testParallel: false,
      testFailedFirst: false,
      testMaxfail: 0,
      testDir: "tests",
      patch: (updates) => set(updates as Partial<SystemToolsState>),
    }),
    { name: "wsroute-system-tools" }
  )
);

// ── Report & Presentation Generator (§H interim — archived logic/gen pipeline)

export type ReportGenTab = "dataset" | "simulation" | "presentation";
export type ReportGenEngine = "native" | "legacy";

export interface ReportGenState {
  tab: ReportGenTab;
  /** "native" runs the in-app §H generator; "legacy" spawns the archived Python scripts. */
  engine: ReportGenEngine;
  theme: "default" | "dark" | "light";
  // dataset analysis
  dsNpzCsv: string;
  dsTdCsv: string;
  dsNpzDir: string;
  dsOutMd: string;
  dsFiguresDir: string;
  dsForce: boolean;
  dsFiguresOnly: boolean;
  dsHtml: boolean;
  // simulation analysis
  simMode: "report" | "parse";
  simFontsize: number;
  simParetoPoints: "default" | "all" | "front";
  simHorizons: string;
  simScenarios: string;
  simStrategies: string;
  simConstructors: string;
  simImprovers: string;
  simAcceptance: string;
  simOutMd: string;
  simFiguresDir: string;
  simForce: boolean;
  simFiguresOnly: boolean;
  simHtml: boolean;
  simMapMode: "street" | "scatter";
  simHeatmapLabels: "both" | "show" | "hide";
  parseOutputDir: string;
  parseOutCsv: string;
  // presentation
  presFiguresDir: string;
  presOut: string;
  presAuthor: string;
  presCoauthors: string;
  presGroups: string;
  presResultsTable: "30d" | "90d" | "all" | "none";
  presResultsSplit: "none" | "strategy" | "constructor" | "improver";
  presSpeakerScript: boolean;
  presSpeakerOut: string;
  presImageMode: "native" | "fetch";
  presExcel: boolean;
  presHtml: boolean;
  presPdf: boolean;
  extraArgs: string;
  patch: (updates: Partial<Omit<ReportGenState, "patch">>) => void;
}

export const useReportGenStore = create<ReportGenState>()(
  persist(
    (set) => ({
      tab: "simulation" as ReportGenTab,
      engine: "native" as ReportGenEngine,
      theme: "default" as const,
      dsNpzCsv: "",
      dsTdCsv: "",
      dsNpzDir: "",
      dsOutMd: "",
      dsFiguresDir: "",
      dsForce: true,
      dsFiguresOnly: false,
      dsHtml: false,
      simMode: "report" as const,
      simFontsize: 0,
      simParetoPoints: "default" as const,
      simHorizons: "",
      simScenarios: "",
      simStrategies: "",
      simConstructors: "",
      simImprovers: "",
      simAcceptance: "",
      simOutMd: "",
      simFiguresDir: "",
      simForce: true,
      simFiguresOnly: false,
      simHtml: false,
      simMapMode: "street" as const,
      simHeatmapLabels: "both" as const,
      parseOutputDir: "outputs/simulations/90days",
      parseOutCsv: "public/global/simulation/simulation_summary_90d.csv",
      presFiguresDir: "public/figures/simulation/30d",
      presOut: "assets/windows/wsmart_route_results.pptx",
      presAuthor: "",
      presCoauthors: "",
      presGroups: "",
      presResultsTable: "30d" as const,
      presResultsSplit: "none" as const,
      presSpeakerScript: false,
      presSpeakerOut: "",
      presImageMode: "native" as const,
      presExcel: false,
      presHtml: false,
      presPdf: false,
      extraArgs: "",
      patch: (updates) => set(updates as Partial<ReportGenState>),
    }),
    { name: "wsroute-report-gen" }
  )
);

// ── Data Generation Wizard (§G.11)

interface DataGenState {
  dataSource: "synthetic" | "tsplib" | "sensor";
  tsplibPath: string;
  sensorCsvPath: string;
  problem: string;
  distributions: string[];
  datasetType: string;
  seed: number;
  overwrite: boolean;
  area: string;
  numLoc: number;
  nSamples: number;
  nDays: number;
  extraOverrides: string;
  patch: (updates: Partial<Omit<DataGenState, "patch">>) => void;
}

export const useDataGenStore = create<DataGenState>()(
  persist(
    (set) => ({
      dataSource: "synthetic",
      tsplibPath: "",
      sensorCsvPath: "",
      problem: "ptp",
      distributions: ["gamma3"],
      datasetType: "test_simulator",
      seed: 42,
      overwrite: true,
      area: "figueiradafoz",
      numLoc: 350,
      nSamples: 1,
      nDays: 30,
      extraOverrides: "",
      patch: (updates) => set(updates as Partial<DataGenState>),
    }),
    {
      name: "wsroute-data-gen",
      version: 1,
      migrate: (persisted, version) => {
        const state = persisted as DataGenState;
        if (version < 1 && RETIRED_PROBLEM_IDS.has(state.problem)) {
          state.problem = "ptp";
        }
        if (!DATA_GEN_PROBLEMS.includes(state.problem as (typeof DATA_GEN_PROBLEMS)[number])) {
          state.problem = "ptp";
        }
        return state;
      },
    }
  )
);

// ── Cloud Launcher (CR-14 / §J.1)

export interface CloudLauncherState {
  jobType: "command" | "manifest";
  command: string;
  manifestPath: string;
  policyFilter: string;
  jobName: string;
  provider: CloudProvider;
  compute: CloudComputeKind;
  cloudProfile: string;
  maxParallel: number;
  cpus: number;
  memoryGb: number;
  spot: boolean;
  retries: number;
  maxWallSeconds: number;
  idleSeconds: number;
  gurobiMode: CloudGurobiMode;
  gurobiSecret: string;
  /** CR-10: machine type and region for prices, and the job's cost cap (0: none). */
  machineType: string;
  region: string;
  budgetUsd: number;
  /** CR-09: priority, one placement target per line, and the warm pool. */
  priority: number;
  targets: string;
  pool: boolean;
  root: string;
  patch: (updates: Partial<Omit<CloudLauncherState, "patch">>) => void;
}

export const useCloudLauncherStore = create<CloudLauncherState>()(
  persist(
    (set) => ({
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
      machineType: "",
      region: "",
      budgetUsd: 0,
      priority: 0,
      targets: "",
      pool: false,
      gurobiMode: "none",
      gurobiSecret: "",
      root: "outputs/cloud",
      patch: (updates) => set(updates as Partial<CloudLauncherState>),
    }),
    { name: "wsroute-cloud-launcher" }
  )
);
