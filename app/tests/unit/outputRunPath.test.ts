import { describe, expect, it } from "vitest";
import {
  extractOutputRunPathFromLogLines,
  localPathFromUri,
  mlflowRunDirFromArtifactUri,
  outputRunPathFromHydraArtifact,
  outputRunPathFromJsonl,
  resolveLocalProjectPath,
  sqlitePathFromStorageUrl,
  sqliteStoragePathFromUrl,
} from "../../src/utils/runs/outputRunPath";

describe("outputRunPathFromJsonl", () => {
  it("strips the /hydra/ suffix to the run root", () => {
    expect(outputRunPathFromJsonl("assets/output/30days/run1/hydra/log.jsonl")).toBe(
      "assets/output/30days/run1"
    );
  });

  it("falls back to the parent directory", () => {
    expect(outputRunPathFromJsonl("assets/output/run2/day_log.jsonl")).toBe("assets/output/run2");
  });

  it("normalises Windows separators", () => {
    expect(outputRunPathFromJsonl("assets\\output\\run\\hydra\\x.jsonl")).toBe("assets/output/run");
  });
});

describe("outputRunPathFromHydraArtifact", () => {
  it("handles artifacts inside hydra/ and a bare hydra dir", () => {
    expect(outputRunPathFromHydraArtifact("out/run/hydra/config.yaml")).toBe("out/run");
    expect(outputRunPathFromHydraArtifact("out/run/hydra")).toBe("out/run");
  });
});

describe("extractOutputRunPathFromLogLines", () => {
  it("prefers the newest matching line", () => {
    const lines = [
      "Hydra config snapshot saved → assets/output/old/hydra/snapshot.yaml",
      "Pruned config saved → assets/output/new/hydra/pruned.yaml",
    ];
    expect(extractOutputRunPathFromLogLines(lines)).toBe("assets/output/new");
  });

  it("extracts bare assets/output paths, skipping jsonl artefacts", () => {
    expect(
      extractOutputRunPathFromLogLines(["writing assets/output/30days/runA/log.jsonl"])
    ).toBeNull();
    expect(extractOutputRunPathFromLogLines(["saved to assets/output/30days/runA"])).toBe(
      "assets/output/30days/runA"
    );
  });

  it("returns null with no matches", () => {
    expect(extractOutputRunPathFromLogLines(["nothing here"])).toBeNull();
  });

  it("reads the ASCII arrow the simulator prints now", () => {
    const lines = [
      "[INFO] Hydra config snapshot saved -> C:\\wsr\\outputs/simulations\\3days\\run\\hydra",
    ];
    expect(extractOutputRunPathFromLogLines(lines)).toBe("C:/wsr/outputs/simulations/3days/run");
  });

  it("extracts bare outputs/simulations paths", () => {
    expect(
      extractOutputRunPathFromLogLines(["saved to outputs/simulations/30days/runA"])
    ).toBe("outputs/simulations/30days/runA");
  });
});

describe("URI helpers", () => {
  it("localPathFromUri decodes file:// URIs", () => {
    expect(localPathFromUri("file:///home/user/ml%20runs")).toBe("/home/user/ml runs");
    expect(localPathFromUri("file:///C:/Users/x")).toBe("C:/Users/x");
    expect(localPathFromUri("/plain/path")).toBe("/plain/path");
    expect(localPathFromUri("https://remote/store")).toBeNull();
    expect(localPathFromUri("")).toBeNull();
  });

  it("mlflowRunDirFromArtifactUri strips /artifacts", () => {
    expect(mlflowRunDirFromArtifactUri("file:///mlruns/0/abc/artifacts")).toBe("/mlruns/0/abc");
    expect(mlflowRunDirFromArtifactUri("https://mlflow.remote/abc")).toBeNull();
  });

  it("sqlitePathFromStorageUrl parses sqlite:/// URLs", () => {
    expect(sqlitePathFromStorageUrl("sqlite:///optuna.db")).toBe("optuna.db");
    expect(sqlitePathFromStorageUrl("postgres://x")).toBeNull();
    expect(sqlitePathFromStorageUrl("sqlite:///  ")).toBeNull();
  });

  it("resolveLocalProjectPath anchors relative paths at the project root", () => {
    expect(resolveLocalProjectPath("./mlruns", "/repo/")).toBe("/repo/mlruns");
    expect(resolveLocalProjectPath("/abs/mlruns", "/repo")).toBe("/abs/mlruns");
    expect(resolveLocalProjectPath("mlruns", null)).toBe("mlruns");
    expect(resolveLocalProjectPath("  ", "/repo")).toBeNull();
  });

  it("sqliteStoragePathFromUrl composes both helpers", () => {
    expect(sqliteStoragePathFromUrl("sqlite:///optuna.db", "/repo")).toBe("/repo/optuna.db");
    expect(sqliteStoragePathFromUrl("mysql://h/db", "/repo")).toBeNull();
  });
});
