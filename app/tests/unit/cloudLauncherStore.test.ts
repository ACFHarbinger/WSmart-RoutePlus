import { describe, expect, it, beforeEach } from "vitest";
import { useCloudLauncherStore } from "../../src/store/launchers";

describe("useCloudLauncherStore (CR-14)", () => {
  beforeEach(() => {
    // Reset defaults before each test
    useCloudLauncherStore.getState().patch({
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
    });
  });

  it("initializes with sensible defaults", () => {
    const state = useCloudLauncherStore.getState();
    expect(state.jobType).toBe("command");
    expect(state.provider).toBe("local");
    expect(state.compute).toBe("process");
    expect(state.maxParallel).toBe(4);
    expect(state.cpus).toBe(2);
    expect(state.memoryGb).toBe(4);
    expect(state.spot).toBe(false);
    expect(state.gurobiMode).toBe("none");
    expect(state.root).toBe("outputs/cloud");
  });

  it("updates subset of properties with patch", () => {
    const store = useCloudLauncherStore.getState();
    store.patch({
      provider: "gcp",
      compute: "batch",
      spot: true,
      cpus: 8,
      memoryGb: 32,
    });

    const updated = useCloudLauncherStore.getState();
    expect(updated.provider).toBe("gcp");
    expect(updated.compute).toBe("batch");
    expect(updated.spot).toBe(true);
    expect(updated.cpus).toBe(8);
    expect(updated.memoryGb).toBe(32);
    // Unchanged properties remain untouched
    expect(updated.jobType).toBe("command");
    expect(updated.maxParallel).toBe(4);
  });

  it("switches to manifest mode and updates manifest path and policy", () => {
    const store = useCloudLauncherStore.getState();
    store.patch({
      jobType: "manifest",
      manifestPath: "assets/manifests/rerun.yaml",
      policyFilter: "aco_hh",
    });

    const updated = useCloudLauncherStore.getState();
    expect(updated.jobType).toBe("manifest");
    expect(updated.manifestPath).toBe("assets/manifests/rerun.yaml");
    expect(updated.policyFilter).toBe("aco_hh");
  });

  it("updates Gurobi license mode and secret key name", () => {
    const store = useCloudLauncherStore.getState();
    store.patch({
      gurobiMode: "wls_secret",
      gurobiSecret: "wls_key_cloud_runner",
    });

    const updated = useCloudLauncherStore.getState();
    expect(updated.gurobiMode).toBe("wls_secret");
    expect(updated.gurobiSecret).toBe("wls_key_cloud_runner");
  });
});
