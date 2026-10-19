import { describe, expect, it, beforeEach } from "vitest";
import { useCloudMonitorStore } from "../../src/store/cloudMonitor";

describe("useCloudMonitorStore (CR-15)", () => {
  beforeEach(() => {
    useCloudMonitorStore.getState().patch({
      selectedJobId: null,
      stateFilter: "all",
      searchQuery: "",
      autoRefresh: true,
      root: "outputs/cloud",
      activeTab: "tasks",
    });
  });

  it("initializes with sensible defaults", () => {
    const state = useCloudMonitorStore.getState();
    expect(state.selectedJobId).toBeNull();
    expect(state.stateFilter).toBe("all");
    expect(state.searchQuery).toBe("");
    expect(state.autoRefresh).toBe(true);
    expect(state.root).toBe("outputs/cloud");
    expect(state.activeTab).toBe("tasks");
  });

  it("updates selected job ID and active tab", () => {
    const store = useCloudMonitorStore.getState();
    store.setSelectedJobId("job-abc-123");
    store.setActiveTab("metrics");

    const updated = useCloudMonitorStore.getState();
    expect(updated.selectedJobId).toBe("job-abc-123");
    expect(updated.activeTab).toBe("metrics");
  });

  it("updates search query and state filters", () => {
    const store = useCloudMonitorStore.getState();
    store.setSearchQuery("experiment-01");
    store.setStateFilter("running");
    store.setAutoRefresh(false);

    const updated = useCloudMonitorStore.getState();
    expect(updated.searchQuery).toBe("experiment-01");
    expect(updated.stateFilter).toBe("running");
    expect(updated.autoRefresh).toBe(false);
  });
});
