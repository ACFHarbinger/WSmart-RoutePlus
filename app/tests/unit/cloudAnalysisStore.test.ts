import { describe, expect, it, beforeEach } from "vitest";
import { useCloudAnalysisStore } from "../../src/store/cloudAnalysis";

describe("useCloudAnalysisStore (CR-16)", () => {
  beforeEach(() => {
    useCloudAnalysisStore.getState().reset();
  });

  it("initializes with default state", () => {
    const state = useCloudAnalysisStore.getState();
    expect(state.selectedJobId).toBeNull();
    expect(state.activeTab).toBe("overview");
    expect(state.taskFilter).toBe("all");
    expect(state.searchQuery).toBe("");
    expect(state.sortBy).toBe("task_id");
    expect(state.sortDirection).toBe("asc");
  });

  it("updates selected job ID and active tab", () => {
    const { setSelectedJobId, setActiveTab } = useCloudAnalysisStore.getState();
    setSelectedJobId("job-xyz-999");
    setActiveTab("waterfall");

    const updated = useCloudAnalysisStore.getState();
    expect(updated.selectedJobId).toBe("job-xyz-999");
    expect(updated.activeTab).toBe("waterfall");
  });

  it("updates task filters and search query", () => {
    const { setTaskFilter, setSearchQuery } = useCloudAnalysisStore.getState();
    setTaskFilter("failed");
    setSearchQuery("bpc");

    const updated = useCloudAnalysisStore.getState();
    expect(updated.taskFilter).toBe("failed");
    expect(updated.searchQuery).toBe("bpc");
  });

  it("toggles sort direction when selecting same column", () => {
    const { setSort } = useCloudAnalysisStore.getState();

    // Default is task_id asc. Setting task_id again should toggle to desc
    setSort("task_id");
    expect(useCloudAnalysisStore.getState().sortDirection).toBe("desc");

    // Setting another column should switch column with default or specified direction
    setSort("duration", "desc");
    expect(useCloudAnalysisStore.getState().sortBy).toBe("duration");
    expect(useCloudAnalysisStore.getState().sortDirection).toBe("desc");
  });

  it("resets back to initial state", () => {
    const store = useCloudAnalysisStore.getState();
    store.setSelectedJobId("job-123");
    store.setActiveTab("cost");
    store.reset();

    const resetState = useCloudAnalysisStore.getState();
    expect(resetState.selectedJobId).toBeNull();
    expect(resetState.activeTab).toBe("overview");
  });
});
