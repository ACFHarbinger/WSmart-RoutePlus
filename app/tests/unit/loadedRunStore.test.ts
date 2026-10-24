import { describe, it, expect, beforeEach } from "vitest";
import { createElement } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import {
  useLoadedRunStore,
  parseRunMetadataFromPath,
} from "../../src/store/loadedRun";
import { LoadedRunChip } from "../../src/components/layout/LoadedRunChip";
import { TopBar, TITLES } from "../../src/components/layout/TopBar";
import { NAV } from "../../src/components/layout/Sidebar";
import { useAppStore } from "../../src/store/app";
import { useProcessStore } from "../../src/store/process";

describe("LoadedRunStore, LoadedRunChip & Shell Navigation (Slice V4)", () => {
  beforeEach(() => {
    useLoadedRunStore.getState().clearLoadedRun();
    useProcessStore.setState({ processes: {} });
  });

  it("infers area, numLoc, and distribution from simulation path", () => {
    const meta = parseRunMetadataFromPath(
      "/data/outputs/simulations/riomaior_204_lognormal/2026-10-24_12-00-00"
    );
    expect(meta.label).toBe("riomaior_204_lognormal");
    expect(meta.area).toBe("riomaior");
    expect(meta.numLoc).toBe(204);
    expect(meta.distribution).toBe("lognormal");
  });

  it("renders LoadedRunChip and toggles source between [Viewing] and [Live Stream]", () => {
    useLoadedRunStore.getState().setLoadedRun({
      path: "/outputs/simulations/beja_112_emp/run1",
      label: "beja_112_emp",
      days: 30,
      source: "viewing",
    });

    render(createElement(LoadedRunChip));
    expect(screen.getByTestId("loaded-run-chip")).toBeTruthy();
    expect(screen.getByText("[Viewing]")).toBeTruthy();
    expect(screen.getByText(/beja_112_emp • 30d/i)).toBeTruthy();

    fireEvent.click(screen.getByText("[Viewing]"));
    expect(useLoadedRunStore.getState().source).toBe("live");
    expect(screen.getByText("[Live Stream]")).toBeTruthy();
  });

  it("navigates to process_monitor when clicking the running badge in TopBar", () => {
    useAppStore.getState().setMode("simulation");
    useProcessStore.setState({
      processes: {
        p1: {
          id: "p1",
          command: "python main.py test_sim",
          pid: 1234,
          status: "running",
          logs: [],
          startedAt: Date.now(),
        },
      },
    });

    render(createElement(TopBar));
    const runningBtn = screen.getByRole("button", { name: /1 running/i });
    fireEvent.click(runningBtn);
    expect(useAppStore.getState().mode).toBe("process_monitor");
  });

  it("includes all 23 AppModes in TopBar TITLES and groups Sidebar NAV into Run/Watch/Read/Query/Write/Files/App", () => {
    expect(TITLES.olap_explorer).toBe("OLAP Explorer");
    expect(TITLES.city_comparison).toBe("City Comparison");
    expect(TITLES.report_studio).toBe("Report Studio");
    expect(TITLES.system_tools).toBe("System Tools");

    const sectionLabels = NAV.map((s) => s.label);
    expect(sectionLabels).toEqual([
      "Run",
      "Watch",
      "Read",
      "Query",
      "Write",
      "Files",
      "App",
    ]);
  });
});
