import { beforeEach, describe, expect, it, vi } from "vitest";

const invoke = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: (...args: unknown[]) => invoke(...args) }));

import { syncCloseToBackground } from "../../src/utils/app/backgroundMode";
import { useAppStore } from "../../src/store/app";

describe("background mode", () => {
  beforeEach(() => {
    invoke.mockReset();
  });

  it("returns the mode the backend applied", async () => {
    invoke.mockResolvedValue(false);
    expect(await syncCloseToBackground(true)).toBe(false);
    expect(invoke).toHaveBeenCalledWith("set_close_to_background", { enabled: true });
  });

  it("reports off when there is no Tauri backend", async () => {
    invoke.mockImplementation(async () => {
      throw new Error("no backend");
    });
    expect(await syncCloseToBackground(true)).toBe(false);
  });

  it("is on by default and the setter tells the backend", () => {
    invoke.mockResolvedValue(true);
    expect(useAppStore.getState().closeToBackground).toBe(true);
    useAppStore.getState().setCloseToBackground(false);
    expect(useAppStore.getState().closeToBackground).toBe(false);
    expect(invoke).toHaveBeenCalledWith("set_close_to_background", { enabled: false });
  });
});
