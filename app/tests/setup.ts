import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

// Tauri IPC is not available under jsdom — modules that import the API at the
// top level are fine, but any accidental invoke during a test should fail loudly
// with a recognisable message instead of an undefined-internals TypeError.
(globalThis as Record<string, unknown>).__TAURI_INTERNALS__ = {
  invoke: vi.fn(async (cmd: string) => {
    throw new Error(`Tauri command invoked in test environment: ${cmd}`);
  }),
  transformCallback: (cb: unknown) => cb,
};

// Polyfill localStorage in test environment for Zustand persist middleware
const createStorageMock = () => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = String(value);
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
    get length() {
      return Object.keys(store).length;
    },
    key: (i: number) => Object.keys(store)[i] ?? null,
  };
};

const storage = createStorageMock();
Object.defineProperty(globalThis, "localStorage", {
  value: storage,
  writable: true,
  configurable: true,
});
if (typeof window !== "undefined") {
  Object.defineProperty(window, "localStorage", {
    value: storage,
    writable: true,
    configurable: true,
  });
}
