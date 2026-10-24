import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ProcessEntry, ProcessStatus } from "../types";

interface ProcessState {
  processes: Record<string, ProcessEntry>;
  lastEventAt: number | null;
  addProcess: (entry: Omit<ProcessEntry, "logLines">) => void;
  appendLog: (id: string, line: string) => void;
  appendLogs: (id: string, lines: string[]) => void;
  updateStatus: (id: string, status: ProcessStatus, exitCode?: number) => void;
  removeProcess: (id: string) => void;
  clearCompleted: () => void;
}

const MAX_LOG_LINES = 2000;

export const useProcessStore = create<ProcessState>()(
  persist(
    (set, get) => ({
      processes: {},
      lastEventAt: null,

      addProcess: (entry) =>
        set((s) => {
          const now = Date.now();
          return {
            lastEventAt: now,
            processes: {
              ...s.processes,
              [entry.id]: { ...entry, logLines: [], lastEventAt: now },
            },
          };
        }),

      appendLogs: (id, lines) => {
        if (lines.length === 0) return;
        set((s) => {
          const proc = s.processes[id];
          if (!proc) return s;
          const now = Date.now();
          const combined =
            proc.logLines.length + lines.length <= MAX_LOG_LINES
              ? [...proc.logLines, ...lines]
              : [...proc.logLines.slice(-(MAX_LOG_LINES - lines.length)), ...lines].slice(-MAX_LOG_LINES);
          return {
            lastEventAt: now,
            processes: {
              ...s.processes,
              [id]: {
                ...proc,
                lastEventAt: now,
                logLines: combined,
              },
            },
          };
        });
      },

      appendLog: (id, line) => {
        get().appendLogs(id, [line]);
      },

      updateStatus: (id, status, exitCode) =>
        set((s) => {
          const proc = s.processes[id];
          if (!proc) return s;
          const now = Date.now();
          return {
            lastEventAt: now,
            processes: {
              ...s.processes,
              [id]: { ...proc, status, exitCode, lastEventAt: now },
            },
          };
        }),

      removeProcess: (id) =>
        set((s) => {
          const next = { ...s.processes };
          delete next[id];
          return { processes: next };
        }),

      clearCompleted: () =>
        set((s) => ({
          processes: Object.fromEntries(
            Object.entries(s.processes).filter(([, p]) => p.status === "running")
          ),
        })),
    }),
    {
      name: "wsmart-studio-processes",
      // Persist only the last 50 completed processes, stripping log lines
      // (logs are volatile and can be very large).
      partialize: (s) => ({
        processes: Object.fromEntries(
          Object.entries(s.processes)
            .filter(([, p]) => p.status !== "running")
            .slice(-50)
            .map(([id, p]) => [id, { ...p, logLines: [] }])
        ),
      }),
    }
  )
);
