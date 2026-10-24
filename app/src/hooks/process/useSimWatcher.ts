import { useEffect } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { useSimStore } from "../../store/sim";
import type { DayLogEntry, PolicyVizEntry, SimBatchUpdate, SimFailureEntry } from "../../types";

/**
 * Starts the Rust incremental file-watcher for a simulation log file and pipes
 * batched `SimBatchUpdate` objects into the sim store as they arrive.
 */
export function useSimWatcher(logPath: string | null) {
  useEffect(() => {
    if (!logPath) return;

    let unlisten: (() => void) | undefined;

    (async () => {
      const ulBatch = await listen<SimBatchUpdate>("sim:batch_update", (event) => {
        useSimStore.getState().applyBatch(event.payload);
      });
      const ulDay = await listen<DayLogEntry>("sim:day_update", (event) => {
        useSimStore.getState().addEntry(event.payload);
      });
      const ulViz = await listen<PolicyVizEntry>("sim:policy_viz_update", (event) => {
        useSimStore.getState().addPolicyVizEntry(event.payload);
      });
      const ulFailure = await listen<SimFailureEntry>("sim:failure_update", (event) => {
        useSimStore.getState().addFailureEntry(event.payload);
      });
      unlisten = () => {
        ulBatch();
        ulDay();
        ulViz();
        ulFailure();
      };

      try {
        await invoke("start_sim_watcher", { path: logPath });
        useSimStore.getState().setWatching(true);
      } catch (err) {
        console.error("Failed to start sim watcher:", err);
      }
    })();

    return () => {
      unlisten?.();
      invoke("stop_sim_watcher").catch(() => {});
      useSimStore.getState().setWatching(false);
    };
  }, [logPath]);
}
