/**
 * Analytical workflow strip — Run → Watch → Read → Query → Write + Files (Slice V4).
 */
import { ChevronRight } from "lucide-react";
import { useAppStore } from "../../store/app";
import type { AppMode } from "../../types";

export const WORKFLOW_STEPS: Array<{ label: string; group: string; mode: AppMode }> = [
  { label: "Run: Sim Launcher", group: "Run", mode: "sim_launcher" },
  { label: "Watch: Digital Twin", group: "Watch", mode: "simulation" },
  { label: "Read: Summary", group: "Read", mode: "simulation_summary" },
  { label: "Read: Benchmark", group: "Read", mode: "benchmark" },
  { label: "Query: OLAP", group: "Query", mode: "olap_explorer" },
  { label: "Write: Report Studio", group: "Write", mode: "report_studio" },
  { label: "Files: Outputs", group: "Files", mode: "output_browser" },
];

export function WorkflowNav() {
  const { mode, setMode } = useAppStore();

  return (
    <nav
      className="flex items-center gap-1 px-4 py-1.5 bg-noc-deep border-b border-noc-border overflow-x-auto select-none"
      aria-label="Analytical workflow"
    >
      {WORKFLOW_STEPS.map((step, i) => {
        const active = mode === step.mode;
        return (
          <span key={step.mode} className="flex items-center shrink-0">
            {i > 0 && (
              <ChevronRight size={10} className="text-noc-textMuted mx-0.5 opacity-60" />
            )}
            <button
              type="button"
              onClick={() => setMode(step.mode)}
              className={`text-[11px] px-2.5 py-0.5 rounded-full transition-colors whitespace-nowrap font-medium ${
                active
                  ? "bg-noc-card border border-noc-cyan/40 text-noc-cyan font-semibold shadow-sm"
                  : "text-noc-textMuted hover:text-noc-textPrimary hover:bg-canvas-hover"
              }`}
            >
              {step.label}
            </button>
          </span>
        );
      })}
    </nav>
  );
}
