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
      className="flex items-center gap-0.5 px-5 py-1.5 bg-canvas-bg border-b border-canvas-border overflow-x-auto"
      aria-label="Analytical workflow"
    >
      {WORKFLOW_STEPS.map((step, i) => {
        const active = mode === step.mode;
        return (
          <span key={step.mode} className="flex items-center shrink-0">
            {i > 0 && (
              <ChevronRight size={10} className="text-canvas-muted mx-0.5" />
            )}
            <button
              type="button"
              onClick={() => setMode(step.mode)}
              className={`text-[10px] px-2 py-0.5 rounded-full transition-colors whitespace-nowrap ${
                active
                  ? "bg-accent-primary/25 text-accent-secondary font-medium"
                  : "text-canvas-muted hover:text-canvas-text hover:bg-canvas-hover"
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
