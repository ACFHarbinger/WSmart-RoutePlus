import { useCallback, useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Compass, X } from "lucide-react";
import { useLayoutStore } from "../../store/layout";

const STEPS = [
  {
    title: "Navigate views",
    body: "Use the sidebar to switch between Monitor, Analysis, Launch, and Files. The workflow strip above each page shows where you are in the analytical flow.",
    spotlight: "sidebar",
  },
  {
    title: "Command palette",
    body: "Press Ctrl+K (or click the search icon) to fuzzy-search any view, action, or recent file. Import .wsroute bundles and jump to Simulation Summary from here.",
    spotlight: "command-palette",
  },
  {
    title: "Simulation Digital Twin",
    body: "Open a .jsonl log to watch KPIs, bin-fill strips, and route maps update day-by-day. Use the day scrubber and playback controls to replay a run.",
    spotlight: "simulation",
  },
  {
    title: "Output Browser",
    body: "Browse the simulation runs (outputs/simulations/), inspect run configs, export .wsroute bundles, or drag-drop a bundle anywhere in the app to import it.",
    spotlight: "output-browser",
  },
  {
    title: "Launch & monitor",
    body: "Spawn simulations and training jobs from the Launch section. Process Monitor streams stdout in real time and fires toast notifications on completion.",
    spotlight: "launch",
  },
];

export function GuidedTour() {
  const {
    guidedTourOpen,
    guidedTourStep,
    setGuidedTourOpen,
    setGuidedTourStep,
    setGuidedTourDismissed,
  } = useLayoutStore();
  const [spotlightRect, setSpotlightRect] = useState<DOMRect | null>(null);

  const close = useCallback(() => {
    setGuidedTourOpen(false);
    setGuidedTourStep(0);
    setSpotlightRect(null);
  }, [setGuidedTourOpen, setGuidedTourStep]);

  const finish = useCallback(() => {
    setGuidedTourDismissed(true);
    close();
  }, [setGuidedTourDismissed, close]);

  useEffect(() => {
    if (!guidedTourOpen) {
      setSpotlightRect(null);
      return;
    }

    const spotlightId = STEPS[guidedTourStep]?.spotlight;
    const el = spotlightId ? document.querySelector(`[data-tour="${spotlightId}"]`) : null;
    if (!el) {
      setSpotlightRect(null);
      return;
    }

    const update = () => setSpotlightRect(el.getBoundingClientRect());
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [guidedTourOpen, guidedTourStep]);

  if (!guidedTourOpen) return null;

  const step = STEPS[guidedTourStep];
  const isLast = guidedTourStep === STEPS.length - 1;

  return (
    <>
      {spotlightRect && (
        <div
          className="fixed z-[49] pointer-events-none rounded-lg ring-2 ring-accent-primary"
          style={{
            top: spotlightRect.top - 4,
            left: spotlightRect.left - 4,
            width: spotlightRect.width + 8,
            height: spotlightRect.height + 8,
            boxShadow: "0 0 0 9999px rgba(0, 0, 0, 0.65)",
          }}
        />
      )}

      <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4 pointer-events-none">
        <div className="card max-w-md w-full space-y-4 pointer-events-auto shadow-2xl">
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-2">
              <Compass size={16} className="text-accent-primary" />
              <h2 className="text-sm font-semibold text-gray-100">Studio guided tour</h2>
            </div>
            <button onClick={close} className="btn-ghost p-1" title="Close tour">
              <X size={14} />
            </button>
          </div>

          <p className="text-[10px] text-canvas-muted uppercase tracking-wide">
            Step {guidedTourStep + 1} of {STEPS.length}
          </p>
          <h3 className="text-sm font-medium text-gray-200">{step.title}</h3>
          <p className="text-xs text-canvas-muted leading-relaxed">{step.body}</p>

          <div className="flex gap-1">
            {STEPS.map((_, i) => (
              <span
                key={i}
                className={`h-1 flex-1 rounded-full ${
                  i === guidedTourStep ? "bg-accent-primary" : "bg-canvas-border"
                }`}
              />
            ))}
          </div>

          <div className="flex items-center justify-between pt-1">
            <button
              onClick={() => setGuidedTourStep(Math.max(0, guidedTourStep - 1))}
              disabled={guidedTourStep === 0}
              className="btn-ghost text-xs flex items-center gap-1 disabled:opacity-30"
            >
              <ChevronLeft size={12} />
              Back
            </button>
            {isLast ? (
              <button onClick={finish} className="btn-primary text-sm">
                Finish
              </button>
            ) : (
              <button
                onClick={() => setGuidedTourStep(guidedTourStep + 1)}
                className="btn-primary text-sm flex items-center gap-1"
              >
                Next
                <ChevronRight size={12} />
              </button>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
