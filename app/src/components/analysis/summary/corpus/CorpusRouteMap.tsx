import { lazy, Suspense, useEffect, useState } from "react";

import { useCorpusSceneState, useSimulationCorpusStore } from "../../../../store/simulationCorpus";
import { corpusMapFrame } from "../../../../utils/corpus/routeFrame";

const DeckRouteMap = lazy(() => import("../../../maps/DeckRouteMap"));

function reducedMotionNow(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => reducedMotionNow());
  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduced(query.matches);
    apply();
    query.addEventListener("change", apply);
    return () => query.removeEventListener("change", apply);
  }, []);
  return reduced;
}

/** One corpus day on the shared route map. Unmounts with Simulation Summary. */
export function CorpusRouteMap() {
  const { view } = useCorpusSceneState();
  const playbackDay = useSimulationCorpusStore((state) => state.playbackDay);
  const playbackPlaying = useSimulationCorpusStore((state) => state.playbackPlaying);
  const reducedMotion = useReducedMotion();

  if (playbackDay == null || !view) return null;
  const frame = corpusMapFrame(view, playbackDay);
  if (frame.reason) {
    return (
      <section className="card space-y-2" aria-label="Corpus route map">
        <h2 className="text-sm font-semibold">Route map</h2>
        <p className="text-xs text-canvas-muted">{frame.reason}</p>
      </section>
    );
  }
  if (frame.routes.length === 0) return null;

  return (
    <section className="card space-y-2" aria-label="Corpus route map">
      <h2 className="text-sm font-semibold">Route map</h2>
      <Suspense fallback={<p className="text-xs text-canvas-muted">Loading route map…</p>}>
        <DeckRouteMap routes={frame.routes} animate={playbackPlaying && !reducedMotion} />
      </Suspense>
    </section>
  );
}
