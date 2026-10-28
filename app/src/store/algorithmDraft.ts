import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface AlgorithmDraft {
  problem: string;
  graphSize: number;
  seed: number;
  packing: string | null;
  routing: string | null;
  model: string | null;
  routeImprovement: string | null;
  acceptance: string | null;
  operators: string[];
}

function initialDraft(): AlgorithmDraft {
  return {
    problem: "ptp",
    graphSize: 5,
    seed: 1,
    packing: null,
    routing: null,
    model: null,
    routeImprovement: null,
    acceptance: null,
    operators: [],
  };
}

interface AlgorithmDraftState extends AlgorithmDraft {
  setProblem: (problem: string) => void;
  setGraphSize: (graphSize: number) => void;
  setSeed: (seed: number) => void;
  setPacking: (packing: string | null) => void;
  setRouting: (routing: string | null) => void;
  setModel: (model: string | null) => void;
  setRouteImprovement: (routeImprovement: string | null) => void;
  setAcceptance: (acceptance: string | null) => void;
  setOperators: (operators: string[]) => void;
  setDraft: (partial: Partial<AlgorithmDraft>) => void;
  clear: () => void;
}

export const useAlgorithmDraftStore = create<AlgorithmDraftState>()(
  persist(
    (set) => ({
      ...initialDraft(),
      setProblem: (problem) => set({ problem }),
      setGraphSize: (graphSize) => set({ graphSize }),
      setSeed: (seed) => set({ seed }),
      setPacking: (packing) => set({ packing }),
      setRouting: (routing) => set({ routing }),
      setModel: (model) => set({ model }),
      setRouteImprovement: (routeImprovement) => set({ routeImprovement }),
      setAcceptance: (acceptance) => set({ acceptance }),
      setOperators: (operators) => set({ operators }),
      setDraft: (partial) => set(partial),
      clear: () => set(initialDraft()),
    }),
    { name: "wsmart-algorithm-draft" }
  )
);
