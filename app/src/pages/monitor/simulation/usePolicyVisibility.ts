import { useCallback, useMemo, useState } from "react";

export function usePolicyVisibility(policies: string[]) {
  const [chartPolicies, setChartPolicies] = useState<string[]>([]);
  const activeChartPolicies = useMemo(
    () => (chartPolicies.length > 0 ? chartPolicies : policies),
    [chartPolicies, policies]
  );

  const toggleChartPolicy = useCallback(
    (p: string) => {
      setChartPolicies((prev) => {
        const active = prev.length > 0 ? prev : policies;
        if (active.includes(p)) {
          const next = active.filter((x) => x !== p);
          return next.length === 0 ? policies : next;
        }
        return [...active, p];
      });
    },
    [policies]
  );

  const [mapPolicies, setMapPolicies] = useState<string[]>([]);
  const activeMapPolicies = useMemo(
    () => (mapPolicies.length > 0 ? mapPolicies : policies),
    [mapPolicies, policies]
  );

  const toggleMapPolicy = useCallback(
    (p: string) => {
      setMapPolicies((prev) => {
        const active = prev.length > 0 ? prev : policies;
        if (active.includes(p)) {
          const next = active.filter((x) => x !== p);
          return next.length === 0 ? policies : next;
        }
        return [...active, p];
      });
    },
    [policies]
  );

  return {
    activeChartPolicies,
    toggleChartPolicy,
    activeMapPolicies,
    toggleMapPolicy,
    setMapPolicies,
  };
}
