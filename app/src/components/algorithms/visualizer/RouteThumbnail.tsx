interface RouteThumbnailProps {
  routes: number[][];
  coordinates?: Array<[number, number]> | null;
}

const VIEW = 100;
const PAD = 8;

function fakePoint(index: number): [number, number] {
  const angle = index * 2.399963229728653;
  return [Math.cos(angle), Math.sin(angle)];
}

function layout(
  routes: number[][],
  coordinates: Array<[number, number]> | null | undefined
): Map<number, [number, number]> {
  const ids = new Set<number>();
  for (const route of routes) {
    for (const id of route) ids.add(id);
  }
  const given = coordinates ?? null;
  const useGiven =
    given !== null && [...ids].every((id) => Array.isArray(given[id]) && given[id].length >= 2);
  const raw = new Map<number, [number, number]>();
  for (const id of ids) {
    const point = useGiven && given ? given[id] : fakePoint(id);
    if (point) raw.set(id, point);
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of raw.values()) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const points = new Map<number, [number, number]>();
  for (const [id, [x, y]] of raw) {
    const px = PAD + ((x - minX) / spanX) * (VIEW - PAD * 2);
    const py = PAD + ((y - minY) / spanY) * (VIEW - PAD * 2);
    points.set(id, [px, py]);
  }
  return points;
}

export function RouteThumbnail({ routes, coordinates }: RouteThumbnailProps) {
  if (!routes.length) {
    return <p className="text-xs text-noc-textMuted">No route recorded on this step.</p>;
  }
  const points = layout(routes, coordinates);
  return (
    <svg
      viewBox={`0 0 ${VIEW} ${VIEW}`}
      role="img"
      aria-label="Route thumbnail"
      className="w-full h-28 bg-noc-deep border border-noc-border rounded-lg"
    >
      {routes.map((route, routeIndex) => {
        const pts = route
          .map((id) => points.get(id))
          .filter((point): point is [number, number] => point != null)
          .map(([x, y]) => `${x},${y}`)
          .join(" ");
        return (
          <polyline
            key={`route-${routeIndex}`}
            points={pts}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.6}
            className={routeIndex % 2 === 0 ? "text-noc-cyan" : "text-noc-indigoText"}
          />
        );
      })}
      {[...points].map(([id, [x, y]]) => (
        <circle
          key={`node-${id}`}
          cx={x}
          cy={y}
          r={id === 0 ? 2.6 : 1.8}
          fill="currentColor"
          className={id === 0 ? "text-noc-cyan" : "text-noc-textMuted"}
        />
      ))}
    </svg>
  );
}
