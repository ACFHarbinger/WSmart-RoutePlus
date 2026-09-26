import { useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { RouteScenario } from "../data/types";

/**
 * V3 (3-D) — the 30 days of routes stacked in time. x/y are the MDS embedding
 * of the road-distance matrix; z is the day index, so the third axis is the
 * planning horizon. Lazy-loaded behind the 2-D view; never in the initial
 * bundle.
 */

const WIDTH = 10;
const HEIGHT = 10;
const DEPTH = 20;

function to3d(x: number, y: number, day: number, nDays: number): [number, number, number] {
  const z = nDays > 1 ? (day / (nDays - 1)) * DEPTH - DEPTH / 2 : 0;
  return [x * WIDTH - WIDTH / 2, -y * HEIGHT + HEIGHT / 2, z];
}

function readCssVar(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return value || fallback;
}

function Scene({ scenario, day }: { scenario: RouteScenario; day: number }) {
  const nDays = scenario.days.length;
  const depot = scenario.nodes.find((n) => n.depot);

  const colors = useMemo(
    () => ({
      route: readCssVar("--brand-route", "#4da6ff"),
      depot: readCssVar("--brand-eco", "#3dffa8"),
      faint: readCssVar("--map-line", "#5c6b80"),
    }),
    [],
  );

  const routes = useMemo(() => {
    return scenario.days.map((d, di) => ({
      points: d.tour.map((i) => {
        const n = scenario.nodes[i];
        return new THREE.Vector3(...to3d(n.x, n.y, di, nDays));
      }),
      active: di === day,
    }));
  }, [scenario, day, nDays]);

  return (
    <>
      <ambientLight intensity={0.9} />
      <directionalLight position={[10, 15, 10]} intensity={1.1} />
      <OrbitControls enablePan={false} makeDefault />

      {routes.map((r, di) => {
        if (r.points.length < 2) return null;
        return (
          <Line
            key={di}
            points={r.points}
            color={r.active ? colors.route : colors.faint}
            lineWidth={r.active ? 2 : 0.6}
            transparent
            opacity={r.active ? 1 : 0.22}
          />
        );
      })}

      {depot && (
        <Line
          points={[
            new THREE.Vector3(...to3d(depot.x, depot.y, 0, nDays)),
            new THREE.Vector3(...to3d(depot.x, depot.y, nDays - 1, nDays)),
          ]}
          color={colors.depot}
          lineWidth={1.4}
        />
      )}

      <ActiveDayMarker day={day} nDays={nDays} color={colors.route} />
    </>
  );
}

function ActiveDayMarker({
  day,
  nDays,
  color,
}: {
  day: number;
  nDays: number;
  color: string;
}) {
  const ref = useRef<THREE.Group>(null);
  useFrame(({ clock }) => {
    if (ref.current) {
      const s = 1 + 0.015 * Math.sin(clock.elapsedTime * 2);
      ref.current.scale.setScalar(s);
    }
  });
  return (
    <group ref={ref} position={[0, 0, to3d(0, 0, day, nDays)[2]]}>
      <mesh>
        <planeGeometry args={[WIDTH + 0.4, HEIGHT + 0.4]} />
        <meshBasicMaterial color={color} transparent opacity={0.05} />
      </mesh>
    </group>
  );
}

export default function RouteMap3D({ scenario }: { scenario: RouteScenario }) {
  const [day, setDay] = useState(0);
  const nDays = scenario.days.length;
  const current = scenario.days[day];

  return (
    <div>
      <div className="viz-route-controls">
        <label htmlFor="viz-route3d-day">Day</label>
        <input
          id="viz-route3d-day"
          className="viz-slider"
          type="range"
          min={0}
          max={nDays - 1}
          value={day}
          onChange={(e) => setDay(Number(e.target.value))}
          style={{ flex: 1 }}
        />
        <span className="viz-day-readout">
          Day {day + 1} / {nDays}
        </span>
      </div>
      <div style={{ width: "100%", height: 460, position: "relative" }}>
        <Canvas camera={{ position: [0, -6, 26], fov: 45 }}>
          <Scene scenario={scenario} day={day} />
        </Canvas>
      </div>
      <div className="viz-metric-strip">
        <span>
          Bins collected
          <strong>{current?.ncol ?? 0}</strong>
        </span>
        <span>
          Distance
          <strong>{current?.km.toLocaleString()} km</strong>
        </span>
        <span>
          Efficiency
          <strong>{current?.kgkm.toFixed(2)} kg/km</strong>
        </span>
      </div>
      <p className="viz-note">
        The vertical axis is the planning horizon: each horizontal layer is one day's
        route. x/y is a multidimensional-scaling embedding of the road-distance matrix,
        so bins that are far apart on the road are far apart in the view. Drag to rotate,
        scroll to zoom.
      </p>
    </div>
  );
}
