/**
 * deck.gl route map with tile basemap (§G.16), OrbitView Cartesian mode (§G.3.4),
 * and optional Hexagonal Spatial Density Clustering overlay (Owner Ruling #3, Slice V5).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { OrbitView, COORDINATE_SYSTEM } from "@deck.gl/core";
import { PathLayer, PolygonLayer, ScatterplotLayer } from "@deck.gl/layers";
import { TripsLayer } from "@deck.gl/geo-layers";
import MapGL from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import { CanvasExportButton } from "../common/CanvasExportButton";
import { FailureOverlayLegend } from "../analysis/routes/FailureOverlayLegend";
import { resolveBinPositions } from "../../utils/map/mapPositions";
import { FAILURE_RGB, TOUR_DIFF_RGB } from "../../utils/map/routeFailureOverlay";
import {
  buildFailureHighlights,
  buildHexDensityCells,
  buildRouteGeometry,
  buildTourDiffHighlights,
  fillRgb,
  stopRadius,
  TRIP_SEGMENT_MS,
  type FailureHighlightPoint,
  type HexDensityCell,
  type MapRoute,
  type TourDiffPoint,
  type VehicleTourStop,
} from "../../utils/map/deckRouteGeometry";

export type { MapRoute };

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

interface Props {
  routes: MapRoute[];
  animate?: boolean;
  playbackSpeed?: number;
  showFailureOverlay?: boolean;
  showRouteDiff?: boolean;
  showHexDensity?: boolean;
}

export default function DeckRouteMap({
  routes,
  animate = false,
  playbackSpeed = 1,
  showFailureOverlay = true,
  showRouteDiff = false,
  showHexDensity: initialHexDensity = false,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [pitch3d, setPitch3d] = useState(false);
  const [hexDensityOn, setHexDensityOn] = useState(initialHexDensity);
  const [mapView, setMapView] = useState({
    longitude: 0,
    latitude: 0,
    zoom: 13,
    pitch: 0,
    bearing: 0,
  });
  const [orbitView, setOrbitView] = useState({
    target: [0, 0, 0] as [number, number, number],
    rotationX: 35,
    rotationOrbit: 0,
    zoom: 1.2,
  });

  useEffect(() => {
    setHexDensityOn(initialHexDensity);
  }, [initialHexDensity]);

  const geometries = useMemo(
    () => routes.map((r) => ({ route: r, ...buildRouteGeometry(r.data) })),
    [routes]
  );

  const hasGeo = geometries.some((g) => g.hasGeo);
  const hasBins = geometries.some((g) => g.hasBins);
  const cartesianMode = hasBins && !hasGeo;

  const exportFilename = cartesianMode ? "route-map-orbit.png" : "route-map-tile.png";
  const getExportCanvas = useCallback(
    () => containerRef.current?.querySelector("canvas") ?? null,
    []
  );
  const maxTripLength = Math.max(...geometries.map((g) => g.tripLength), 0);
  const cartesianSystem = cartesianMode ? COORDINATE_SYSTEM.CARTESIAN : COORDINATE_SYSTEM.LNGLAT;

  const centerView = useMemo(() => {
    const centers = geometries.filter((g) => g.hasGeo).map((g) => g.center);
    if (!centers.length) {
      return { longitude: 0, latitude: 0, zoom: 12 };
    }
    const longitude = centers.reduce((a, c) => a + c.longitude, 0) / centers.length;
    const latitude = centers.reduce((a, c) => a + c.latitude, 0) / centers.length;
    return { longitude, latitude, zoom: 13 };
  }, [geometries]);

  useEffect(() => {
    setMapView((v) => ({
      ...v,
      ...centerView,
      pitch: pitch3d ? 45 : 0,
    }));
  }, [centerView, pitch3d]);

  const tripSignature = geometries.map((g) => g.tripPath.length).join(",");

  useEffect(() => {
    setCurrentTime(0);
  }, [tripSignature, animate]);

  useEffect(() => {
    if (!animate || maxTripLength <= 0) return;

    let frame: number;
    let last = performance.now();

    const tick = (now: number) => {
      const dt = (now - last) * playbackSpeed;
      last = now;
      setCurrentTime((t) => {
        const next = t + dt;
        return next > maxTripLength ? 0 : next;
      });
      frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [animate, playbackSpeed, maxTripLength]);

  const pos3d = (p: [number, number], z = 0): [number, number, number] => [p[0], p[1], z];

  const hexCells = useMemo(
    () => (hexDensityOn ? buildHexDensityCells(routes[0]?.data, 10) : []),
    [hexDensityOn, routes]
  );

  const failureHighlights = useMemo(() => {
    if (!showFailureOverlay) return [] as FailureHighlightPoint[];
    const base = geometries[0];
    if (!base?.hasBins) return [];
    const coords = routes.find((r) => r.data.all_bin_coords?.length)?.data.all_bin_coords;
    if (!coords?.length) return [];
    const { posById } = resolveBinPositions(coords);
    const merged: FailureHighlightPoint[] = [];
    const seen = new Set<string>();

    for (const route of routes) {
      for (const point of buildFailureHighlights(posById, route.data.failure_analysis)) {
        const key = `${point.kind}:${point.position.join(",")}`;
        if (seen.has(key)) continue;
        seen.add(key);
        merged.push(point);
      }
    }
    return merged;
  }, [geometries, routes, showFailureOverlay]);

  const tourDiffHighlights = useMemo(() => {
    if (!showRouteDiff || routes.length !== 2) return [] as TourDiffPoint[];
    const first = routes[0]?.data;
    const second = routes[1]?.data;
    if (!first?.all_bin_coords?.length || !second?.all_bin_coords?.length) return [];
    const { posById } = resolveBinPositions(first.all_bin_coords);
    return buildTourDiffHighlights(first, second, posById);
  }, [routes, showRouteDiff]);

  const showFailureLegend = showFailureOverlay && failureHighlights.length > 0;
  const showDiffLegend = showRouteDiff && tourDiffHighlights.length > 0 && routes.length === 2;

  const layers = useMemo(() => {
    const result = [];
    const base = geometries[0];

    if (hexDensityOn && hexCells.length > 0) {
      result.push(
        new PolygonLayer({
          id: "hex-density-clusters",
          data: hexCells,
          coordinateSystem: cartesianSystem,
          getPolygon: (d: HexDensityCell) =>
            cartesianMode ? d.polygon.map((p) => pos3d(p, 0.005)) : d.polygon,
          getFillColor: (d: HexDensityCell) =>
            [...fillRgb(d.maxFill), 85] as [number, number, number, number],
          getLineColor: (d: HexDensityCell) =>
            [...fillRgb(d.maxFill), 200] as [number, number, number, number],
          stroked: true,
          filled: true,
          lineWidthMinPixels: 1,
        })
      );
    }

    if (base?.idlePoints.length) {
      result.push(
        new ScatterplotLayer({
          id: "idle-bins",
          data: base.idlePoints,
          coordinateSystem: cartesianSystem,
          getPosition: (d: { position: [number, number] }) =>
            cartesianMode ? pos3d(d.position, 0) : d.position,
          getFillColor: [75, 85, 99, 120],
          getRadius: 40,
          radiusMinPixels: 3,
          radiusMaxPixels: 8,
        })
      );
    }

    if (base?.depotPoint) {
      result.push(
        new ScatterplotLayer({
          id: "depot",
          data: [base.depotPoint],
          coordinateSystem: cartesianSystem,
          getPosition: (d: { position: [number, number] }) =>
            cartesianMode ? pos3d(d.position, 0.05) : d.position,
          getFillColor: [251, 191, 36, 255],
          getLineColor: [255, 255, 255, 200],
          stroked: true,
          getRadius: 90,
          radiusMinPixels: 8,
          radiusMaxPixels: 14,
        })
      );
    }

    for (const g of geometries) {
      const policyRgb: [number, number, number, number] = [...g.route.color, 220];
      const id = g.route.id;
      const vehicleSegments =
        g.vehicles.length > 0
          ? g.vehicles
          : [{ vehicleId: 0, color: g.route.color, path: g.path, tripPath: g.tripPath, tripLength: g.tripLength }];

      for (const vehicle of vehicleSegments) {
        const rgb: [number, number, number, number] =
          vehicleSegments.length > 1 ? [...vehicle.color, 230] : policyRgb;

        if (animate && vehicle.tripPath.length >= 2) {
          const tripData = cartesianMode
            ? [
                {
                  path: vehicle.tripPath.map((p) => pos3d([p[0], p[1]], 0.02)),
                  timestamps: vehicle.tripPath.map((p) => p[2]),
                },
              ]
            : [{ path: vehicle.tripPath }];
          result.push(
            new TripsLayer({
              id: `trips-${id}-v${vehicle.vehicleId}`,
              data: tripData,
              coordinateSystem: cartesianSystem,
              getPath: (d: { path: [number, number, number][]; timestamps?: number[] }) => d.path,
              getTimestamps: (d: { path: [number, number, number][]; timestamps?: number[] }) =>
                d.timestamps ?? d.path.map((p) => p[2]),
              getColor: rgb,
              currentTime,
              trailLength: Math.max(vehicle.tripLength * 0.6, TRIP_SEGMENT_MS * 2),
              capRounded: true,
              fadeTrail: true,
              widthMinPixels: 3,
            })
          );
        } else if (vehicle.path.length >= 2) {
          const pathData = cartesianMode
            ? [{ path: vehicle.path.map((p) => pos3d(p, 0.02)) }]
            : [{ path: vehicle.path }];
          result.push(
            new PathLayer({
              id: `path-${id}-v${vehicle.vehicleId}`,
              data: pathData,
              coordinateSystem: cartesianSystem,
              getPath: (d: { path: [number, number][] | [number, number, number][] }) => d.path,
              getColor: rgb,
              getWidth: vehicleSegments.length > 1 || routes.length > 1 ? 3 : 4,
              widthMinPixels: 2,
            })
          );
        }
      }

      const multiVehicleStops = g.vehicles.length > 1 && g.vehicleTourStops.length > 0;
      const stopGroups = multiVehicleStops
        ? g.vehicleTourStops.reduce<Record<number, VehicleTourStop[]>>((acc, stop) => {
            if (!acc[stop.vehicleId]) acc[stop.vehicleId] = [];
            acc[stop.vehicleId].push(stop);
            return acc;
          }, {})
        : null;

      if (stopGroups) {
        for (const [vehicleId, stops] of Object.entries(stopGroups)) {
          result.push(
            new ScatterplotLayer({
              id: `stops-${id}-v${vehicleId}`,
              data: stops,
              coordinateSystem: cartesianSystem,
              getPosition: (d: VehicleTourStop) =>
                cartesianMode ? pos3d(d.position, d.fill * 0.003) : d.position,
              getFillColor: (d: VehicleTourStop) =>
                [...fillRgb(d.fill), 230] as [number, number, number, number],
              getLineColor: (d: VehicleTourStop) => [...d.color, 255] as [number, number, number, number],
              stroked: true,
              getRadius: (d: VehicleTourStop) => stopRadius(d.fill, d.collected, routes.length > 1),
              radiusMinPixels: 4,
              radiusMaxPixels: 14,
            })
          );
        }
      } else if (g.tourPoints.length) {
        result.push(
          new ScatterplotLayer({
            id: `stops-${id}`,
            data: g.tourPoints,
            coordinateSystem: cartesianSystem,
            getPosition: (d: { position: [number, number]; fill: number }) =>
              cartesianMode ? pos3d(d.position, d.fill * 0.003) : d.position,
            getFillColor: (d: { fill: number }) => [...fillRgb(d.fill), 230] as [number, number, number, number],
            getLineColor: [...g.route.color, 255],
            stroked: routes.length > 1,
            getRadius: (d: { fill: number; collected: number }) =>
              stopRadius(d.fill, d.collected, routes.length > 1),
            radiusMinPixels: 4,
            radiusMaxPixels: 14,
          })
        );
      }
    }

    if (failureHighlights.length) {
      const overflowPts = failureHighlights.filter((p) => p.kind === "overflow");
      const skippedPts = failureHighlights.filter((p) => p.kind === "skipped");

      if (overflowPts.length) {
        result.push(
          new ScatterplotLayer({
            id: "failure-overflow",
            data: overflowPts,
            coordinateSystem: cartesianSystem,
            getPosition: (d: FailureHighlightPoint) =>
              cartesianMode ? pos3d(d.position, 0.08) : d.position,
            getFillColor: [...FAILURE_RGB.overflow, 255] as [number, number, number, number],
            getLineColor: [255, 255, 255, 255],
            stroked: true,
            getRadius: 95,
            radiusMinPixels: 7,
            radiusMaxPixels: 16,
          })
        );
      }

      if (skippedPts.length) {
        result.push(
          new ScatterplotLayer({
            id: "failure-skipped",
            data: skippedPts,
            coordinateSystem: cartesianSystem,
            getPosition: (d: FailureHighlightPoint) =>
              cartesianMode ? pos3d(d.position, 0.06) : d.position,
            getFillColor: [...FAILURE_RGB.skipped, 240] as [number, number, number, number],
            getLineColor: [255, 255, 255, 220],
            stroked: true,
            getRadius: 80,
            radiusMinPixels: 6,
            radiusMaxPixels: 14,
          })
        );
      }
    }

    if (tourDiffHighlights.length) {
      const onlyFirst = tourDiffHighlights.filter((p) => p.side === "onlyFirst");
      const onlySecond = tourDiffHighlights.filter((p) => p.side === "onlySecond");

      if (onlyFirst.length) {
        result.push(
          new ScatterplotLayer({
            id: "tour-diff-first",
            data: onlyFirst,
            coordinateSystem: cartesianSystem,
            getPosition: (d: TourDiffPoint) =>
              cartesianMode ? pos3d(d.position, 0.1) : d.position,
            getFillColor: [0, 0, 0, 0],
            getLineColor: [...TOUR_DIFF_RGB.onlyFirst, 255] as [number, number, number, number],
            stroked: true,
            filled: false,
            getRadius: 110,
            radiusMinPixels: 8,
            radiusMaxPixels: 18,
            lineWidthMinPixels: 2,
          })
        );
      }

      if (onlySecond.length) {
        result.push(
          new ScatterplotLayer({
            id: "tour-diff-second",
            data: onlySecond,
            coordinateSystem: cartesianSystem,
            getPosition: (d: TourDiffPoint) =>
              cartesianMode ? pos3d(d.position, 0.1) : d.position,
            getFillColor: [0, 0, 0, 0],
            getLineColor: [...TOUR_DIFF_RGB.onlySecond, 255] as [number, number, number, number],
            stroked: true,
            filled: false,
            getRadius: 110,
            radiusMinPixels: 8,
            radiusMaxPixels: 18,
            lineWidthMinPixels: 2,
          })
        );
      }
    }

    return result;
  }, [
    geometries,
    animate,
    currentTime,
    routes.length,
    cartesianMode,
    cartesianSystem,
    failureHighlights,
    tourDiffHighlights,
    hexDensityOn,
    hexCells,
  ]);

  if (!hasBins) {
    return (
      <p className="text-xs text-canvas-muted py-4 text-center">
        Load a simulation log with bin coordinates to view the route map.
      </p>
    );
  }

  return (
    <div
      ref={containerRef}
      className="relative h-[320px] rounded-lg overflow-hidden border border-canvas-border"
    >
      {cartesianMode ? (
        <DeckGL
          views={new OrbitView({ orbitAxis: "Z" })}
          viewState={orbitView}
          onViewStateChange={({ viewState }) => {
            if ("rotationOrbit" in viewState) {
              setOrbitView({
                target: (viewState.target as [number, number, number]) ?? [0, 0, 0],
                rotationX: viewState.rotationX ?? orbitView.rotationX,
                rotationOrbit: viewState.rotationOrbit ?? orbitView.rotationOrbit,
                zoom: viewState.zoom ?? orbitView.zoom,
              });
            }
          }}
          controller
          layers={layers}
        />
      ) : (
        <DeckGL
          viewState={mapView}
          onViewStateChange={({ viewState }) => {
            if ("latitude" in viewState && "longitude" in viewState) {
              setMapView({
                longitude: viewState.longitude,
                latitude: viewState.latitude,
                zoom: viewState.zoom ?? mapView.zoom,
                pitch: viewState.pitch ?? mapView.pitch,
                bearing: viewState.bearing ?? mapView.bearing,
              });
            }
          }}
          controller
          layers={layers}
        >
          <MapGL mapStyle={MAP_STYLE} />
        </DeckGL>
      )}
      <div className="absolute top-2 right-2 flex items-center gap-1">
        <button
          type="button"
          onClick={() => setHexDensityOn((v) => !v)}
          className={`btn-ghost text-xs px-1.5 py-0.5 bg-black/50 rounded ${
            hexDensityOn ? "text-accent-secondary" : ""
          }`}
          title="Toggle Hexagonal Spatial Density Clustering overlay"
        >
          {hexDensityOn ? "Hex Cluster (on)" : "Hex Cluster (off)"}
        </button>
        {cartesianMode ? (
          <span className="text-[10px] px-1.5 py-0.5 bg-black/50 rounded text-accent-secondary">
            OrbitView
          </span>
        ) : (
          <button
            type="button"
            onClick={() => setPitch3d((v) => !v)}
            className={`btn-ghost text-xs px-1.5 py-0.5 bg-black/50 rounded ${
              pitch3d ? "text-accent-secondary" : ""
            }`}
            title="Toggle 3D pitch"
          >
            {pitch3d ? "3D (on)" : "3D (off)"}
          </button>
        )}
        <CanvasExportButton
          canvas={getExportCanvas}
          filename={exportFilename}
          size={11}
          className="bg-black/50 rounded px-1.5 py-0.5"
        />
      </div>
      {(showFailureLegend || showDiffLegend) && (
        <div className="absolute bottom-2 right-2 max-w-[55%] rounded bg-black/55 px-2 py-1">
          <FailureOverlayLegend
            showOverflow={failureHighlights.some((p) => p.kind === "overflow")}
            showSkipped={failureHighlights.some((p) => p.kind === "skipped")}
            showTourDiff={showDiffLegend}
            tourDiffLabels={
              showDiffLegend ? [routes[0].label, routes[1].label] : undefined
            }
          />
        </div>
      )}
      {(routes.length > 1 || geometries.some((g) => g.vehicles.length > 1)) && (
        <div className="absolute bottom-2 left-2 flex flex-wrap gap-1.5 max-w-[90%]">
          {routes.map((r) => {
            const geom = geometries.find((g) => g.route.id === r.id);
            if (geom && geom.vehicles.length > 1) {
              return geom.vehicles.map((v) => (
                <span
                  key={`${r.id}-v${v.vehicleId}`}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-black/50 text-gray-200 font-mono"
                  style={{ borderLeft: `3px solid rgb(${v.color.join(",")})` }}
                >
                  {r.label} · V{v.vehicleId + 1}
                </span>
              ));
            }
            return (
              <span
                key={r.id}
                className="text-[10px] px-1.5 py-0.5 rounded bg-black/50 text-gray-200 font-mono"
                style={{ borderLeft: `3px solid rgb(${r.color.join(",")})` }}
              >
                {r.label}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
