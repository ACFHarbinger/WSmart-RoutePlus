/**
 * deck.gl route map with tile basemap (§G.16), OrbitView Cartesian mode (§G.3.4),
 * and optional Hexagonal Spatial Density Clustering overlay (Owner Ruling #3, Slice V5).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { OrbitView, COORDINATE_SYSTEM } from "@deck.gl/core";
import MapGL from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import { CanvasExportButton } from "../common/CanvasExportButton";
import { FailureOverlayLegend } from "../analysis/routes/FailureOverlayLegend";
import { resolveBinPositions } from "../../utils/map/mapPositions";
import {
  buildFailureHighlights,
  buildHexDensityCells,
  buildRouteGeometry,
  buildTourDiffHighlights,
  type FailureHighlightPoint,
  type MapRoute,
  type TourDiffPoint,
} from "../../utils/map/deckRouteGeometry";
import { buildDeckLayers } from "../../utils/map/deckRouteLayers";

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

  const layers = useMemo(
    () =>
      buildDeckLayers({
        geometries,
        routes,
        animate,
        currentTime,
        cartesianMode,
        cartesianSystem,
        hexDensityOn,
        hexCells,
        failureHighlights,
        tourDiffHighlights,
      }),
    [
      geometries,
      routes,
      animate,
      currentTime,
      cartesianMode,
      cartesianSystem,
      hexDensityOn,
      hexCells,
      failureHighlights,
      tourDiffHighlights,
    ]
  );

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
