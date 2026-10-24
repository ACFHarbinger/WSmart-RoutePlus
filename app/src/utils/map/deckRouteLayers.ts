import { PathLayer, PolygonLayer, ScatterplotLayer } from "@deck.gl/layers";
import { TripsLayer } from "@deck.gl/geo-layers";
import { FAILURE_RGB, TOUR_DIFF_RGB } from "./routeFailureOverlay";
import {
  fillRgb,
  stopRadius,
  TRIP_SEGMENT_MS,
  type FailureHighlightPoint,
  type HexDensityCell,
  type MapRoute,
  type TourDiffPoint,
  type VehicleTourStop,
} from "./deckRouteGeometry";

export const pos3d = (p: [number, number], z = 0): [number, number, number] => [
  p[0],
  p[1],
  z,
];

export interface DeckLayersOptions {
  geometries: any[];
  routes: MapRoute[];
  animate: boolean;
  currentTime: number;
  cartesianMode: boolean;
  cartesianSystem: any;
  hexDensityOn: boolean;
  hexCells: HexDensityCell[];
  failureHighlights: FailureHighlightPoint[];
  tourDiffHighlights: TourDiffPoint[];
}

export function buildDeckLayers({
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
}: DeckLayersOptions) {
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
    const policyRgb: [number, number, number, number] = [
      g.route.color[0],
      g.route.color[1],
      g.route.color[2],
      220,
    ];
    const id = g.route.id;
    const vehicleSegments =
      g.vehicles.length > 0
        ? g.vehicles
        : [
            {
              vehicleId: 0,
              color: g.route.color,
              path: g.path,
              tripPath: g.tripPath,
              tripLength: g.tripLength,
            },
          ];

    for (const vehicle of vehicleSegments) {
      const rgb: [number, number, number, number] =
        vehicleSegments.length > 1
          ? [vehicle.color[0], vehicle.color[1], vehicle.color[2], 230]
          : policyRgb;

      if (animate && vehicle.tripPath.length >= 2) {
        const tripData = cartesianMode
          ? [
              {
                path: vehicle.tripPath.map((p: [number, number, number]) =>
                  pos3d([p[0], p[1]], 0.02)
                ),
                timestamps: vehicle.tripPath.map(
                  (p: [number, number, number]) => p[2]
                ),
              },
            ]
          : [{ path: vehicle.tripPath }];
        result.push(
          new TripsLayer({
            id: `trips-${id}-v${vehicle.vehicleId}`,
            data: tripData,
            coordinateSystem: cartesianSystem,
            getPath: (d: { path: [number, number, number][]; timestamps?: number[] }) =>
              d.path,
            getTimestamps: (d: {
              path: [number, number, number][];
              timestamps?: number[];
            }) => d.timestamps ?? d.path.map((p) => p[2]),
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
          ? [{ path: vehicle.path.map((p: [number, number]) => pos3d(p, 0.02)) }]
          : [{ path: vehicle.path }];
        result.push(
          new PathLayer({
            id: `path-${id}-v${vehicle.vehicleId}`,
            data: pathData,
            coordinateSystem: cartesianSystem,
            getPath: (d: {
              path: [number, number][] | [number, number, number][];
            }) => d.path,
            getColor: rgb,
            getWidth: vehicleSegments.length > 1 || routes.length > 1 ? 3 : 4,
            widthMinPixels: 2,
          })
        );
      }
    }

    const multiVehicleStops =
      g.vehicles.length > 1 && g.vehicleTourStops.length > 0;
    const stopGroups = multiVehicleStops
      ? g.vehicleTourStops.reduce(
          (acc: Record<number, VehicleTourStop[]>, stop: VehicleTourStop) => {
            if (!acc[stop.vehicleId]) acc[stop.vehicleId] = [];
            acc[stop.vehicleId].push(stop);
            return acc;
          },
          {}
        )
      : null;

    if (stopGroups) {
      for (const [vehicleId, stops] of Object.entries(stopGroups)) {
        result.push(
          new ScatterplotLayer({
            id: `stops-${id}-v${vehicleId}`,
            data: stops as VehicleTourStop[],
            coordinateSystem: cartesianSystem,
            getPosition: (d: VehicleTourStop) =>
              cartesianMode ? pos3d(d.position, d.fill * 0.003) : d.position,
            getFillColor: (d: VehicleTourStop) =>
              [...fillRgb(d.fill), 230] as [number, number, number, number],
            getLineColor: (d: VehicleTourStop) =>
              [d.color[0], d.color[1], d.color[2], 255] as [
                number,
                number,
                number,
                number,
              ],
            stroked: true,
            getRadius: (d: VehicleTourStop) =>
              stopRadius(d.fill, d.collected, routes.length > 1),
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
          getFillColor: (d: { fill: number }) =>
            [...fillRgb(d.fill), 230] as [number, number, number, number],
          getLineColor: [
            g.route.color[0],
            g.route.color[1],
            g.route.color[2],
            255,
          ] as [number, number, number, number],
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
          getFillColor: [...FAILURE_RGB.overflow, 255] as [
            number,
            number,
            number,
            number,
          ],
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
          getFillColor: [...FAILURE_RGB.skipped, 240] as [
            number,
            number,
            number,
            number,
          ],
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
    const onlySecond = tourDiffHighlights.filter(
      (p) => p.side === "onlySecond"
    );

    if (onlyFirst.length) {
      result.push(
        new ScatterplotLayer({
          id: "tour-diff-first",
          data: onlyFirst,
          coordinateSystem: cartesianSystem,
          getPosition: (d: TourDiffPoint) =>
            cartesianMode ? pos3d(d.position, 0.1) : d.position,
          getFillColor: [0, 0, 0, 0],
          getLineColor: [...TOUR_DIFF_RGB.onlyFirst, 255] as [
            number,
            number,
            number,
            number,
          ],
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
          getLineColor: [...TOUR_DIFF_RGB.onlySecond, 255] as [
            number,
            number,
            number,
            number,
          ],
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
}
