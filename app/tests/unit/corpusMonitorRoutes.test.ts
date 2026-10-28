import { describe, expect, it } from "vitest";

import type { MapRoute } from "../../src/utils/map/deckRouteGeometry";
import { applyCorpusMapFrame } from "../../src/utils/corpus/monitorRoutes";

function route(id: string): MapRoute {
  return { id, label: id, data: {}, color: [0, 0, 0] };
}

describe("applyCorpusMapFrame", () => {
  const current = [route("log-day")];

  it("keeps the current route when playback day is null", () => {
    const applied = applyCorpusMapFrame(current, { routes: [route("corpus")], reason: null }, null);
    expect(applied.routes.map((item) => item.id)).toEqual(["log-day"]);
    expect(applied.reason).toBeNull();
    expect(applied.driven).toBe(false);
  });

  it("replaces the current route when the frame has geometry", () => {
    const applied = applyCorpusMapFrame(current, { routes: [route("corpus-day")], reason: null }, 3);
    expect(applied.routes.map((item) => item.id)).toEqual(["corpus-day"]);
    expect(applied.reason).toBeNull();
    expect(applied.driven).toBe(true);
  });

  it("keeps the current route and the reason when the frame has no geometry", () => {
    const applied = applyCorpusMapFrame(
      current,
      { routes: [], reason: "This partial run has no route geometry." },
      3
    );
    expect(applied.routes.map((item) => item.id)).toEqual(["log-day"]);
    expect(applied.reason).toBe("This partial run has no route geometry.");
    expect(applied.driven).toBe(false);
  });
});
