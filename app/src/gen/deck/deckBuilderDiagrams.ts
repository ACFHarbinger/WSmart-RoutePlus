/**
 * Native shape diagram renderers for PPTX deck generation (§H.6).
 */
import type PptxGenJS from "pptxgenjs";
import {
  SLIDE_W,
  ACCENT,
  DARK,
  MUTED,
  LIGHT_TXT,
  WHITE,
  GREEN,
  ORANGE,
  LIGHT_FILL,
  ARROW_ORANGE,
  PIPELINE_STAGES,
} from "./deckBuilderHelpers";

type Slide = ReturnType<PptxGenJS["addSlide"]>;

export function pipelineDiagram(slide: Slide): number {
  const n = PIPELINE_STAGES.length;
  const gap = 0.25;
  const left0 = 0.5;
  const totalW = SLIDE_W - 1.0;
  const stageW = (totalW - gap * (n - 1)) / n;
  const top = 1.9;
  const h = 1.5;
  const subH = 1.1;
  PIPELINE_STAGES.forEach(([name, caption], i) => {
    const left = left0 + i * (stageW + gap);
    const optional = name.includes("optional");
    slide.addShape("chevron", {
      x: left,
      y: top,
      w: stageW,
      h,
      fill: { color: optional ? "8A9BB0" : ACCENT },
    });
    slide.addText(name.replace(/\n/g, "\n"), {
      x: left,
      y: top,
      w: stageW,
      h,
      fontSize: 15,
      bold: true,
      color: WHITE,
      align: "center",
      valign: "middle",
    });
    slide.addText(caption, {
      x: left,
      y: top + h + 0.15,
      w: stageW,
      h: subH,
      fontSize: 14,
      color: MUTED,
      align: "center",
      valign: "top",
    });
  });
  return top + h + 0.15 + subH;
}

export function policyGridDiagram(slide: Slide): void {
  const columns: [string, (string | [string, string[]])[], "flat" | "grouped"][] = [
    [
      "Packing",
      ["Look-Ahead (LA)", "Last-Minute (LM, CF70)", "Last-Minute (LM, CF90)", "Service-Level (SL1)", "Service-Level (SL2)"],
      "flat",
    ],
    [
      "Routing Algorithm",
      [
        ["Exact Methods", ["BPC", "SWC-TCF"]],
        ["Meta-Heuristics", ["HGS", "ALNS", "SANS", "PG-CLNS", "PSOMA"]],
        ["Hyper-Heuristics", ["ACO-HH"]],
      ],
      "grouped",
    ],
    ["Route Improver", ["Fast-TSP", "Local Search (CLS)"], "flat"],
  ];
  const top = 1.15;
  const bottom = 6.5;
  const gap = 0.3;
  const left0 = 0.6;
  const colW = (SLIDE_W - 1.2 - gap * (columns.length - 1)) / columns.length;
  const headerH = 0.55;
  columns.forEach(([header, items, kind], ci) => {
    const left = left0 + ci * (colW + gap);
    slide.addShape("roundRect", { x: left, y: top, w: colW, h: headerH, fill: { color: DARK }, rectRadius: 0.06 });
    slide.addText(header, { x: left, y: top, w: colW, h: headerH, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle" });
    const bodyTop = top + headerH + 0.15;
    const bodyH = bottom - bodyTop;
    if (kind === "flat") {
      const flat = items as string[];
      const itemGap = 0.12;
      const itemH = (bodyH - itemGap * (flat.length - 1)) / flat.length;
      flat.forEach((label, ii) => {
        const y = bodyTop + ii * (itemH + itemGap);
        slide.addShape("roundRect", { x: left, y, w: colW, h: itemH, fill: { color: ACCENT }, rectRadius: 0.06 });
        slide.addText(label, { x: left, y, w: colW, h: itemH, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle" });
      });
    } else {
      const grouped = items as [string, string[]][];
      const grpGap = 0.15;
      const grpH = (bodyH - grpGap * (grouped.length - 1)) / grouped.length;
      grouped.forEach(([grpLabel, subItems], gi) => {
        const y = bodyTop + gi * (grpH + grpGap);
        slide.addShape("rect", {
          x: left,
          y,
          w: colW,
          h: grpH,
          fill: { color: WHITE, transparency: 100 },
          line: { color: MUTED, width: 1, dashType: "dash" },
        });
        slide.addText(
          [
            { text: grpLabel, options: { fontSize: 12, bold: true, color: ACCENT, align: "center", breakLine: true } },
            { text: subItems.join("  ·  "), options: { fontSize: 11, color: DARK, align: "center" } },
          ],
          { x: left, y, w: colW, h: grpH, valign: "top" }
        );
      });
    }
  });
}

export function doeTreeDiagram(slide: Slide): number {
  const horizons: [string, [string, string[]][]][] = [
    ["30 Days", [["RM-100", ["Empirical", "Gamma-3"]], ["RM-170", ["Empirical", "Gamma-3"]], ["FFZ-350", ["Empirical", "Gamma-3"]]]],
    ["90 Days\n(Pareto-front policies only)", [["RM-100", ["Empirical", "Gamma-3"]], ["RM-170", ["Empirical", "Gamma-3"]], ["FFZ-350", ["Empirical", "Gamma-3"]]]],
  ];
  const top = 1.25;
  const rootW = 2.2;
  const rootH = 0.5;
  slide.addShape("roundRect", { x: (SLIDE_W - rootW) / 2, y: top, w: rootW, h: rootH, fill: { color: DARK }, rectRadius: 0.08 });
  slide.addText("Simulation Runs", { x: (SLIDE_W - rootW) / 2, y: top, w: rootW, h: rootH, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle" });
  const nH = horizons.length;
  const gapH = 0.4;
  const hzTop = top + rootH + 0.35;
  const hzW = (SLIDE_W - 1.2 - gapH * (nH - 1)) / nH;
  const hzH = 0.6;
  const left0 = 0.6;
  let distBottom = hzTop;
  horizons.forEach(([hzLabel, scenarios], hi) => {
    const hzLeft = left0 + hi * (hzW + gapH);
    slide.addShape("roundRect", { x: hzLeft, y: hzTop, w: hzW, h: hzH, fill: { color: ACCENT }, rectRadius: 0.06 });
    slide.addText(hzLabel, { x: hzLeft, y: hzTop, w: hzW, h: hzH, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle" });
    const nS = scenarios.length;
    const gapS = 0.15;
    const scTop = hzTop + hzH + 0.3;
    const scW = (hzW - gapS * (nS - 1)) / nS;
    const scH = 0.5;
    scenarios.forEach(([scLabel, dists], si) => {
      const scLeft = hzLeft + si * (scW + gapS);
      slide.addShape("rect", { x: scLeft, y: scTop, w: scW, h: scH, fill: { color: "8A9BB0" } });
      slide.addText(scLabel, { x: scLeft, y: scTop, w: scW, h: scH, fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle" });
      const distTop = scTop + scH + 0.1;
      slide.addText(dists.join(" / "), { x: scLeft, y: distTop, w: scW, h: 0.4, fontSize: 10, color: DARK, align: "center", valign: "top" });
      distBottom = distTop + 0.4;
    });
  });
  return distBottom;
}

export function algoTaxonomyDiagram(slide: Slide): void {
  const groups: [string, string, string[]][] = [
    ["Exact Methods", ACCENT, ["Branch-Price-and-Cut\n(BPC)", "Smart Waste Collection\nTwo-Commodity Flow (SWC-TCF)"]],
    ["Meta-Heuristics", GREEN, [
      "Hybrid Genetic Search (HGS)",
      "Adaptive Large Neighborhood Search (ALNS)",
      "Simulated Annealing Neighborhood Search (SANS)",
      "Policy-Gradient Cooperative LNS (PG-CLNS)",
      "Particle Swarm Optimisation Memetic Algorithm (PSOMA)",
    ]],
    ["Hyper-Heuristics", ORANGE, ["Ant Colony Optimisation Hyper-Heuristic (ACO-HH)"]],
  ];
  const top = 1.25;
  const bottom = 4.15;
  const gap = 0.35;
  const left0 = 0.6;
  const colW = (SLIDE_W - 1.2 - gap * (groups.length - 1)) / groups.length;
  const headerH = 0.6;
  groups.forEach(([header, color, items], ci) => {
    const left = left0 + ci * (colW + gap);
    slide.addShape("roundRect", { x: left, y: top, w: colW, h: headerH, fill: { color }, rectRadius: 0.06 });
    slide.addText(header, { x: left, y: top, w: colW, h: headerH, fontSize: 17, bold: true, color: WHITE, align: "center", valign: "middle" });
    const bodyTop = top + headerH + 0.15;
    const bodyH = bottom - bodyTop;
    const itemGap = 0.12;
    const itemH = (bodyH - itemGap * (items.length - 1)) / items.length;
    items.forEach((label, ii) => {
      const y = bodyTop + ii * (itemH + itemGap);
      slide.addShape("roundRect", { x: left, y, w: colW, h: itemH, fill: { color: LIGHT_FILL }, line: { color, width: 1.25 }, rectRadius: 0.06 });
      slide.addText(label, { x: left, y, w: colW, h: itemH, fontSize: 11, bold: true, color: DARK, align: "center", valign: "middle" });
    });
  });
}

export function renderObjectiveShapes(slide: Slide): void {
  slide.addShape("rect", { x: 4.126, y: 1.242, w: 5.634, h: 0.438, fill: { color: DARK } });
  slide.addText("One framework to compare them all", { x: 4.126, y: 1.242, w: 5.634, h: 0.438, fontSize: 15, italic: true, color: WHITE, align: "center", valign: "middle" });

  const rbox = (x: number, y: number, w: number, h: number, color: string, text: string, size = 14) => {
    slide.addShape("roundRect", { x, y, w, h, fill: { color }, rectRadius: 0.06 });
    slide.addText(text, { x, y, w, h, fontSize: size, bold: true, color: WHITE, align: "center", valign: "middle" });
  };
  rbox(2.059, 1.791, 1.426, 0.457, ACCENT, "Exact Methods", 13);
  rbox(2.059, 2.414, 1.426, 0.6, GREEN, "Meta-Heuristics", 13);
  rbox(2.059, 3.159, 1.426, 0.6, ORANGE, "Hyper-Heuristics", 13);
  slide.addShape("roundRect", { x: 5.049, y: 1.704, w: 2.426, h: 2.046, fill: { color: DARK }, rectRadius: 0.08 });
  slide.addText(
    [
      { text: "One Shared Simulator", options: { fontSize: 15, bold: true, color: WHITE, align: "center", breakLine: true } },
      { text: "Multi-day simulator:\nregion × graph size × demand", options: { fontSize: 11, color: LIGHT_TXT, align: "center", paraSpaceBefore: 6 } },
    ],
    { x: 5.049, y: 1.704, w: 2.426, h: 2.046, valign: "middle" }
  );
  rbox(8.612, 2.158, 1.564, 1.139, ACCENT, "Fair Benchmarks\n(KPIs)", 13);
  const connector = (x: number, y: number, w: number, h: number, color: string, flipV = false) =>
    slide.addShape("line", { x, y, w, h, flipV, line: { color, width: 2, endArrowType: "triangle" } });
  connector(3.485, 2.02, 1.564, 0.707, ACCENT);
  connector(3.485, 2.714, 1.564, 0.013, GREEN);
  connector(3.485, 2.727, 1.564, 0.732, ORANGE, true);
  connector(7.475, 2.727, 1.136, 0.001, ACCENT);

  const taxHeader = (x: number, y: number, color: string, text: string) => {
    slide.addShape("roundRect", { x, y, w: 3.811, h: 0.6, fill: { color }, rectRadius: 0.06 });
    slide.addText(text, { x, y, w: 3.811, h: 0.6, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle" });
  };
  const taxItem = (x: number, y: number, h: number, text: string, lineColor: string, size = 13) => {
    slide.addShape("roundRect", { x, y, w: 3.811, h, fill: { color: LIGHT_FILL }, line: { color: lineColor, width: 1.25 }, rectRadius: 0.06 });
    slide.addText(text, { x, y, w: 3.811, h, fontSize: size, bold: true, color: DARK, align: "center", valign: "middle" });
  };
  taxHeader(0.64, 4.258, ACCENT, "Exact Methods");
  taxItem(0.64, 5.008, 1.015, "Branch-and-Price-and-Cut (BPC)", ACCENT);
  taxItem(0.64, 6.143, 1.015, "Smart Waste Collection —\nTwo-Commodity Flow (SWC-TCF)", ACCENT);
  taxHeader(4.801, 4.258, GREEN, "Meta-Heuristics");
  (
    [
      ["Hybrid Genetic Search (HGS)", 5.008],
      ["Adaptive Large Neighborhood Search (ALNS)", 5.462],
      ["Simulated Annealing Neighborhood Search (SANS)", 5.916],
      ["Pheromone-Guided Cooperative Large Neighborhood Search (PG-CLNS)", 6.37],
      ["Particle Swarm Optimization Memetic Algorithm (PSOMA)", 6.824],
    ] as [string, number][]
  ).forEach(([algo], i) => taxItem(4.801, 5.008 + i * 0.49, 0.45, algo, GREEN, 10));
  taxHeader(8.962, 4.258, ORANGE, "Hyper-Heuristics");
  taxItem(8.962, 5.008, 2.15, "Ant Colony Optimization\nHyper-Heuristic (ACO-HH)", ORANGE);
}

export function renderSimulatorShapes(slide: Slide, binImg: string, truckImg: string): void {
  const binAndLabel = (x: number, y: number, pct: string) => {
    slide.addImage({ data: binImg, x, y, w: 0.495, h: 0.703 });
    slide.addText(pct, { x: x + 0.023, y: y + 0.36, w: 0.449, h: 0.286, fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle" });
  };
  const leftBins: [number, number, string][] = [
    [1.063, 4.912, "87%"], [0.431, 5.399, "60%"], [0.708, 6.182, "42%"], [1.405, 5.6, "50%"],
    [1.538, 4.57, "30%"], [2.063, 5.113, "80%"], [2.412, 6.26, "20%"], [2.851, 5.335, "55%"],
  ];
  for (const [x, y, pct] of leftBins) binAndLabel(x, y, pct);
  slide.addShape("rect", { x: 0.99, y: 4.902, w: 0.599, h: 0.759, fill: { color: WHITE, transparency: 100 }, line: { color: "FF0000", width: 2.5 } });
  const rightBins: [number, number, string][] = [
    [7.424, 4.698, "87%"], [6.791, 5.185, "60%"], [7.068, 5.968, "42%"], [7.766, 5.386, "50%"],
    [7.899, 4.356, "30%"], [8.424, 4.899, "80%"], [8.773, 6.046, "20%"], [9.212, 5.121, "55%"],
  ];
  for (const [x, y, pct] of rightBins) binAndLabel(x, y, pct);
  slide.addImage({ data: truckImg, x: 7.7, y: 6.673, w: 1.231, h: 0.943 });

  const tourArrows: [number, number, number, number, boolean, boolean][] = [
    [8.005, 6.032, 0.217, 0.756, true, true],
    [7.245, 5.688, 0.528, 0.158, true, true],
    [7.039, 5.049, 0.385, 0.136, false, true],
    [7.706, 4.567, 0.226, 0.146, false, true],
    [8.299, 4.661, 0.373, 0.238, false, false],
    [8.462, 5.575, 0.14, 1.213, true, false],
  ];
  for (const [x, y, w, h, flipH, flipV] of tourArrows) {
    slide.addShape("line", {
      x,
      y,
      w,
      h,
      flipH,
      flipV,
      line: { color: ARROW_ORANGE, width: 1.5, endArrowType: "triangle" },
    });
  }

  slide.addShape("rightBrace", {
    x: 7.977,
    y: -0.194,
    w: 0.323,
    h: 8.752,
    rotate: 90,
    fill: { color: WHITE, transparency: 100 },
    line: { color: ACCENT, width: 1.5 },
  });
}
