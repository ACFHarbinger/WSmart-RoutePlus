/** Families and ports the builder will connect. */

export interface LinkEnd {
  entryId: string;
  family: string;
}

export function normFamily(family: string): string {
  const key = family.toLowerCase().replace(/[\s-]+/g, "_");
  if (key === "operator" || key === "operators") return "operator";
  if (key === "cut" || key === "cuts" || key === "separation") return "separation";
  if (key === "improvement" || key === "route_improvement") return "route_improvement";
  if (key === "acceptance_criteria") return "acceptance";
  return key;
}

/** Branch-and-cut / branch-and-price routing ids that can host a cut. */
export function routingAcceptsCuts(entryId: string): boolean {
  const id = entryId.toLowerCase();
  return id === "bpc" || id === "swc_tcf" || id.includes("branch") || id.includes("bpc");
}

/**
 * Return a refusal reason, or null when the two modules may be linked.
 * Direction does not matter: packing→routing and routing→packing are the same pair.
 */
export function linkRefusal(source: LinkEnd, target: LinkEnd): string | null {
  const left = normFamily(source.family);
  const right = normFamily(target.family);

  if (left === "acceptance" || right === "acceptance") {
    const other = left === "acceptance" ? right : left;
    const otherId = left === "acceptance" ? target.entryId : source.entryId;
    if (other !== "routing") {
      const where = other === "packing" ? "packing" : other;
      return `Refused: acceptance connects only to a routing node, not to ${where} (${otherId}).`;
    }
    return null;
  }

  if (left === "separation" || right === "separation") {
    const other = left === "separation" ? target : source;
    const otherFamily = left === "separation" ? right : left;
    if (otherFamily === "routing" && routingAcceptsCuts(other.entryId)) return null;
    const packingNote =
      otherFamily === "packing"
        ? ` It must not connect to packing (${other.entryId}).`
        : ` ${other.entryId} is not a branch-and-cut routing module.`;
    return (
      `Refused: a separation or cut module connects only to a routing id that is bpc, swc_tcf, ` +
      `or whose id contains "branch" or "bpc".${packingNote}`
    );
  }

  if (left === "packing" || right === "packing") {
    const other = left === "packing" ? right : left;
    if (other !== "routing") {
      return `Refused: packing connects to routing, not to ${other}.`;
    }
    return null;
  }

  if (left === "route_improvement" || right === "route_improvement") {
    const other = left === "route_improvement" ? right : left;
    if (other !== "routing") {
      return `Refused: route improvement connects to routing, not to ${other}.`;
    }
    return null;
  }

  if (left === "operator" || right === "operator") {
    const other = left === "operator" ? right : left;
    if (other !== "routing") {
      return `Refused: operators connect to a routing node, not to ${other}.`;
    }
    return null;
  }

  return `Refused: ${left} cannot connect to ${right}.`;
}
