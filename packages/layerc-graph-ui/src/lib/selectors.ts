import { citationCounts } from "./graphTransform";
import type {
  EvidenceSource,
  ForceNode,
  GraphEdgeType,
  GraphFilterState,
  GraphLinkData,
  GraphMode,
  GraphNodeType,
  GraphTransformResult,
} from "./types";

/** Top-cited evidence ids by agent citations */
function topCitedEvidenceIds(
  result: GraphTransformResult,
  limit: number,
): Set<string> {
  const counts = citationCounts(result.agents);
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  return new Set(sorted.slice(0, limit).map(([id]) => id));
}

export interface FilteredGraph {
  nodes: ForceNode[];
  links: GraphLinkData[];
}

export function applyGraphFilters(
  result: GraphTransformResult,
  mode: GraphMode,
  filters: GraphFilterState,
): FilteredGraph {
  let { nodes, links } = result;

  const idSet = new Set(nodes.map((n) => n.id));

  /** Drop evidence not in top-N unless expandAllEvidence */
  if (!filters.expandAllEvidence) {
    const top = topCitedEvidenceIds(result, filters.topEvidenceLimit);
    const evKeep = new Set<string>();
    for (const n of nodes) {
      if (n.type !== "evidence" || !n.evidenceId) continue;
      if (top.has(n.evidenceId)) evKeep.add(n.id);
    }
    nodes = nodes.filter(
      (n) => n.type !== "evidence" || !n.evidenceId || evKeep.has(n.id),
    );
    const allowed = new Set(nodes.map((n) => n.id));
    links = links.filter(
      (l) =>
        allowed.has(l.source) && allowed.has(l.target),
    );
  }

  if (filters.onlyCitedEvidence) {
    const cited = new Set<string>();
    for (const [, ag] of Object.entries(result.agents)) {
      if (!ag?.citations) continue;
      for (const c of ag.citations) {
        cited.add(`ev:${c}`);
      }
    }
    nodes = nodes.filter(
      (n) => n.type !== "evidence" || cited.has(n.id),
    );
    const allowed = new Set(nodes.map((n) => n.id));
    links = links.filter(
      (l) => allowed.has(l.source) && allowed.has(l.target),
    );
  }

  if (!filters.evidenceSources.has("all")) {
    nodes = nodes.filter((n) => {
      if (n.type !== "evidence") return true;
      const src = n.source;
      if (!src) return false;
      return filters.evidenceSources.has(src as EvidenceSource);
    });
    const allowed = new Set(nodes.map((n) => n.id));
    links = links.filter(
      (l) => allowed.has(l.source) && allowed.has(l.target),
    );
  }

  if (filters.onlyActionable) {
    const keep = new Set<string>();
    for (const n of nodes) {
      if (n.type === "action") keep.add(n.id);
    }
    for (const l of links) {
      if (l.type === "targets") {
        keep.add(l.source);
        keep.add(l.target);
      }
      if (l.type === "recommends") {
        keep.add(l.target);
      }
    }
    const agentRa = "agent:response_advisor";
    if (idSet.has(agentRa)) keep.add(agentRa);
    const ep = result.episodeNodeId;
    keep.add(ep);

    nodes = nodes.filter((n) => keep.has(n.id));
    const allowed = new Set(nodes.map((n) => n.id));
    links = links.filter(
      (l) => allowed.has(l.source) && allowed.has(l.target),
    );
  }

  /** Node type filter */
  nodes = nodes.filter((n) => filters.nodeTypes.has(n.type));
  const allowedIds = new Set(nodes.map((n) => n.id));
  links = links.filter(
    (l) => allowedIds.has(l.source) && allowedIds.has(l.target),
  );

  /** Mode subgraph */
  links = filterLinksByMode(links, mode);
  const used = new Set<string>();
  for (const l of links) {
    used.add(l.source);
    used.add(l.target);
  }
  nodes = nodes.filter((n) => used.has(n.id));

  return { nodes, links };
}

function filterLinksByMode(
  links: GraphLinkData[],
  mode: GraphMode,
): GraphLinkData[] {
  if (mode === "case") return links;

  if (mode === "citation") {
    const allow: GraphEdgeType[] = ["cites"];
    return links.filter((l) => allow.includes(l.type));
  }

  if (mode === "decision") {
    const allow: GraphEdgeType[] = [
      "produced_by",
      "recommends",
      "targets",
      "guarded_by",
      "derived_from",
      "has_entity",
      "has_artifact",
    ];
    return links.filter((l) => allow.includes(l.type));
  }

  return links;
}

const DECISION_AGENT_IDS = new Set([
  "agent:triage",
  "agent:response_advisor",
]);

/** Decision view: triage + response advisor + actions + guardrails + targets (no hunt_planner). */
export function refineDecisionModeNodes(
  nodes: ForceNode[],
  links: GraphLinkData[],
  episodeNodeId: string,
): { nodes: ForceNode[]; links: GraphLinkData[] } {
  const wantedTypes = new Set<GraphNodeType>([
    "episode",
    "agent",
    "action",
    "guardrail",
    "entity",
    "artifact",
  ]);
  let fnodes = nodes.filter((n) => {
    if (!wantedTypes.has(n.type)) return false;
    if (n.type === "agent") {
      return DECISION_AGENT_IDS.has(n.id);
    }
    return true;
  });
  const allowed = new Set(fnodes.map((n) => n.id));
  const flinks = links.filter(
    (l) => allowed.has(l.source) && allowed.has(l.target),
  );
  const used = new Set<string>();
  for (const l of flinks) {
    used.add(l.source);
    used.add(l.target);
  }
  fnodes = fnodes.filter((n) => used.has(n.id) || n.id === episodeNodeId);
  return { nodes: fnodes, links: flinks };
}

export function applyGraphView(
  result: GraphTransformResult,
  mode: GraphMode,
  filters: GraphFilterState,
): FilteredGraph {
  let out = applyGraphFilters(result, mode, filters);
  if (mode === "decision") {
    out = refineDecisionModeNodes(
      out.nodes,
      out.links,
      result.episodeNodeId,
    );
  }
  return out;
}
