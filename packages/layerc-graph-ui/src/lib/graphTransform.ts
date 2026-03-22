import type {
  AgentId,
  AgentOutputJson,
  AgentOutputsJson,
  EvidenceItemJson,
  GraphLinkData,
  GraphTransformResult,
  LayerCCasePayload,
} from "./types";
import { link } from "./edgeBuilders";
import {
  actionNodeId,
  agentNodeId,
  artifactNodeId,
  buildActionNode,
  buildAgentNode,
  buildArtifactNode,
  buildEpisodeNode,
  buildEntityNode,
  buildEvidenceNode,
  buildGuardrailNode,
  entityNodeId,
  episodeNodeId,
  evidenceNodeId,
  GUARD_NODE_ID,
  inferEntityType,
} from "./nodeBuilders";

export function pickPrimaryRuleBundle(
  ao: AgentOutputsJson | undefined,
): Partial<Record<AgentId, AgentOutputJson>> | null {
  if (!ao || typeof ao !== "object") return null;
  const br = ao.by_rule;
  if (br && typeof br === "object") {
    const keys = Object.keys(br);
    if (keys.length > 0) {
      const first = br[keys[0]];
      if (first && typeof first === "object") {
        const out: Partial<Record<AgentId, AgentOutputJson>> = {};
        for (const k of [
          "triage",
          "hunt_planner",
          "response_advisor",
          "entity_investigation",
          "cti_correlation",
        ] as const) {
          const v = first[k];
          if (v && typeof v === "object" && "agent_id" in v) {
            out[k] = v;
          }
        }
        if (Object.keys(out).length > 0) return out;
      }
    }
  }
  const fallback: Partial<Record<AgentId, AgentOutputJson>> = {};
  if (ao.triage) fallback.triage = ao.triage;
  if (ao.hunt_planner) fallback.hunt_planner = ao.hunt_planner;
  if (ao.response_advisor) fallback.response_advisor = ao.response_advisor;
  return Object.keys(fallback).length > 0 ? fallback : null;
}

function triageLevelFromAgents(
  agents: Partial<Record<AgentId, AgentOutputJson>>,
): string | undefined {
  const t = agents.triage;
  const s = t?.structured;
  if (s && typeof s === "object" && "triage_level" in s) {
    return String((s as { triage_level?: string }).triage_level);
  }
  return undefined;
}

function collectCitationIndex(
  agents: Partial<Record<AgentId, AgentOutputJson>>,
): Map<string, Set<string>> {
  const m = new Map<string, Set<string>>();
  for (const ag of Object.values(agents)) {
    if (!ag?.citations?.length) continue;
    const aid = ag.agent_id;
    for (const eid of ag.citations) {
      if (!m.has(eid)) m.set(eid, new Set());
      m.get(eid)!.add(aid);
    }
  }
  return m;
}

function citationCounts(
  agents: Partial<Record<AgentId, AgentOutputJson>>,
): Map<string, number> {
  const counts = new Map<string, number>();
  for (const ag of Object.values(agents)) {
    if (!ag?.citations?.length) continue;
    for (const eid of ag.citations) {
      counts.set(eid, (counts.get(eid) ?? 0) + 1);
    }
  }
  return counts;
}

function matchTargetsToGraphIds(
  target: string | undefined,
  entityNodes: { id: string; label: string }[],
  artifactNodes: { id: string; label: string }[],
): string[] {
  if (!target?.trim()) return [];
  const tokens = target.split(/[,;]/).map((s) => s.trim()).filter(Boolean);
  const hits = new Set<string>();
  for (const tok of tokens) {
    for (const n of entityNodes) {
      if (n.label.includes(tok) || tok.includes(n.label)) hits.add(n.id);
    }
    for (const n of artifactNodes) {
      if (n.label.includes(tok) || tok.includes(n.label)) hits.add(n.id);
    }
  }
  const firstEnt = entityNodes[0];
  if (hits.size === 0 && firstEnt) {
    hits.add(firstEnt.id);
  }
  return [...hits];
}

export function buildGraphFromCase(payload: LayerCCasePayload): GraphTransformResult {
  const episodeId = payload.episode_id;
  const ep = payload.episode;
  const evidenceItems: EvidenceItemJson[] = payload.evidence_json?.items ?? [];

  const agents = pickPrimaryRuleBundle(payload.agent_outputs_json) ?? {};

  const triageLevel = triageLevelFromAgents(agents);
  const alertCount =
    payload.alert_count ??
    (ep?.events && Array.isArray(ep.events) ? ep.events.length : undefined);

  const nodes: GraphTransformResult["nodes"] = [];
  const links: GraphLinkData[] = [];

  const epNode = buildEpisodeNode(episodeId, {
    triageLevel,
    alertCount,
    status: payload.status,
    label: episodeId,
  });
  nodes.push(epNode);

  const entityLabels: { id: string; label: string }[] = [];
  const entities = ep?.entities ?? [];
  entities.forEach((name, idx) => {
    const id = entityNodeId(idx);
    const n = buildEntityNode(idx, name, inferEntityType(name));
    nodes.push(n);
    entityLabels.push({ id, label: name });
    links.push(link(episodeNodeId(episodeId), id, "has_entity"));
  });

  const artifacts = ep?.artifacts ?? [];
  const artifactLabels: { id: string; label: string }[] = [];
  artifacts.forEach((a, idx) => {
    const val = String(a.value ?? "");
    const id = artifactNodeId(idx);
    const n = buildArtifactNode(idx, val, a.type);
    nodes.push(n);
    artifactLabels.push({ id, label: val });
    links.push(link(episodeNodeId(episodeId), id, "has_artifact"));
  });

  const agentOrder: AgentId[] = [
    "triage",
    "hunt_planner",
    "response_advisor",
    "entity_investigation",
    "cti_correlation",
  ];
  for (const aid of agentOrder) {
    const ag = agents[aid];
    if (!ag) continue;
    const an = buildAgentNode(ag);
    nodes.push(an);
    links.push(link(episodeNodeId(episodeId), an.id, "produced_by"));
    for (const cid of ag.citations ?? []) {
      links.push(link(an.id, evidenceNodeId(cid), "cites"));
    }
  }

  for (const e of evidenceItems) {
    nodes.push(
      buildEvidenceNode(e.evidence_id, {
        label: e.title,
        source: e.source,
        evidenceKind: e.kind,
        score: e.score,
        body: e.body,
        raw: e,
      }),
    );
  }

  const ra = agents.response_advisor;
  const structured = ra?.structured;
  const actionsRaw =
    structured &&
    typeof structured === "object" &&
    Array.isArray((structured as { actions?: unknown }).actions)
      ? (structured as { actions: Record<string, unknown>[] }).actions
      : [];

  let actionIdx = 0;
  for (const raw of actionsRaw) {
    const act = raw as { action?: unknown; target?: unknown; guardrails?: unknown };
    const actionType =
      act.action == null
        ? "unknown"
        : typeof act.action === "string" || typeof act.action === "number"
          ? String(act.action)
          : "unknown";
    const target =
      act.target == null
        ? undefined
        : typeof act.target === "string" || typeof act.target === "number"
          ? String(act.target)
          : undefined;
    const an = buildActionNode(actionIdx, {
      actionType,
      target,
      expectedImpact: ra?.structured
        ? String((ra.structured as { expected_impact?: string }).expected_impact ?? "")
        : undefined,
      raw: raw,
    });
    nodes.push(an);
    if (ra) {
      links.push(link(agentNodeId("response_advisor"), an.id, "recommends"));
    }
    const tgtIds = matchTargetsToGraphIds(target, entityLabels, artifactLabels);
    for (const tid of tgtIds) {
      links.push(link(an.id, tid, "targets"));
    }
    actionIdx += 1;
  }

  const guardTexts: string[] = [];
  for (const raw of actionsRaw) {
    const act = raw as { guardrails?: unknown };
    const g = act.guardrails;
    if (typeof g === "string" && g.trim()) guardTexts.push(g.trim());
  }
  if (guardTexts.length > 0 && actionsRaw.length > 0) {
    const gnode = buildGuardrailNode(guardTexts.join(" | "));
    nodes.push(gnode);
    for (let i = 0; i < actionIdx; i++) {
      links.push(link(actionNodeId(i), GUARD_NODE_ID, "guarded_by"));
    }
  }

  const citationIndex = {
    evidenceToAgents: collectCitationIndex(agents),
  };

  return {
    nodes,
    links,
    citationIndex,
    agents,
    episodeNodeId: episodeNodeId(episodeId),
  };
}

export { citationCounts };
