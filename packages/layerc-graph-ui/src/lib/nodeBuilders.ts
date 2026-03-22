import type {
  AgentId,
  AgentOutputJson,
  ForceNode,
  GraphNodeData,
} from "./types";

export function episodeNodeId(episodeId: string): string {
  return `ep:${episodeId}`;
}

export function agentNodeId(agentId: AgentId): string {
  return `agent:${agentId}`;
}

export function evidenceNodeId(evidenceId: string): string {
  return `ev:${evidenceId}`;
}

export function entityNodeId(idx: number): string {
  return `ent:${idx}`;
}

export function artifactNodeId(idx: number): string {
  return `art:${idx}`;
}

export function actionNodeId(idx: number): string {
  return `act:${idx}`;
}

export const GUARD_NODE_ID = "guard:policy";

export function buildEpisodeNode(
  episodeId: string,
  opts: {
    triageLevel?: string;
    alertCount?: number;
    status?: string;
    label?: string;
  },
): ForceNode {
  const data: GraphNodeData = {
    id: episodeNodeId(episodeId),
    type: "episode",
    label: opts.label ?? episodeId,
    triageLevel: opts.triageLevel,
    alertCount: opts.alertCount,
    status: opts.status,
  };
  return { ...data, fx: 0, fy: 0 };
}

export function buildAgentNode(agent: AgentOutputJson): ForceNode {
  return {
    id: agentNodeId(agent.agent_id),
    type: "agent",
    label: agent.agent_id.replace(/_/g, " "),
    agentId: agent.agent_id,
    confidence: agent.confidence,
    summary: agent.summary,
    raw: agent,
  };
}

export function buildEvidenceNode(
  evidenceId: string,
  partial: {
    label: string;
    source?: string;
    evidenceKind?: GraphNodeData["evidenceKind"];
    score?: number;
    body?: string;
    raw?: unknown;
  },
): ForceNode {
  return {
    id: evidenceNodeId(evidenceId),
    type: "evidence",
    evidenceId,
    label: partial.label,
    source: partial.source,
    evidenceKind: partial.evidenceKind,
    score: partial.score,
    body: partial.body,
    raw: partial.raw,
  };
}

export function buildEntityNode(
  idx: number,
  name: string,
  entityType = "host",
): ForceNode {
  return {
    id: entityNodeId(idx),
    type: "entity",
    label: name,
    entityType,
  };
}

export function buildArtifactNode(
  idx: number,
  value: string,
  artifactType?: string,
): ForceNode {
  return {
    id: artifactNodeId(idx),
    type: "artifact",
    label: value,
    artifactType: artifactType ?? "artifact",
  };
}

export function buildActionNode(
  idx: number,
  partial: {
    actionType: string;
    target?: string;
    expectedImpact?: string;
    raw?: unknown;
  },
): ForceNode {
  return {
    id: actionNodeId(idx),
    type: "action",
    label: partial.actionType.replace(/_/g, " "),
    actionType: partial.actionType,
    target: partial.target,
    expectedImpact: partial.expectedImpact,
    raw: partial.raw,
  };
}

export function buildGuardrailNode(summary: string): ForceNode {
  return {
    id: GUARD_NODE_ID,
    type: "guardrail",
    label: "Policy guardrails",
    summary: summary.slice(0, 200),
    raw: { summary },
  };
}

export function inferEntityType(name: string): string {
  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(name) || name.includes(":"))
    return "ip";
  if (name.includes("\\") || name.includes("/")) return "file";
  if (name.includes(".")) return "host";
  return "entity";
}
