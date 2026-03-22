/**
 * Layer C investigation graph — aligned with AgenticRAG Episode / EvidenceSet / AgentOutput JSON.
 */

export type EvidenceSource = "opencti" | "kb" | "taxii" | "other";
export type EvidenceKind = "stix_object" | "snippet" | "relation";

export interface EvidenceItemJson {
  evidence_id: string;
  source: EvidenceSource;
  kind: EvidenceKind;
  title: string;
  body: string;
  score?: number;
  ts_ms?: number;
  provenance?: Record<string, unknown>;
}

export interface EvidenceSetJson {
  episode_id: string;
  run_id?: string;
  items: EvidenceItemJson[];
  stats?: Record<string, unknown>;
}

export type AgentId =
  | "triage"
  | "hunt_planner"
  | "response_advisor"
  | "entity_investigation"
  | "cti_correlation";

export interface AgentOutputJson {
  agent_id: AgentId;
  episode_id: string;
  run_id?: string;
  summary: string;
  confidence: number;
  citations: string[];
  assumptions?: string[];
  next_required_data?: string[];
  structured: Record<string, unknown>;
}

export interface AgentOutputsJson {
  by_rule?: Record<string, Partial<Record<AgentId, AgentOutputJson>>>;
  triage?: AgentOutputJson;
  hunt_planner?: AgentOutputJson;
  response_advisor?: AgentOutputJson;
}

export interface EpisodeJson {
  episode_id: string;
  run_id?: string;
  t0_ms?: number;
  t1_ms?: number;
  entities?: string[];
  artifacts?: Array<{ type?: string; value?: string }>;
  sequence_tags?: string[];
  events?: unknown[];
  risk_context?: Record<string, unknown>;
}

export interface LayerCCasePayload {
  episode_id: string;
  run_id?: string;
  status?: string;
  target_ip?: string;
  episode?: EpisodeJson;
  evidence_json?: EvidenceSetJson;
  agent_outputs_json?: AgentOutputsJson;
  writeback_json?: Record<string, unknown>;
  /** Convenience: alert count from run or episode stats */
  alert_count?: number;
}

export type GraphNodeType =
  | "episode"
  | "entity"
  | "artifact"
  | "agent"
  | "evidence"
  | "action"
  | "guardrail";

export type GraphEdgeType =
  | "has_entity"
  | "has_artifact"
  | "produced_by"
  | "cites"
  | "recommends"
  | "targets"
  | "guarded_by"
  | "derived_from";

export type GraphMode = "case" | "citation" | "decision";

export interface GraphNodeData {
  id: string;
  type: GraphNodeType;
  label: string;
  triageLevel?: string;
  alertCount?: number;
  status?: string;
  entityType?: string;
  artifactType?: string;
  agentId?: AgentId;
  confidence?: number;
  summary?: string;
  source?: string;
  evidenceKind?: EvidenceKind;
  score?: number;
  evidenceId?: string;
  actionType?: string;
  target?: string;
  expectedImpact?: string;
  severity?: string;
  body?: string;
  /** Full payload slice for inspector JSON toggle */
  raw?: unknown;
}

/** react-force-graph compatible node */
export type ForceNode = GraphNodeData & {
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
};

export interface GraphLinkData {
  source: string;
  target: string;
  type: GraphEdgeType;
  label?: string;
}

export interface CitationIndex {
  /** evidence_id -> agent ids that cite it */
  evidenceToAgents: Map<string, Set<string>>;
}

export interface GraphTransformResult {
  nodes: ForceNode[];
  links: GraphLinkData[];
  citationIndex: CitationIndex;
  /** agent_id -> AgentOutputJson */
  agents: Partial<Record<AgentId, AgentOutputJson>>;
  episodeNodeId: string;
}

export interface GraphFilterState {
  nodeTypes: Set<GraphNodeType>;
  evidenceSources: Set<EvidenceSource | "all">;
  onlyCitedEvidence: boolean;
  onlyActionable: boolean;
  /** false = top-N cited only */
  expandAllEvidence: boolean;
  topEvidenceLimit: number;
}

export const defaultGraphFilter = (): GraphFilterState => ({
  nodeTypes: new Set([
    "episode",
    "entity",
    "artifact",
    "agent",
    "evidence",
    "action",
    "guardrail",
  ]),
  evidenceSources: new Set(["all"]),
  onlyCitedEvidence: false,
  onlyActionable: false,
  expandAllEvidence: false,
  topEvidenceLimit: 8,
});
