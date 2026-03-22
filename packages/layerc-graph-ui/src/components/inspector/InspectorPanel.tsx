"use client";

import type { ReactNode } from "react";
import type {
  ForceNode,
  GraphTransformResult,
  LayerCCasePayload,
} from "@/lib/types";
import { ActionInspector } from "./ActionInspector";
import { AgentInspector } from "./AgentInspector";
import { EpisodeInspector } from "./EpisodeInspector";
import { EntityInspector } from "./EntityInspector";
import { EvidenceInspector } from "./EvidenceInspector";
import { GuardrailInspector } from "./GuardrailInspector";
import { ScrollArea } from "@/ui/scroll-area";

function evidenceItemById(
  payload: LayerCCasePayload,
  id: string,
) {
  return payload.evidence_json?.items.find((i) => i.evidence_id === id);
}

export function InspectorPanel({
  selectedId,
  payload,
  graph,
  nodeMap,
}: {
  selectedId: string | null;
  payload: LayerCCasePayload;
  graph: GraphTransformResult;
  nodeMap: Map<string, ForceNode>;
}) {
  if (!selectedId) {
    return (
      <div className="text-muted-foreground flex min-h-[120px] flex-1 items-start rounded-md border border-dashed p-4 text-sm">
        Select a node on the graph to inspect citations, agents, and actions.
      </div>
    );
  }

  const node = nodeMap.get(selectedId);
  if (!node) {
    return (
      <div className="text-muted-foreground flex min-h-[120px] flex-1 items-start rounded-md border border-dashed p-4 text-sm">
        Node not found.
      </div>
    );
  }

  const citeIdx = graph.citationIndex.evidenceToAgents;

  let body: ReactNode;
  switch (node.type) {
    case "episode":
      body = <EpisodeInspector payload={payload} />;
      break;
    case "agent": {
      const ag = node.agentId ? graph.agents[node.agentId] : undefined;
      body = ag ? (
        <AgentInspector agent={ag} />
      ) : (
        <p className="text-muted-foreground text-sm">Missing agent output.</p>
      );
      break;
    }
    case "evidence": {
      const eid = node.evidenceId ?? "";
      const item = evidenceItemById(payload, eid);
      const agents = [...(citeIdx.get(eid) ?? [])];
      body = item ? (
        <EvidenceInspector item={item} citedByAgents={agents} />
      ) : (
        <p className="text-muted-foreground text-sm">Evidence not in payload.</p>
      );
      break;
    }
    case "action":
      body = (
        <ActionInspector
          action={
            (node.raw as Record<string, unknown>) ?? {
              action: node.actionType,
              target: node.target,
            }
          }
        />
      );
      break;
    case "entity":
    case "artifact":
      body = <EntityInspector node={node} />;
      break;
    case "guardrail":
      body = <GuardrailInspector node={node} />;
      break;
    default:
      body = <pre className="text-xs">{JSON.stringify(node, null, 2)}</pre>;
  }

  return (
    <ScrollArea className="min-h-0 flex-1 pr-2">
      {body}
    </ScrollArea>
  );
}
