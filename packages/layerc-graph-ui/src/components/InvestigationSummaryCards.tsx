"use client";

import type { GraphTransformResult, LayerCCasePayload } from "@/lib/types";
import { formatConfidence, triageLevelToBucket } from "@/lib/formatters";
import { Card, CardContent } from "@/ui/card";

function maxConfidence(agents: GraphTransformResult["agents"]): number {
  let m = 0;
  for (const ag of Object.values(agents)) {
    if (ag && ag.confidence > m) m = ag.confidence;
  }
  return m;
}

function actionCountFromGraph(graph: GraphTransformResult): number {
  const ra = graph.agents.response_advisor;
  const s = ra?.structured;
  if (s && typeof s === "object" && Array.isArray((s as { actions?: unknown }).actions)) {
    return (s as { actions: unknown[] }).actions.length;
  }
  return 0;
}

export function InvestigationSummaryCards({
  payload,
  graph,
}: {
  payload: LayerCCasePayload;
  graph: GraphTransformResult;
}) {
  const tri = graph.agents.triage;
  const tl =
    tri?.structured && typeof tri.structured === "object"
      ? String(
          (tri.structured as { triage_level?: string }).triage_level ?? "—",
        )
      : "—";
  const bucket = triageLevelToBucket(
    (tri?.structured as { triage_level?: string } | undefined)?.triage_level,
  );

  const evidenceCount = payload.evidence_json?.items.length ?? 0;
  const agentCount = Object.keys(graph.agents).length;
  const actions = actionCountFromGraph(graph);

  const cards = [
    { label: "Triage", value: tl, sub: bucket },
    { label: "Alerts", value: String(payload.alert_count ?? "—") },
    { label: "Evidence", value: String(evidenceCount) },
    { label: "Agents", value: String(agentCount) },
    { label: "Actions", value: String(actions) },
    {
      label: "Top confidence",
      value: formatConfidence(maxConfidence(graph.agents)),
    },
  ];

  return (
    <div className="flex min-w-0 flex-nowrap gap-2 overflow-x-auto pb-0.5 md:overflow-x-visible">
      {cards.map((c) => {
        const subDup =
          c.sub &&
          c.sub.toLowerCase() === String(c.value).trim().toLowerCase();
        return (
        <Card
          key={c.label}
          className="min-w-[6.75rem] shrink-0 py-3 shadow-sm md:min-w-0 md:flex-1 md:basis-0"
        >
          <CardContent className="px-3 py-0">
            <p className="text-muted-foreground text-[11px] uppercase tracking-wide whitespace-nowrap">
              {c.label}
            </p>
            <div className="flex flex-nowrap items-baseline gap-x-1.5">
              <p className="text-foreground text-lg font-semibold tabular-nums whitespace-nowrap">
                {c.value}
              </p>
              {c.sub && !subDup && (
                <p className="text-muted-foreground text-[10px] capitalize whitespace-nowrap">
                  {c.sub}
                </p>
              )}
            </div>
          </CardContent>
        </Card>
        );
      })}
    </div>
  );
}
