"use client";

import type { AgentOutputJson } from "@/lib/types";
import { formatConfidence } from "@/lib/formatters";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/card";
import { ScrollArea } from "@/ui/scroll-area";
import { useState } from "react";

export function AgentInspector({ agent }: { agent: AgentOutputJson }) {
  const [showJson, setShowJson] = useState(false);
  return (
    <Card className="border-0 shadow-none">
      <CardHeader className="pb-2">
        <CardTitle className="text-base capitalize">
          {agent.agent_id.replace(/_/g, " ")}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div>
          <span className="text-muted-foreground">Summary</span>
          <p className="text-foreground/90 leading-snug">{agent.summary}</p>
        </div>
        <div>
          <span className="text-muted-foreground">Confidence</span>
          <p>{formatConfidence(agent.confidence)}</p>
        </div>
        <div>
          <span className="text-muted-foreground">Citations</span>
          <p>{agent.citations?.length ?? 0} evidence id(s)</p>
          <ScrollArea className="max-h-20">
            <ul className="font-mono text-[11px]">
              {(agent.citations ?? []).map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          </ScrollArea>
        </div>
        {(agent.assumptions?.length ?? 0) > 0 && (
          <div>
            <span className="text-muted-foreground">Assumptions</span>
            <ul className="list-inside list-disc text-xs">
              {agent.assumptions!.map((a, i) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
          </div>
        )}
        {(agent.next_required_data?.length ?? 0) > 0 && (
          <div>
            <span className="text-muted-foreground">Next required data</span>
            <ul className="list-inside list-disc text-xs">
              {agent.next_required_data!.map((a, i) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
          </div>
        )}
        <button
          type="button"
          className="text-primary text-xs underline-offset-4 hover:underline"
          onClick={() => setShowJson((v) => !v)}
        >
          {showJson ? "Hide" : "Show"} structured JSON
        </button>
        {showJson && (
          <pre className="bg-muted max-h-48 overflow-auto rounded p-2 text-[10px] leading-tight">
            {JSON.stringify(agent.structured, null, 2)}
          </pre>
        )}
      </CardContent>
    </Card>
  );
}
