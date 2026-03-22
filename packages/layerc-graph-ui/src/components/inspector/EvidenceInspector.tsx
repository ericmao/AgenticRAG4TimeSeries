"use client";

import type { EvidenceItemJson } from "@/lib/types";
import { evidenceSourceLabel } from "@/lib/formatters";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/card";
import { ScrollArea } from "@/ui/scroll-area";

export function EvidenceInspector({
  item,
  citedByAgents,
}: {
  item: EvidenceItemJson;
  citedByAgents: string[];
}) {
  return (
    <Card className="border-0 shadow-none">
      <CardHeader className="pb-2">
        <CardTitle className="text-base leading-tight">{item.title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="bg-muted rounded px-1.5 py-0.5">
            {evidenceSourceLabel(item.source)}
          </span>
          <span className="bg-muted rounded px-1.5 py-0.5">{item.kind}</span>
          {item.score != null && (
            <span className="text-muted-foreground">
              score {(item.score * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <div>
          <span className="text-muted-foreground">evidence_id</span>
          <p className="font-mono text-xs">{item.evidence_id}</p>
        </div>
        <div>
          <span className="text-muted-foreground">Body</span>
          <ScrollArea className="max-h-40">
            <p className="text-foreground/90 text-xs leading-relaxed whitespace-pre-wrap">
              {item.body}
            </p>
          </ScrollArea>
        </div>
        <div>
          <span className="text-muted-foreground">Cited by agents</span>
          {citedByAgents.length === 0 ? (
            <p className="text-muted-foreground text-xs">—</p>
          ) : (
            <ul className="text-xs">
              {citedByAgents.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
