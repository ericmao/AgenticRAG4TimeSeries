"use client";

import type { LayerCCasePayload } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/card";
import { ScrollArea } from "@/ui/scroll-area";

export function EpisodeInspector({
  payload,
}: {
  payload: LayerCCasePayload;
}) {
  const ep = payload.episode;
  const t0 = ep?.t0_ms;
  const t1 = ep?.t1_ms;
  return (
    <Card className="border-0 shadow-none">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Episode</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div>
          <span className="text-muted-foreground">episode_id</span>
          <p className="font-mono text-xs">{payload.episode_id}</p>
        </div>
        {payload.run_id && (
          <div>
            <span className="text-muted-foreground">run_id</span>
            <p className="font-mono text-xs">{payload.run_id}</p>
          </div>
        )}
        {payload.status && (
          <div>
            <span className="text-muted-foreground">status</span>
            <p>{payload.status}</p>
          </div>
        )}
        {payload.alert_count != null && (
          <div>
            <span className="text-muted-foreground">alert_count</span>
            <p>{payload.alert_count}</p>
          </div>
        )}
        {t0 != null && t1 != null && (
          <div>
            <span className="text-muted-foreground">Time window</span>
            <p className="font-mono text-xs">
              {new Date(t0).toISOString()} — {new Date(t1).toISOString()}
            </p>
          </div>
        )}
        {ep?.entities && ep.entities.length > 0 && (
          <div>
            <span className="text-muted-foreground">Entities</span>
            <ScrollArea className="max-h-24">
              <ul className="list-inside list-disc text-xs">
                {ep.entities.map((e) => (
                  <li key={e}>{e}</li>
                ))}
              </ul>
            </ScrollArea>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
