"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/ui/card";

function unknownToDisplayString(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") {
    return String(v);
  }
  return JSON.stringify(v);
}

export function ActionInspector({
  action,
}: {
  action: Record<string, unknown>;
}) {
  const a = action as {
    target?: unknown;
    duration_minutes?: unknown;
    guardrails?: unknown;
    rollback_conditions?: unknown;
    action?: unknown;
  };
  const target = a.target != null ? unknownToDisplayString(a.target) : "—";
  const duration = a.duration_minutes;
  const guard = a.guardrails;
  const rollback = a.rollback_conditions;
  const titleRaw = typeof a.action === "string" ? a.action : "action";

  return (
    <Card className="border-0 shadow-none">
      <CardHeader className="pb-2">
        <CardTitle className="text-base capitalize">
          {titleRaw.replace(/_/g, " ")}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div>
          <span className="text-muted-foreground">Target</span>
          <p className="font-mono text-xs break-all">{target}</p>
        </div>
        {duration != null && (
          <div>
            <span className="text-muted-foreground">Duration (min)</span>
            <p>{unknownToDisplayString(duration)}</p>
          </div>
        )}
        {typeof guard === "string" && (
          <div>
            <span className="text-muted-foreground">Guardrails</span>
            <p className="text-xs leading-snug">{guard}</p>
          </div>
        )}
        {Array.isArray(rollback) && (
          <div>
            <span className="text-muted-foreground">Rollback conditions</span>
            <ul className="list-inside list-disc text-xs">
              {rollback.map((r, i) => (
                <li key={i}>{String(r)}</li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
