"use client";

import type { ForceNode } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/card";

export function EntityInspector({
  node,
}: {
  node: ForceNode;
}) {
  return (
    <Card className="border-0 shadow-none">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{node.label}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div>
          <span className="text-muted-foreground">Type</span>
          <p className="capitalize">{node.entityType ?? node.artifactType ?? "—"}</p>
        </div>
        <p className="text-muted-foreground text-xs">
          Related actions appear as edges from Action nodes (targets).
        </p>
      </CardContent>
    </Card>
  );
}
