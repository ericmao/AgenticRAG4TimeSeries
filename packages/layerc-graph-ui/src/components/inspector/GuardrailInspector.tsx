"use client";

import type { ForceNode } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/card";

export function GuardrailInspector({ node }: { node: ForceNode }) {
  return (
    <Card className="border-0 shadow-none">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{node.label}</CardTitle>
      </CardHeader>
      <CardContent className="text-sm">
        <p className="text-foreground/90 text-xs leading-relaxed">{node.summary}</p>
      </CardContent>
    </Card>
  );
}
