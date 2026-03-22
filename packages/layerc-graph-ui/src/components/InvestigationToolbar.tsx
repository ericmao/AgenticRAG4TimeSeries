"use client";

import { GraphModeTabs } from "./graph/GraphModeTabs";
import type { GraphMode } from "@/lib/types";

export function InvestigationToolbar({
  title,
  mode,
  onModeChange,
}: {
  title: string;
  mode: GraphMode;
  onModeChange: (m: GraphMode) => void;
}) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <h1 className="text-foreground text-lg font-semibold tracking-tight">
        {title}
      </h1>
      <GraphModeTabs value={mode} onChange={onModeChange} />
    </div>
  );
}
