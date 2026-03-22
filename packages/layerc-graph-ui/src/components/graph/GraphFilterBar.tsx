"use client";

import type { EvidenceSource, GraphFilterState, GraphNodeType } from "@/lib/types";
import { defaultGraphFilter } from "@/lib/types";
import { Label } from "@/ui/label";
import { Switch } from "@/ui/switch";
import { cn } from "@/lib/utils";

const NODE_TYPES: GraphNodeType[] = [
  "episode",
  "entity",
  "artifact",
  "agent",
  "evidence",
  "action",
  "guardrail",
];

export function GraphFilterBar({
  filters,
  onChange,
}: {
  filters: GraphFilterState;
  onChange: (f: GraphFilterState) => void;
}) {
  const set = (patch: Partial<GraphFilterState>) =>
    onChange({ ...filters, ...patch });

  const sourceVal = filters.evidenceSources.has("all")
    ? "all"
    : ([...filters.evidenceSources][0] ?? "all");

  return (
    <div className="bg-card/50 flex flex-col gap-3 rounded-lg border p-3 text-sm">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <Switch
            id="cited-only"
            checked={filters.onlyCitedEvidence}
            onCheckedChange={(v) => set({ onlyCitedEvidence: v })}
          />
          <Label htmlFor="cited-only" className="text-muted-foreground">
            Only cited evidence
          </Label>
        </div>
        <div className="flex items-center gap-2">
          <Switch
            id="actionable"
            checked={filters.onlyActionable}
            onCheckedChange={(v) => set({ onlyActionable: v })}
          />
          <Label htmlFor="actionable" className="text-muted-foreground">
            Actionable only
          </Label>
        </div>
        <div className="flex items-center gap-2">
          <Switch
            id="expand-ev"
            checked={filters.expandAllEvidence}
            onCheckedChange={(v) => set({ expandAllEvidence: v })}
          />
          <Label htmlFor="expand-ev" className="text-muted-foreground">
            Expand all evidence
          </Label>
        </div>
      </div>
      <div className="flex flex-wrap items-end gap-4">
        <div className="flex flex-col gap-1">
          <span className="text-muted-foreground text-xs">Evidence source</span>
          <select
            className={cn(
              "border-input bg-background h-8 w-[140px] rounded-md border px-2 text-sm",
            )}
            value={sourceVal}
            onChange={(e) => {
              const v = e.target.value;
              if (v === "all") {
                set({ evidenceSources: new Set(["all"]) });
              } else {
                set({
                  evidenceSources: new Set([v as EvidenceSource]),
                });
              }
            }}
          >
            <option value="all">All sources</option>
            <option value="kb">KB</option>
            <option value="opencti">CTI</option>
            <option value="taxii">TAXII</option>
            <option value="other">Other</option>
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-muted-foreground text-xs">Top evidence limit</span>
          <select
            className="border-input bg-background h-8 w-[100px] rounded-md border px-2 text-sm"
            value={String(filters.topEvidenceLimit)}
            onChange={(e) =>
              set({ topEvidenceLimit: Number.parseInt(e.target.value, 10) || 8 })
            }
          >
            {[4, 8, 12, 24].map((n) => (
              <option key={n} value={String(n)}>
                {n}
              </option>
            ))}
          </select>
        </div>
        <button
          type="button"
          className="text-primary text-xs underline-offset-4 hover:underline"
          onClick={() => onChange(defaultGraphFilter())}
        >
          Reset filters
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        <span className="text-muted-foreground w-full text-xs">Node types</span>
        {NODE_TYPES.map((t) => {
          const on = filters.nodeTypes.has(t);
          return (
            <button
              key={t}
              type="button"
              className={`rounded border px-2 py-0.5 text-xs capitalize ${
                on
                  ? "border-primary bg-primary/10"
                  : "text-muted-foreground border-transparent opacity-60"
              }`}
              onClick={() => {
                const next = new Set(filters.nodeTypes);
                if (next.has(t)) next.delete(t);
                else next.add(t);
                if (next.size === 0) return;
                set({ nodeTypes: next });
              }}
            >
              {t.replace("_", " ")}
            </button>
          );
        })}
      </div>
    </div>
  );
}
