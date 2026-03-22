"use client";

export function GraphLegend({ compact = false }: { compact?: boolean }) {
  const items = [
    { c: "#3b82f6", label: "Episode (low / noise)" },
    { c: "#eab308", label: "Episode (suspicious)" },
    { c: "#ef4444", label: "Episode (high / critical)" },
    { c: "#a855f7", label: "Agent" },
    { c: "#22c55e", label: "Evidence" },
    { c: "#3b82f6", label: "Action (collect)" },
    { c: "#eab308", label: "Action (watchlist)" },
    { c: "#f97316", label: "Action (isolate)" },
    { c: "#38bdf8", label: "Entity" },
    { c: "#f97316", label: "Guardrail" },
  ];
  if (compact) {
    return (
      <div className="text-muted-foreground max-h-8 shrink-0 overflow-x-auto overflow-y-hidden border-b border-border/20 pb-1.5">
        <div className="flex w-max min-w-full flex-nowrap gap-x-2.5 text-[9px] leading-tight">
          {items.map((i) => (
            <span key={i.label} className="inline-flex shrink-0 items-center gap-0.5 whitespace-nowrap">
              <span
                className="inline-block size-2 shrink-0 rounded-sm"
                style={{ background: i.c }}
              />
              {i.label}
            </span>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="text-muted-foreground flex shrink-0 flex-wrap gap-x-3 gap-y-1 border-b border-border/30 pb-2 text-[10px] sm:gap-x-4 sm:text-xs">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1">
          <span
            className="inline-block size-2.5 rounded-sm"
            style={{ background: i.c }}
          />
          {i.label}
        </span>
      ))}
    </div>
  );
}
