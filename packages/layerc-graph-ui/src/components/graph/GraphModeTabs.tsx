"use client";

import type { GraphMode } from "@/lib/types";
import { cn } from "@/lib/utils";

const MODES: { id: GraphMode; label: string }[] = [
  { id: "case", label: "案件關聯圖" },
  { id: "citation", label: "證據引用圖" },
  { id: "decision", label: "決策路徑圖" },
];

export function GraphModeTabs({
  value,
  onChange,
}: {
  value: GraphMode;
  onChange: (m: GraphMode) => void;
}) {
  return (
    <div
      className="inline-flex w-fit flex-wrap items-center gap-1 rounded-md border p-[3px]"
      role="tablist"
    >
      {MODES.map((m) => (
        <button
          key={m.id}
          type="button"
          role="tab"
          aria-selected={value === m.id}
          className={cn(
            "rounded-sm px-3 py-1.5 text-xs sm:text-sm transition-colors",
            value === m.id
              ? "bg-muted text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
          )}
          onClick={() => onChange(m.id)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
