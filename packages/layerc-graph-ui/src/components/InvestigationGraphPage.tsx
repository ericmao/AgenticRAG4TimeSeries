"use client";

import { InvestigationSummaryCards } from "@/components/InvestigationSummaryCards";
import { InvestigationToolbar } from "@/components/InvestigationToolbar";
import { GraphFilterBar } from "@/components/graph/GraphFilterBar";
import { GraphLegend } from "@/components/graph/GraphLegend";
import { InvestigationGraphCanvas } from "@/components/graph/InvestigationGraphCanvas";
import { InspectorPanel } from "@/components/inspector/InspectorPanel";
import { EpisodeListPanel } from "@/components/lists/EpisodeListPanel";
import { buildGraphFromCase } from "@/lib/graphTransform";
import { applyGraphView } from "@/lib/selectors";
import type { GraphMode, LayerCCasePayload } from "@/lib/types";
import { defaultGraphFilter } from "@/lib/types";
import { useMemo, useState } from "react";
import type { EpisodeListEntry } from "@/components/lists/EpisodeListPanel";

export function InvestigationGraphPage({
  payload,
  episodeList,
  listHrefBase = "/investigations/graph",
  title = "Layer C Investigation Graph",
  initialMode,
  embedded = false,
}: {
  payload: LayerCCasePayload;
  episodeList: EpisodeListEntry[];
  /** Prefix for episode links in the case navigator, e.g. `/investigations/graph?run_id=` */
  listHrefBase?: string;
  title?: string;
  /** Initial graph mode (e.g. from `?mode=`). In `embedded` mode the mode stays fixed. */
  initialMode?: GraphMode;
  /** Minimal layout for iframe embeds (Runs 列表三圖). */
  embedded?: boolean;
}) {
  const graph = useMemo(() => buildGraphFromCase(payload), [payload]);

  const [mode, setMode] = useState<GraphMode>(() => initialMode ?? "case");
  const [filters, setFilters] = useState(defaultGraphFilter);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const graphMode: GraphMode = embedded ? (initialMode ?? "case") : mode;

  const filtered = useMemo(
    () => applyGraphView(graph, graphMode, filters),
    [graph, graphMode, filters],
  );

  const nodeMap = useMemo(() => {
    const m = new Map<string, (typeof graph.nodes)[0]>();
    for (const n of graph.nodes) m.set(n.id, n);
    return m;
  }, [graph]);

  if (embedded) {
    return (
      <div className="embed-graph-root text-slate-800 flex h-full min-h-0 flex-1 flex-col gap-1 overflow-hidden p-1.5">
        <GraphLegend compact />
        <div className="flex min-h-0 flex-1 flex-col">
          <InvestigationGraphCanvas
            nodes={filtered.nodes}
            links={filtered.links}
            mode={graphMode}
            selectedId={selectedId}
            onSelectNode={setSelectedId}
            embedded
          />
        </div>
      </div>
    );
  }

  return (
    <div className="box-border flex min-h-0 flex-1 flex-col overflow-hidden bg-muted/40 px-3 pb-4 pt-2 sm:px-4 md:px-5">
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-sm ring-1 ring-neutral-950/6">
        <header className="shrink-0 space-y-4 border-b border-border/80 p-4 md:p-5">
          <InvestigationToolbar
            title={title}
            mode={mode}
            onModeChange={setMode}
          />
          <InvestigationSummaryCards payload={payload} graph={graph} />
          <GraphFilterBar filters={filters} onChange={setFilters} />
        </header>

        <main className="grid min-h-0 flex-1 grid-cols-1 gap-4 bg-muted/25 p-3 lg:min-h-[min(560px,calc(100dvh-13.5rem))] lg:grid-cols-12 lg:grid-rows-1 lg:gap-0 lg:p-0">
        <section className="flex min-h-[min(420px,55vh)] flex-col gap-2 lg:col-span-6 lg:h-full lg:min-h-0 lg:border-r lg:border-border/50 lg:p-3">
          <GraphLegend />
          <InvestigationGraphCanvas
            nodes={filtered.nodes}
            links={filtered.links}
            mode={mode}
            selectedId={selectedId}
            onSelectNode={setSelectedId}
          />
        </section>

        <aside className="flex min-h-[200px] flex-col lg:col-span-3 lg:h-full lg:min-h-0 lg:border-r lg:border-border/50 lg:p-3">
          <EpisodeListPanel
            episodes={episodeList}
            currentEpisodeId={payload.episode_id}
            hrefBase={listHrefBase}
          />
        </aside>

        <aside className="flex min-h-[200px] flex-col lg:col-span-3 lg:h-full lg:min-h-0 lg:p-3">
          <p className="text-muted-foreground mb-2 shrink-0 text-xs font-medium uppercase">
            Inspector
          </p>
          <div className="flex min-h-0 flex-1 flex-col">
            <InspectorPanel
              selectedId={selectedId}
              payload={payload}
              graph={graph}
              nodeMap={nodeMap}
            />
          </div>
        </aside>
      </main>
      </div>
    </div>
  );
}
