"use client";

import { Input } from "@/ui/input";
import { ScrollArea } from "@/ui/scroll-area";
import { useMemo, useState } from "react";

export interface EpisodeListEntry {
  episode_id: string;
  triage_level: string;
  alert_count: number;
  top_entity: string;
  top_action: string;
  status: string;
  /** e.g. cert / wazuh — from analysis_runs.source when provided */
  source?: string;
  /** When set (e.g. MVP UI), links use `?run_id=` instead of path segment. */
  run_id?: number;
}

function defaultHref(base: string, e: EpisodeListEntry): string {
  if (e.run_id != null) {
    const sep = base.includes("?") ? "&" : "?";
    return `${base}${sep}run_id=${e.run_id}`;
  }
  const b = base.replace(/\/$/, "");
  return `${b}/${encodeURIComponent(e.episode_id)}`;
}

export function EpisodeListPanel({
  episodes,
  currentEpisodeId,
  hrefBase = "/dashboard/investigations",
  getEpisodeHref,
}: {
  episodes: EpisodeListEntry[];
  currentEpisodeId?: string;
  /** Base URL for links; with `run_id` on entry appends `?run_id=`. */
  hrefBase?: string;
  getEpisodeHref?: (e: EpisodeListEntry) => string;
}) {
  const [q, setQ] = useState("");
  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return episodes;
    return episodes.filter(
      (e) =>
        e.episode_id.toLowerCase().includes(s) ||
        e.top_entity.toLowerCase().includes(s) ||
        e.triage_level.toLowerCase().includes(s) ||
        (e.source && e.source.toLowerCase().includes(s)),
    );
  }, [episodes, q]);

  const resolveHref = (e: EpisodeListEntry) =>
    getEpisodeHref ? getEpisodeHref(e) : defaultHref(hrefBase, e);

  return (
    <div className="bg-card flex min-h-0 flex-1 flex-col gap-3 rounded-lg border p-3">
      <div>
        <p className="text-muted-foreground mb-1 text-xs font-medium uppercase">
          Case Navigator
        </p>
        <Input
          placeholder="Search episodes…"
          value={q}
          onChange={(ev) => setQ(ev.target.value)}
          className="h-9"
        />
      </div>
      <ScrollArea className="min-h-[200px] flex-1 pr-2">
        <ul className="space-y-2">
          {filtered.map((e) => {
            const active = e.episode_id === currentEpisodeId;
            return (
              <li key={`${e.episode_id}-${e.run_id ?? ""}`}>
                <a
                  href={resolveHref(e)}
                  className={`block rounded-md border px-3 py-2 text-sm transition-colors ${
                    active
                      ? "border-primary bg-primary/10"
                      : "hover:bg-muted/80 border-transparent"
                  }`}
                >
                  <div className="font-mono text-xs font-medium">{e.episode_id}</div>
                  <div className="text-muted-foreground mt-1 flex flex-wrap gap-x-2 gap-y-0.5 text-[11px]">
                    {e.source ? (
                      <>
                        <span className="rounded bg-muted px-1.5 py-0 font-medium">
                          {e.source}
                        </span>
                        <span>·</span>
                      </>
                    ) : null}
                    <span className="capitalize">{e.triage_level}</span>
                    <span>·</span>
                    <span>{e.alert_count} alerts</span>
                  </div>
                  <div className="text-muted-foreground mt-0.5 text-[11px]">
                    {e.top_entity} · {e.top_action}
                  </div>
                </a>
              </li>
            );
          })}
        </ul>
      </ScrollArea>
    </div>
  );
}
