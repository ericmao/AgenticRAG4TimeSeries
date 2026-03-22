import {
  InvestigationGraphPage,
  InvestigationListPage,
  type EpisodeListEntry,
  type GraphMode,
  type LayerCCasePayload,
} from "@agentic/layerc-graph-ui";
import { StrictMode, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { InvestigationChrome } from "./InvestigationChrome";
import "./index.css";

type CaseRoute =
  | { kind: "list" }
  | { kind: "case"; runId: string | null; episodeId: string | null };

function parseCaseRoute(): CaseRoute {
  const q = new URLSearchParams(window.location.search).get("run_id");
  if (q) {
    return { kind: "case", runId: q, episodeId: null };
  }
  const m = window.location.pathname.match(/^\/investigations\/graph\/([^/]+)\/?$/);
  if (m) {
    const seg = decodeURIComponent(m[1]);
    if (/^\d+$/.test(seg)) {
      return { kind: "case", runId: seg, episodeId: null };
    }
    return { kind: "case", runId: null, episodeId: seg };
  }
  return { kind: "list" };
}

function parseGraphModeFromSearch(): GraphMode | undefined {
  const m = new URLSearchParams(window.location.search).get("mode");
  if (m === "case" || m === "citation" || m === "decision") {
    return m;
  }
  return undefined;
}

function parseEmbeddedFromSearch(): boolean {
  return new URLSearchParams(window.location.search).get("embedded") === "1";
}

function caseFetchUrl(route: Extract<CaseRoute, { kind: "case" }>): string {
  if (route.runId != null) {
    return `/api/investigations/case/${route.runId}`;
  }
  if (route.episodeId != null) {
    return `/api/investigations/case/by-episode/${encodeURIComponent(route.episodeId)}`;
  }
  return "";
}

function App() {
  const [payload, setPayload] = useState<LayerCCasePayload | null>(null);
  const [list, setList] = useState<EpisodeListEntry[]>([]);
  const [caseLoading, setCaseLoading] = useState(false);
  const [listLoading, setListLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const route = useMemo(() => parseCaseRoute(), []);
  const isCase = route.kind === "case";
  const initialMode = useMemo(() => parseGraphModeFromSearch(), []);
  const embedded = useMemo(() => parseEmbeddedFromSearch(), []);

  useEffect(() => {
    if (embedded) {
      setList([]);
      setListLoading(false);
      return;
    }
    fetch("/api/investigations/cases")
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((rows) => setList(Array.isArray(rows) ? rows : []))
      .catch((e) => setError(String(e)))
      .finally(() => setListLoading(false));
  }, [embedded]);

  useEffect(() => {
    if (!isCase) {
      setPayload(null);
      return;
    }
    const url = caseFetchUrl(route);
    if (!url) {
      setPayload(null);
      return;
    }
    setCaseLoading(true);
    setError(null);
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then(setPayload)
      .catch((e) => setError(String(e)))
      .finally(() => setCaseLoading(false));
  }, [isCase, route]);

  if (isCase && error) {
    return (
      <InvestigationChrome payload={null}>
        <div className="bg-muted/25 flex flex-1 flex-col overflow-hidden px-3 pb-4 pt-2 sm:px-4 md:px-5">
          <div className="text-destructive flex flex-1 flex-col rounded-2xl border border-border bg-card p-8 shadow-sm ring-1 ring-black/[0.06]">
            <p className="font-medium">無法載入案件</p>
            <p className="text-muted-foreground mt-2 text-sm">{error}</p>
            <a
              className="text-primary mt-4 inline-block text-sm underline"
              href="/investigations/graph"
            >
              回到調查列表
            </a>
          </div>
        </div>
      </InvestigationChrome>
    );
  }

  if (isCase && caseLoading) {
    return (
      <InvestigationChrome payload={null}>
        <div className="bg-muted/25 flex flex-1 flex-col overflow-hidden px-3 pb-4 pt-2 sm:px-4 md:px-5">
          <div className="text-muted-foreground flex flex-1 flex-col rounded-2xl border border-border bg-card p-8 text-sm shadow-sm ring-1 ring-black/[0.06]">
            載入調查圖…
          </div>
        </div>
      </InvestigationChrome>
    );
  }

  if (isCase && payload) {
    if (embedded) {
      return (
        <div className="flex min-h-0 flex-1 flex-col">
          <InvestigationGraphPage
            payload={payload}
            episodeList={list}
            listHrefBase="/investigations/graph"
            title="Layer C 調查圖"
            initialMode={initialMode}
            embedded
          />
        </div>
      );
    }
    const runLabel =
      route.runId != null ? `Run #${route.runId}` : undefined;
    return (
      <InvestigationChrome payload={payload} runIdLabel={runLabel}>
        <InvestigationGraphPage
          payload={payload}
          episodeList={list}
          listHrefBase="/investigations/graph"
          title="Layer C 調查圖"
          initialMode={initialMode}
          embedded={false}
        />
      </InvestigationChrome>
    );
  }

  if (listLoading) {
    return (
      <InvestigationChrome payload={null}>
        <div className="bg-muted/25 flex flex-1 flex-col overflow-hidden px-3 pb-4 pt-2 sm:px-4 md:px-5">
          <div className="text-muted-foreground flex flex-1 flex-col p-8 text-sm">
            載入列表…
          </div>
        </div>
      </InvestigationChrome>
    );
  }

  if (error && !isCase) {
    return (
      <InvestigationChrome payload={null}>
        <div className="bg-muted/25 flex flex-1 flex-col overflow-hidden px-3 pb-4 pt-2 sm:px-4 md:px-5">
          <div className="text-destructive flex flex-1 flex-col p-8">
            <p>{error}</p>
            <p className="text-muted-foreground mt-2 text-sm">
              請確認已設定 DATABASE_URL 且 MVP UI API 正在執行。
            </p>
          </div>
        </div>
      </InvestigationChrome>
    );
  }

  return (
    <InvestigationChrome payload={null}>
      <div className="bg-muted/25 flex flex-1 flex-col overflow-hidden px-3 pb-4 pt-2 sm:px-4 md:px-5">
        <div className="flex flex-1 flex-col overflow-hidden rounded-2xl border border-border bg-card p-4 shadow-sm ring-1 ring-black/[0.06] md:p-6">
          <InvestigationListPage
            episodes={list}
            hrefBase="/investigations/graph"
            runsHref="/runs"
          />
        </div>
      </div>
    </InvestigationChrome>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <div className="flex min-h-0 flex-1 flex-col">
      <App />
    </div>
  </StrictMode>,
);
