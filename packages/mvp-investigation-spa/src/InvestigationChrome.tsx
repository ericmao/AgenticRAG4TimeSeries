import type { LayerCCasePayload } from "@agentic/layerc-graph-ui";
import type { ReactNode } from "react";

function senseLProductUrl(episodeId: string): string | null {
  const raw = import.meta.env.VITE_SENSEL_ORIGIN as string | undefined;
  if (!raw?.trim() || !episodeId) return null;
  const base = raw.replace(/\/$/, "");
  return `${base}/dashboard/investigations/${encodeURIComponent(episodeId)}`;
}

/**
 * 對齊 senseL `TopNavLayout` + 麵包屑：Runs → 調查列表 → episode；可選連回產品端同一路徑。
 */
export function InvestigationChrome({
  payload,
  runIdLabel,
  children,
}: {
  payload: LayerCCasePayload | null;
  runIdLabel?: string;
  children: ReactNode;
}) {
  const productUrl =
    payload?.episode_id != null
      ? senseLProductUrl(String(payload.episode_id))
      : null;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="border-border/60 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 w-full border-b backdrop-blur">
        <div className="flex min-h-14 flex-wrap items-center gap-x-2 gap-y-1 px-4 py-2.5 text-sm">
          <a
            href="/runs"
            className="text-muted-foreground hover:text-foreground shrink-0 font-medium"
          >
            ← Runs
          </a>
          <span className="text-muted-foreground" aria-hidden>
            ·
          </span>
          <a
            href="/investigations/graph"
            className="text-muted-foreground hover:text-foreground shrink-0"
          >
            調查列表
          </a>
          {payload ? (
            <>
              <span className="text-muted-foreground" aria-hidden>
                /
              </span>
              <span className="text-foreground max-w-[min(100%,28rem)] truncate font-mono text-xs font-semibold md:text-sm">
                {payload.episode_id || "—"}
              </span>
              {runIdLabel ? (
                <span className="text-muted-foreground hidden text-xs sm:inline">
                  （{runIdLabel}）
                </span>
              ) : null}
            </>
          ) : null}
          {productUrl ? (
            <a
              href={productUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary ml-auto shrink-0 text-xs font-medium underline-offset-4 hover:underline"
            >
              在 SenseL 開啟
            </a>
          ) : null}
        </div>
      </header>
      <div className="min-h-0 flex-1 overflow-hidden">{children}</div>
    </div>
  );
}
