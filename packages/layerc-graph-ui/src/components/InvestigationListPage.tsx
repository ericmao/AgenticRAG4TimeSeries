"use client";

import {
  EpisodeListPanel,
  type EpisodeListEntry,
} from "@/components/lists/EpisodeListPanel";

export function InvestigationListPage({
  episodes,
  hrefBase = "/dashboard/investigations",
  runsHref = "/runs",
}: {
  episodes: EpisodeListEntry[];
  hrefBase?: string;
  /** 完整分析紀錄（C1/C2/C3、篩選）— 與本列表同源 DB，欄位較完整 */
  runsHref?: string;
}) {
  return (
    <div className="bg-background flex min-h-0 flex-1 flex-col gap-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="text-foreground text-lg font-semibold tracking-tight">
            Layer C 調查列表
          </h1>
          <p className="text-muted-foreground mt-1 max-w-2xl text-sm">
            自 <code className="rounded bg-muted px-1 py-0.5 text-xs">analysis_runs</code>{" "}
            載入可開圖的 run；若需完整管線狀態、篩選與手動觸發，請至 Analysis runs。
          </p>
        </div>
        <a
          href={runsHref}
          className="text-primary shrink-0 text-sm font-medium underline underline-offset-4 hover:no-underline"
        >
          開啟 Analysis runs（{runsHref}）
        </a>
      </div>

      {episodes.length === 0 ? (
        <div className="text-muted-foreground rounded-lg border border-dashed p-6 text-sm">
          <p>尚無可顯示的調查紀錄。</p>
          <p className="mt-2">
            請確認已寫入資料庫，或前往{" "}
            <a className="text-primary font-medium underline" href={runsHref}>
              Analysis runs
            </a>{" "}
            查看完整列表與操作。
          </p>
        </div>
      ) : (
        <div className="min-h-0 min-w-0 flex-1">
          <EpisodeListPanel episodes={episodes} hrefBase={hrefBase} />
        </div>
      )}
    </div>
  );
}
