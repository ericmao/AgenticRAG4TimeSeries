/** Backend triage_level values → UI bucket for colors */
export function triageLevelToBucket(
  level: string | undefined,
): "low" | "suspicious" | "high" | "unknown" {
  if (!level) return "unknown";
  const t = level.toLowerCase();
  if (t === "noise" || t === "low") return "low";
  if (t === "suspicious" || t === "medium") return "suspicious";
  if (t === "critical" || t === "high") return "high";
  return "unknown";
}

export function triageBucketColor(bucket: ReturnType<typeof triageLevelToBucket>): string {
  switch (bucket) {
    case "low":
      return "#3b82f6";
    case "suspicious":
      return "#eab308";
    case "high":
      return "#ef4444";
    default:
      return "#64748b";
  }
}

export function actionTypeToSeverity(
  action: string | undefined,
): "collect" | "watchlist" | "isolate" | "block" | "unknown" {
  if (!action) return "unknown";
  const a = action.toLowerCase();
  if (a === "collect_more_data") return "collect";
  if (a === "watchlist") return "watchlist";
  if (a === "isolate") return "isolate";
  if (a === "block") return "block";
  return "unknown";
}

export function actionSeverityColor(
  sev: ReturnType<typeof actionTypeToSeverity>,
): string {
  switch (sev) {
    case "collect":
      return "#3b82f6";
    case "watchlist":
      return "#eab308";
    case "isolate":
      return "#f97316";
    case "block":
      return "#ef4444";
    default:
      return "#94a3b8";
  }
}

export function evidenceSourceLabel(source: string | undefined): string {
  if (!source) return "—";
  const s = String(source);
  if (s === "kb") return "KB";
  if (s === "opencti") return "CTI";
  if (s === "taxii") return "TAXII";
  return s;
}

export function formatConfidence(c: number | undefined): string {
  if (c === undefined || Number.isNaN(c)) return "—";
  return `${Math.round(c * 100)}%`;
}
