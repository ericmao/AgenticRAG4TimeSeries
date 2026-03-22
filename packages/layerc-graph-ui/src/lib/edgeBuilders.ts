import type { GraphEdgeType, GraphLinkData } from "./types";

export function link(
  source: string,
  target: string,
  type: GraphEdgeType,
  label?: string,
): GraphLinkData {
  return { source, target, type, label };
}
