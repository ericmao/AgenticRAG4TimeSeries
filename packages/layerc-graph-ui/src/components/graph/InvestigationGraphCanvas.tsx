"use client";

import {
  actionSeverityColor,
  actionTypeToSeverity,
  triageBucketColor,
  triageLevelToBucket,
} from "@/lib/formatters";
import type { ForceNode, GraphLinkData, GraphMode } from "@/lib/types";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { ForceGraphMethods } from "react-force-graph-2d";

export interface InvestigationGraphCanvasProps {
  nodes: ForceNode[];
  links: GraphLinkData[];
  mode: GraphMode;
  selectedId: string | null;
  onSelectNode: (id: string | null) => void;
  /** 淺色嵌入（Runs 三圖 iframe）：實心背景與較深連線，避免裁切與對比不足 */
  embedded?: boolean;
}

function nodeColor(n: ForceNode): string {
  switch (n.type) {
    case "episode": {
      const b = triageLevelToBucket(n.triageLevel);
      return triageBucketColor(b);
    }
    case "agent":
      return "#a855f7";
    case "evidence":
      return "#22c55e";
    case "action": {
      const sev = actionTypeToSeverity(n.actionType);
      return actionSeverityColor(sev);
    }
    case "entity":
      return "#38bdf8";
    case "artifact":
      return "#94a3b8";
    case "guardrail":
      return "#f97316";
    default:
      return "#64748b";
  }
}

export function InvestigationGraphCanvas({
  nodes,
  links,
  mode,
  selectedId,
  onSelectNode,
  embedded = false,
}: InvestigationGraphCanvasProps) {
  const [hoverLink, setHoverLink] = useState<GraphLinkData | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<ForceGraphMethods | null>(null);
  const [dims, setDims] = useState({ width: 800, height: 480 });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => {
      requestAnimationFrame(() => {
        const box = wrapRef.current;
        if (!box) return;
        const w = box.clientWidth;
        const h = box.clientHeight;
        if (w >= 48 && h >= 48) {
          setDims({ width: w, height: h });
        }
      });
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);

  const graphData = useMemo(() => {
    const nds = nodes.map((n) => {
      const copy = { ...n };
      if (n.type === "episode") {
        copy.fx = 0;
        copy.fy = 0;
      } else {
        copy.fx = undefined;
        copy.fy = undefined;
      }
      return copy;
    });
    const lks = links.map((l) => ({
      ...l,
      source: l.source,
      target: l.target,
    }));
    return { nodes: nds, links: lks };
  }, [nodes, links]);

  const fitGraph = useCallback(() => {
    if (dims.width < 48 || dims.height < 48) return;
    const fg = fgRef.current;
    if (!fg) return;
    try {
      fg.zoomToFit(500, embedded ? 36 : 28);
    } catch {
      /* ignore */
    }
  }, [dims.width, dims.height, embedded]);

  useEffect(() => {
    const id = window.setTimeout(fitGraph, embedded ? 520 : 420);
    return () => window.clearTimeout(id);
  }, [fitGraph, links.length, mode, nodes.length]);

  const paintNode = useCallback(
    (node: object, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as ForceNode;
      const label = n.label ?? n.id;
      const size =
        n.type === "episode" ? 10 : n.type === "evidence" ? 5 : 7;
      const x = n.x ?? 0;
      const y = n.y ?? 0;
      const isSel = n.id === selectedId;

      ctx.beginPath();
      if (n.type === "episode") {
        for (let i = 0; i < 6; i++) {
          const a = (Math.PI / 3) * i - Math.PI / 6;
          const px = x + size * Math.cos(a) * 1.2;
          const py = y + size * Math.sin(a) * 1.2;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
      } else if (n.type === "action") {
        ctx.moveTo(x, y - size);
        ctx.lineTo(x + size, y);
        ctx.lineTo(x, y + size);
        ctx.lineTo(x - size, y);
        ctx.closePath();
      } else {
        ctx.arc(x, y, size, 0, 2 * Math.PI);
      }

      ctx.fillStyle = nodeColor(n);
      ctx.globalAlpha = 0.9;
      ctx.fill();
      if (isSel) {
        ctx.strokeStyle = embedded ? "#0f172a" : "#f8fafc";
        ctx.lineWidth = 2 / globalScale;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      const fontSize = Math.max(10 / globalScale, 3);
      ctx.font = `${fontSize}px Inter, system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = embedded
        ? "rgba(15,23,42,0.92)"
        : "rgba(248,250,252,0.92)";
      const lines = label.length > 28 ? `${label.slice(0, 26)}…` : label;
      ctx.fillText(lines, x, y + size + 2 / globalScale);
    },
    [embedded, selectedId],
  );

  const bg = embedded ? "rgb(241 245 249)" : "transparent";
  const linkRgb = embedded
    ? "rgba(51,65,85,0.55)"
    : "rgba(148,163,184,0.45)";

  return (
    <div
      ref={wrapRef}
      className={
        embedded
          ? "relative min-h-[280px] w-full min-w-0 flex-1 overflow-hidden rounded-md border border-slate-300/80 bg-slate-100 lg:min-h-0"
          : "bg-muted/30 relative min-h-[min(320px,45vh)] w-full min-w-0 flex-1 overflow-hidden rounded-lg border lg:min-h-0"
      }
    >
      {hoverLink && (
        <div
          className={
            embedded
              ? "pointer-events-none absolute top-2 left-2 z-10 rounded border border-slate-200 bg-white/95 px-2 py-1 text-xs text-slate-800 shadow"
              : "bg-background/90 pointer-events-none absolute top-2 left-2 z-10 rounded border px-2 py-1 text-xs shadow"
          }
        >
          {hoverLink.type}
          {hoverLink.label ? ` — ${hoverLink.label}` : ""}
        </div>
      )}
      <ForceGraph2D
        ref={fgRef}
        width={dims.width}
        height={dims.height}
        graphData={
          graphData as {
            nodes: object[];
            links: object[];
          }
        }
        backgroundColor={bg}
        linkColor={() => linkRgb}
        linkDirectionalParticles={mode === "citation" ? 1 : 0}
        linkDirectionalParticleWidth={2}
        linkWidth={1}
        nodeLabel={() => ""}
        nodeCanvasObject={paintNode}
        nodePointerAreaPaint={(node, color, ctx) => {
          const n = node as ForceNode;
          const size = n.type === "episode" ? 12 : 9;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(n.x ?? 0, n.y ?? 0, size, 0, 2 * Math.PI);
          ctx.fill();
        }}
        onNodeClick={(n) => {
          onSelectNode((n as ForceNode).id);
        }}
        onBackgroundClick={() => onSelectNode(null)}
        onLinkHover={(l) => {
          setHoverLink(l ? (l as unknown as GraphLinkData) : null);
        }}
        cooldownTicks={mode === "case" ? 120 : 80}
        d3VelocityDecay={0.25}
      />
    </div>
  );
}
