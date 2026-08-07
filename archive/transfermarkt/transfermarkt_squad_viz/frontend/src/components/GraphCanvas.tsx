import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { LinkObject } from "react-force-graph-2d";
import type { GraphData, GraphLink, GraphNode } from "../types";
import { LEAGUE_NODE_ID } from "../types";
import { getTeamColor } from "../utils/teamColors";

interface GraphCanvasProps {
  graph: GraphData;
  selection:
    | { kind: "league" }
    | { kind: "team"; club: string }
    | { kind: "player"; tmId: number }
    | null;
  hoverId: string | null;
  onHover: (id: string | null) => void;
  onSelectLeague: () => void;
  onSelectTeam: (club: string) => void;
  onSelectPlayer: (tmId: number) => void;
  searchFocusId: string | null;
}

const LEAGUE_COLOR = "#d4af37";
const LEAGUE_GLOW = "rgba(212, 175, 55, 0.35)";

export function GraphCanvas({
  graph,
  selection,
  hoverId,
  onHover,
  onSelectLeague,
  onSelectTeam,
  onSelectPlayer,
  searchFocusId,
}: GraphCanvasProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ w: 800, h: 600 });

  const graphData = useMemo(
    () => ({ nodes: graph.nodes, links: graph.links }),
    [graph],
  );

  const spotlightIds = useMemo(() => {
    const set = new Set<string>();
    if (!selection) return set;
    if (selection.kind === "league") {
      set.add(LEAGUE_NODE_ID);
      graph.nodes
        .filter((n) => n.type === "team")
        .forEach((n) => set.add(n.id));
      return set;
    }
    if (selection.kind === "team") {
      set.add(`team:${selection.club}`);
      graph.nodes
        .filter((n) => n.type === "player" && n.club === selection.club)
        .forEach((n) => set.add(n.id));
      return set;
    }
    const pid = `player:${selection.tmId}`;
    set.add(pid);
    const player = graph.nodes.find((n) => n.id === pid);
    if (player) set.add(`team:${player.club}`);
    return set;
  }, [selection, graph.nodes]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      setDims({
        w: entry.contentRect.width,
        h: entry.contentRect.height,
      });
    });
    ro.observe(el);
    setDims({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    fg.d3Force("charge")?.strength((node: GraphNode) => {
      if (node.type === "league") return 0;
      if (node.type === "team") return -300;
      return -28;
    });
    fg.d3Force("link")?.distance((link: LinkObject<GraphNode, GraphLink>) => {
      const s = link.source as GraphNode;
      const t = link.target as GraphNode;
      if (s.type === "league" || t.type === "league") return 105;
      if (s.type === "player" || t.type === "player") return 68;
      return 72;
    });
  }, [graphData]);

  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || !searchFocusId) return;
    const node = graph.nodes.find((n) => n.id === searchFocusId);
    if (node?.x != null && node?.y != null) {
      fg.centerAt(node.x, node.y, 600);
      fg.zoom(searchFocusId === LEAGUE_NODE_ID ? 1.2 : 2.8, 600);
    }
  }, [searchFocusId, graph.nodes]);

  const zoomToClub = useCallback(
    (club: string) => {
      const fg = fgRef.current;
      if (!fg) return;
      const cluster = graph.nodes.filter(
        (n) => n.club === club || (n.type === "team" && n.label === club),
      );
      if (cluster.length === 0) return;
      let minX = Infinity,
        maxX = -Infinity,
        minY = Infinity,
        maxY = -Infinity;
      for (const n of cluster) {
        if (n.x == null || n.y == null) continue;
        minX = Math.min(minX, n.x);
        maxX = Math.max(maxX, n.x);
        minY = Math.min(minY, n.y);
        maxY = Math.max(maxY, n.y);
      }
      if (!Number.isFinite(minX)) return;
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      const span = Math.max(maxX - minX, maxY - minY, 80);
      const { w, h } = dims;
      const scale = Math.min(w, h) / (span * 1.6);
      fg.centerAt(cx, cy, 700);
      fg.zoom(Math.min(Math.max(scale, 0.8), 4), 700);
    },
    [graph.nodes, dims],
  );

  const zoomToLeague = useCallback(() => {
    const fg = fgRef.current;
    if (!fg) return;
    fg.centerAt(0, 0, 700);
    fg.zoom(1.15, 700);
  }, []);

  useEffect(() => {
    if (selection?.kind === "team") {
      zoomToClub(selection.club);
    } else if (selection?.kind === "league") {
      zoomToLeague();
    }
  }, [selection, zoomToClub, zoomToLeague]);

  const paintNode = useCallback(
    (node: GraphNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const isLeague = node.type === "league";
      const isTeam = node.type === "team";
      const color = isLeague ? LEAGUE_COLOR : getTeamColor(node.club);
      const r = isLeague ? 22 : isTeam ? 14 : Math.max(3, node.val * 0.55);
      const dimmed =
        spotlightIds.size > 0 && !spotlightIds.has(node.id) && hoverId !== node.id;
      const hovered = hoverId === node.id;
      const selected =
        (selection?.kind === "league" && node.id === LEAGUE_NODE_ID) ||
        (selection?.kind === "team" && node.id === `team:${selection.club}`) ||
        (selection?.kind === "player" && node.id === `player:${selection.tmId}`);

      if (isLeague && !dimmed) {
        ctx.beginPath();
        ctx.arc(node.x!, node.y!, r + 8 / globalScale, 0, 2 * Math.PI);
        ctx.fillStyle = LEAGUE_GLOW;
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI);
      ctx.fillStyle = dimmed ? "rgba(60, 65, 80, 0.35)" : color;
      ctx.fill();

      if (isLeague || isTeam) {
        ctx.strokeStyle = dimmed
          ? "rgba(255,255,255,0.1)"
          : isLeague
            ? "rgba(255, 235, 180, 0.65)"
            : "rgba(255,255,255,0.35)";
        ctx.lineWidth = (isLeague ? 2.5 : 2) / globalScale;
        ctx.stroke();
      }

      if (hovered || selected) {
        ctx.beginPath();
        ctx.arc(node.x!, node.y!, r + 3 / globalScale, 0, 2 * Math.PI);
        ctx.strokeStyle = selected ? "#ffffff" : "rgba(255,255,255,0.6)";
        ctx.lineWidth = (selected ? 2.5 : 1.5) / globalScale;
        ctx.stroke();
      }

      const showLabel =
        isLeague || isTeam || globalScale > 1.8 || hovered || selected;
      if (showLabel) {
        const label = isLeague
          ? node.label
          : isTeam
            ? node.label.split(" ")[0]
            : node.label.split(" ").slice(-1)[0];
        const fontSize = isLeague ? 13 / globalScale : isTeam ? 12 / globalScale : 10 / globalScale;
        ctx.font = `${isLeague || isTeam ? 600 : 400} ${Math.max(fontSize, 3)}px "Segoe UI", system-ui, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = dimmed ? "rgba(200,205,220,0.25)" : "rgba(230,235,245,0.92)";
        ctx.fillText(label, node.x!, node.y! + r + 2 / globalScale);
      }
    },
    [hoverId, selection, spotlightIds],
  );

  const linkColor = useCallback(
    (link: LinkObject<GraphNode, GraphLink>) => {
      const s = link.source as GraphNode;
      const t = link.target as GraphNode;
      const isLeagueLink = s.type === "league" || t.type === "league";
      const inSpot =
        spotlightIds.size === 0 ||
        (spotlightIds.has(s.id) && spotlightIds.has(t.id));
      const hovered =
        hoverId != null && (s.id === hoverId || t.id === hoverId);
      if (hovered) return "rgba(255,255,255,0.55)";
      if (!inSpot) return "rgba(80, 85, 100, 0.12)";
      if (isLeagueLink) return "rgba(212, 175, 55, 0.45)";
      return "rgba(140, 150, 175, 0.35)";
    },
    [hoverId, spotlightIds],
  );

  const linkWidth = useCallback(
    (link: LinkObject<GraphNode, GraphLink>) => {
      const s = link.source as GraphNode;
      const t = link.target as GraphNode;
      if (s.type === "league" || t.type === "league") return 1.8;
      return 1;
    },
    [],
  );

  const handleNodeClick = useCallback(
    (node: GraphNode) => {
      if (node.type === "league") {
        onSelectLeague();
        zoomToLeague();
      } else if (node.type === "team") {
        onSelectTeam(node.club);
        zoomToClub(node.club);
      } else if (node.tm_id != null) {
        onSelectPlayer(node.tm_id);
      }
    },
    [onSelectLeague, onSelectTeam, onSelectPlayer, zoomToClub, zoomToLeague],
  );

  return (
    <div ref={containerRef} className="graph-canvas">
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        width={dims.w}
        height={dims.h}
        backgroundColor="#0d0f14"
        nodeRelSize={1}
        linkWidth={linkWidth}
        linkColor={linkColor}
        cooldownTicks={140}
        d3AlphaDecay={0.028}
        d3VelocityDecay={0.38}
        enableNodeDrag
        onNodeClick={(node) => handleNodeClick(node as GraphNode)}
        onNodeHover={(node) => onHover(node ? (node as GraphNode).id : null)}
        nodeCanvasObject={paintNode}
        nodePointerAreaPaint={(node, color, ctx) => {
          const n = node as GraphNode;
          const r =
            n.type === "league" ? 28 : n.type === "team" ? 18 : Math.max(5, n.val * 0.7);
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(n.x!, n.y!, r, 0, 2 * Math.PI);
          ctx.fill();
        }}
      />
    </div>
  );
}
