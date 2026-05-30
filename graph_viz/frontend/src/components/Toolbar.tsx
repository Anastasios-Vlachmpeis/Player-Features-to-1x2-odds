import { useMemo, useState } from "react";
import type { GraphData, Selection } from "../types";

interface ToolbarProps {
  graph: GraphData;
  selection: Selection;
  onSelectTeam: (club: string) => void;
  onSelectLeague: () => void;
  onClear: () => void;
  onSearchSelect: (nodeId: string) => void;
}

export function Toolbar({
  graph,
  selection,
  onSelectTeam,
  onSelectLeague,
  onClear,
  onSearchSelect,
}: ToolbarProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const clubs = useMemo(() => {
    return graph.nodes
      .filter((n) => n.type === "team")
      .map((n) => n.club)
      .sort();
  }, [graph.nodes]);

  const selectedClub = selection?.kind === "team" ? selection.club : "";

  const suggestions = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (q.length < 2) return [];
    return graph.nodes
      .filter((n) => n.label.toLowerCase().includes(q))
      .slice(0, 8);
  }, [query, graph.nodes]);

  const handleSearchPick = (nodeId: string) => {
    setQuery("");
    setOpen(false);
    onSearchSelect(nodeId);
  };

  return (
    <header className="toolbar">
      <div className="toolbar-brand">
        <span className="brand-title">Super League Squad Graph</span>
        <span className="brand-meta">
          {graph.meta.teams} teams · {graph.meta.players} players
        </span>
      </div>
      <div className="toolbar-controls">
        <label className="filter-label">
          Focus team
          <select
            value={selectedClub}
            onChange={(e) => {
              const club = e.target.value;
              if (!club) {
                onClear();
                onSelectLeague();
              } else {
                onSelectTeam(club);
              }
            }}
          >
            <option value="">Full league view</option>
            {clubs.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <div className="search-wrap">
          <input
            type="search"
            placeholder="Search player or team…"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            onBlur={() => setTimeout(() => setOpen(false), 150)}
          />
          {open && suggestions.length > 0 && (
            <ul className="search-suggestions">
              {suggestions.map((n) => (
                <li key={n.id}>
                  <button type="button" onMouseDown={() => handleSearchPick(n.id)}>
                    <span className={`suggestion-type ${n.type}`}>
                      {n.type === "league" ? "League" : n.type === "team" ? "Team" : "Player"}
                    </span>
                    {n.label}
                    {n.type === "player" && (
                      <span className="suggestion-club">{n.club}</span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </header>
  );
}
