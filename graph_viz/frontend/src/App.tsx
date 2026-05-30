import { useCallback, useEffect, useState } from "react";
import { DetailPanel } from "./components/DetailPanel";
import { GraphCanvas } from "./components/GraphCanvas";
import { Toolbar } from "./components/Toolbar";
import type { GraphData, PlayersMap, Selection } from "./types";
import "./App.css";

function App() {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [players, setPlayers] = useState<PlayersMap | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [searchFocusId, setSearchFocusId] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/data/graph.json").then((r) => {
        if (!r.ok) throw new Error("Failed to load graph.json");
        return r.json() as Promise<GraphData>;
      }),
      fetch("/data/players.json").then((r) => {
        if (!r.ok) throw new Error("Failed to load players.json");
        return r.json() as Promise<PlayersMap>;
      }),
    ])
      .then(([g, p]) => {
        setGraph(g);
        setPlayers(p);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleSelectTeam = useCallback((club: string) => {
    setSelection({ kind: "team", club });
  }, []);

  const handleSelectLeague = useCallback(() => {
    setSelection({ kind: "league" });
  }, []);

  const handleSelectPlayer = useCallback((tmId: number) => {
    setSelection({ kind: "player", tmId });
  }, []);

  const handleSearchSelect = useCallback(
    (nodeId: string) => {
      setSearchFocusId(nodeId);
      const node = graph?.nodes.find((n) => n.id === nodeId);
      if (!node) return;
      if (node.type === "team") {
        handleSelectTeam(node.club);
      } else if (node.type === "league") {
        handleSelectLeague();
      } else if (node.tm_id != null) {
        handleSelectPlayer(node.tm_id);
      }
    },
    [graph, handleSelectTeam, handleSelectLeague, handleSelectPlayer],
  );

  const handleClear = useCallback(() => {
    setSelection(null);
    setSearchFocusId(null);
  }, []);

  if (loading) {
    return (
      <div className="app-shell loading">
        <p>Loading squad graph…</p>
      </div>
    );
  }

  if (error || !graph || !players) {
    return (
      <div className="app-shell loading">
        <p>Error: {error ?? "Missing data"}</p>
        <p className="hint">
          Run <code>python graph_viz/build_graph.py</code> from the repo root first.
        </p>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <Toolbar
        graph={graph}
        selection={selection}
        onSelectTeam={handleSelectTeam}
        onSelectLeague={handleSelectLeague}
        onClear={handleClear}
        onSearchSelect={handleSearchSelect}
      />
      <main className="app-main">
        <GraphCanvas
          graph={graph}
          selection={selection}
          hoverId={hoverId}
          onHover={setHoverId}
          onSelectLeague={handleSelectLeague}
          onSelectTeam={handleSelectTeam}
          onSelectPlayer={handleSelectPlayer}
          searchFocusId={searchFocusId}
        />
        <DetailPanel
          selection={selection}
          players={players}
          graph={graph}
          onClear={handleClear}
        />
      </main>
    </div>
  );
}

export default App;
