import type { GraphData, PlayerDetail, PlayersMap, Selection } from "../types";
import { formatDateRange, formatMarketValue } from "../utils/format";
import { getTeamColor } from "../utils/teamColors";

interface DetailPanelProps {
  selection: Selection;
  players: PlayersMap;
  graph: GraphData;
  onClear: () => void;
}

function LeagueSummary({ graph, players }: { graph: GraphData; players: PlayersMap }) {
  const teams = graph.nodes.filter((n) => n.type === "team");
  const playerNodes = graph.nodes.filter((n) => n.type === "player");
  const details = playerNodes
    .map((n) => (n.tm_id != null ? players[String(n.tm_id)] : null))
    .filter(Boolean) as PlayerDetail[];

  const totalValue = details.reduce((s, p) => s + (p.market_value_eur ?? 0), 0);
  const ages = details.map((p) => p.age).filter((a): a is number => a != null);
  const avgAge = ages.length ? ages.reduce((a, b) => a + b, 0) / ages.length : null;

  return (
    <div className="detail-content">
      <div className="team-badge league-badge">Super League Greece</div>
      <dl className="detail-grid">
        <div>
          <dt>Teams</dt>
          <dd>{teams.length}</dd>
        </div>
        <div>
          <dt>Players</dt>
          <dd>{playerNodes.length}</dd>
        </div>
        <div>
          <dt>Total market value</dt>
          <dd>{formatMarketValue(totalValue)}</dd>
        </div>
        <div>
          <dt>Average age</dt>
          <dd>{avgAge != null ? avgAge.toFixed(1) : "—"}</dd>
        </div>
      </dl>
      <p className="detail-hint">
        Click a team cluster to zoom and inspect its squad, or use Focus team above.
      </p>
    </div>
  );
}

function TeamSummary({
  club,
  graph,
  players,
}: {
  club: string;
  graph: GraphData;
  players: PlayersMap;
}) {
  const squad = graph.nodes.filter((n) => n.type === "player" && n.club === club);
  const details = squad
    .map((n) => (n.tm_id != null ? players[String(n.tm_id)] : null))
    .filter(Boolean) as PlayerDetail[];

  const totalValue = details.reduce((s, p) => s + (p.market_value_eur ?? 0), 0);
  const ages = details.map((p) => p.age).filter((a): a is number => a != null);
  const avgAge = ages.length ? ages.reduce((a, b) => a + b, 0) / ages.length : null;
  const injuryCount = details.reduce((s, p) => s + p.injuries.length, 0);

  return (
    <div className="detail-content">
      <div
        className="team-badge"
        style={{ borderColor: getTeamColor(club), color: getTeamColor(club) }}
      >
        {club}
      </div>
      <dl className="detail-grid">
        <div>
          <dt>Squad size</dt>
          <dd>{squad.length}</dd>
        </div>
        <div>
          <dt>Total market value</dt>
          <dd>{formatMarketValue(totalValue)}</dd>
        </div>
        <div>
          <dt>Average age</dt>
          <dd>{avgAge != null ? avgAge.toFixed(1) : "—"}</dd>
        </div>
        <div>
          <dt>Injury records</dt>
          <dd>{injuryCount}</dd>
        </div>
      </dl>
      <p className="detail-hint">Click a player node to view their profile.</p>
    </div>
  );
}

function PlayerProfile({ player }: { player: PlayerDetail }) {
  return (
    <div className="detail-content">
      <div className="player-header">
        <h2>{player.full_name}</h2>
        {player.shirt_number != null && (
          <span className="shirt-number">#{player.shirt_number}</span>
        )}
      </div>
      <div
        className="team-badge small"
        style={{
          borderColor: getTeamColor(player.club),
          color: getTeamColor(player.club),
        }}
      >
        {player.club}
      </div>
      <dl className="detail-grid">
        <div>
          <dt>Age</dt>
          <dd>{player.age ?? "—"}</dd>
        </div>
        <div>
          <dt>Nationality</dt>
          <dd>{player.nationality ?? "—"}</dd>
        </div>
        <div>
          <dt>Position</dt>
          <dd>
            {player.primary_position ?? "—"}
            {player.secondary_position ? ` / ${player.secondary_position}` : ""}
          </dd>
        </div>
        <div>
          <dt>Market value</dt>
          <dd>{formatMarketValue(player.market_value_eur)}</dd>
        </div>
        <div>
          <dt>Date of birth</dt>
          <dd>{player.dob ?? "—"}</dd>
        </div>
        <div>
          <dt>Transfermarkt ID</dt>
          <dd>{player.tm_id}</dd>
        </div>
      </dl>
      <section className="injuries-section">
        <h3>Injury history ({player.injuries.length})</h3>
        {player.injuries.length === 0 ? (
          <p className="empty-note">No recorded injuries.</p>
        ) : (
          <ul className="injury-list">
            {player.injuries.slice(0, 12).map((inj, i) => (
              <li key={`${inj.date_from}-${i}`}>
                <span className="injury-type">{inj.injury_type}</span>
                <span className="injury-dates">
                  {formatDateRange(inj.date_from, inj.date_to)}
                </span>
                {inj.matches_missed != null && inj.matches_missed > 0 && (
                  <span className="injury-missed">{inj.matches_missed} missed</span>
                )}
              </li>
            ))}
          </ul>
        )}
        {player.injuries.length > 12 && (
          <p className="empty-note">Showing 12 most recent of {player.injuries.length}.</p>
        )}
      </section>
    </div>
  );
}

export function DetailPanel({ selection, players, graph, onClear }: DetailPanelProps) {
  return (
    <aside className="detail-panel">
      <div className="detail-panel-header">
        <h1>Details</h1>
        {selection && (
          <button type="button" className="btn-clear" onClick={onClear}>
            Clear
          </button>
        )}
      </div>
      {!selection && (
        <div className="detail-empty">
          <p>Select the league hub, a team, or a player on the graph to inspect data.</p>
          <ul>
            <li>
              {graph.meta.teams} teams · {graph.meta.players} players
            </li>
            <li>Click a team to zoom in — the full graph stays visible</li>
            <li>Node size reflects market value</li>
          </ul>
        </div>
      )}
      {selection?.kind === "league" && (
        <LeagueSummary graph={graph} players={players} />
      )}
      {selection?.kind === "team" && (
        <TeamSummary club={selection.club} graph={graph} players={players} />
      )}
      {selection?.kind === "player" && players[String(selection.tmId)] && (
        <PlayerProfile player={players[String(selection.tmId)]} />
      )}
      {selection?.kind === "player" && !players[String(selection.tmId)] && (
        <p className="empty-note">Player data not found.</p>
      )}
    </aside>
  );
}
