export type NodeType = "league" | "team" | "player";

export interface GraphNode {
  id: string;
  type: NodeType;
  label: string;
  club: string;
  val: number;
  position?: string;
  tm_id?: number;
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
}

export interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  type: string;
}

export interface GraphData {
  meta: { teams: number; players: number; links: number };
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface Injury {
  injury_type: string;
  date_from: string;
  date_to: string | null;
  matches_missed: number | null;
}

export interface PlayerDetail {
  tm_id: number;
  full_name: string;
  dob: string | null;
  age: number | null;
  nationality: string | null;
  primary_position: string | null;
  secondary_position: string | null;
  market_value_eur: number | null;
  club: string;
  shirt_number: number | null;
  injuries: Injury[];
}

export type PlayersMap = Record<string, PlayerDetail>;

export type Selection =
  | { kind: "league" }
  | { kind: "team"; club: string }
  | { kind: "player"; tmId: number }
  | null;

export const LEAGUE_NODE_ID = "league:super-league";
