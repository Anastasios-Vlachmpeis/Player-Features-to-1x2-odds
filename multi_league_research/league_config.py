"""Configuration shared by the six-league dataset builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATSAPI_ROOT = PROJECT_ROOT / "data" / "statsapi"

DEVELOPMENT_SEASONS = (
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
)
FINAL_SEASON = "2025-26"
ALL_RESEARCH_SEASONS = (*DEVELOPMENT_SEASONS, FINAL_SEASON)


@dataclass(frozen=True)
class LeagueConfig:
    key: str
    name: str
    division: str
    team_aliases: dict[str, str] = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return STATSAPI_ROOT / self.key

    @property
    def matches_csv(self) -> Path:
        return self.data_dir / "matches.csv"

    @property
    def player_stats_csv(self) -> Path:
        return self.data_dir / "player_match_stats.csv"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "player_stats_raw"


LEAGUES: dict[str, LeagueConfig] = {
    "scotland": LeagueConfig(
        key="scotland",
        name="Scotland",
        division="SC0",
        team_aliases={
            "hamiltonacademical": "hamilton",
            "heartofmidlothian": "hearts",
        },
    ),
    "greece": LeagueConfig(
        key="greece",
        name="Greece",
        division="G1",
        team_aliases={
            "aekathens": "aek",
            "aekifisia": "kifisia",
            "aelarisa": "larisa",
            "apolevadiakos": "levadeiakos",
            "aristhessaloniki": "aris",
            "asterasaktor": "asterastripolis",
            "atromitosathens": "atromitos",
            "gsapollonsmyrnis": "apollon",
            "mgspanserraikos": "panserraikos",
            "npsvolos": "volosnfc",
            "olympiacos": "olympiakos",
            "pasgiannina": "giannina",
            "paslamia1964": "lamia",
        },
    ),
    "belgium": LeagueConfig(
        key="belgium",
        name="Belgium",
        division="B1",
        team_aliases={
            "clubbruggekv": "clubbrugge",
            "fcvdender": "dender",
            "kbeerschotva": "beerschotva",
            "kaagent": "gent",
            "kaseupen": "eupen",
            "krcgenk": "genk",
            "kvkortrijk": "kortrijk",
            "kvmechelen": "mechelen",
            "kvoostende": "oostende",
            "kvcwesterlo": "westerlo",
            "rcsportingcharleroi": "charleroi",
            "rfcseraing": "seraing",
            "royalantwerp": "antwerp",
            "royaleunionsaintgilloise": "stgilloise",
            "royalexcelmouscron": "mouscron",
            "rscanderlecht": "anderlecht",
            "sinttruidensevv": "sttruiden",
            "skbeveren": "waaslandbeveren",
            "standardliege": "standard",
            "svzultewaregem": "waregem",
        },
    ),
    "portugal": LeagueConfig(
        key="portugal",
        name="Portugal",
        division="P1",
        team_aliases={
            "avsfutebolsad": "avs",
            "bsad": "belenenses",
            "cdnacional": "nacional",
            "cfestrelaamadora": "estrela",
            "estorilpraia": "estoril",
            "fcarouca": "arouca",
            "fcporto": "porto",
            "scfarense": "farense",
            "sporting": "splisbon",
            "sportingbraga": "spbraga",
            "vitoriasc": "guimaraes",
        },
    ),
    "netherlands": LeagueConfig(
        key="netherlands",
        name="Netherlands",
        division="N1",
        team_aliases={
            "adodenhaag": "denhaag",
            "afcajax": "ajax",
            "fcgroningen": "groningen",
            "fctwente": "twente",
            "fcutrecht": "utrecht",
            "fcvolendam": "volendam",
            "fortunasittard": "forsittard",
            "heraclesalmelo": "heracles",
            "necnijmegen": "nijmegen",
            "peczwolle": "zwolle",
            "rkcwaalwijk": "waalwijk",
            "sccambuur": "cambuur",
            "scheerenveen": "heerenveen",
            "willemiitilburg": "willemii",
        },
    ),
    "turkey": LeagueConfig(
        key="turkey",
        name="Turkey",
        division="T1",
        team_aliases={
            "adanademirspor": "addemirspor",
            "basaksehir": "buyuksehyr",
            "bodrumfk": "bodrumspor",
            "caykurrizespor": "rizespor",
            "erzurumsporfk": "erzurumbb",
            "fatihkaragumruk": "karagumruk",
            "gaziantepfk": "gaziantep",
            "goztepe": "goztep",
            "mkeankaragucu": "ankaragucu",
        },
    ),
}


def selected_leagues(values: list[str] | None = None) -> list[LeagueConfig]:
    keys = list(dict.fromkeys(values)) if values else list(LEAGUES)
    return [LEAGUES[key] for key in keys]
