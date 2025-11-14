from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from scouting_ml.paths import PROCESSED_DIR


def slugify(value: str) -> str:
    import re

    slug = value.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    return slug.replace("__", "_")


def season_slug_label(season: str) -> str:
    return season.replace("/", "-").replace(" ", "_")


def season_slug(season: str) -> str:
    return season_slug_label(season)


@dataclass(frozen=True)
class LeagueConfig:
    slug: str
    name: str
    tm_league_url: str
    seasons: List[str]
    tm_season_ids: Dict[str, int]
    sofa_league_key: str
    sofa_tournament_id: int
    sofa_season_map: Dict[str, str]
    processed_dataset_pattern: str

    def guess_processed_dataset(self, season: str) -> Path:
        season_slug = season_slug_label(season)
        return Path(self.processed_dataset_pattern.format(season_slug=season_slug))

    @property
    def default_season(self) -> str:
        return self.seasons[0]

    @property
    def tm_season_label(self) -> str:
        return self.default_season

    @property
    def tm_season_id(self) -> int:
        return self.tm_season_ids[self.default_season]

    @property
    def sofa_season_label(self) -> str:
        return self.sofa_season_map[self.default_season]


LEAGUES: Dict[str, LeagueConfig] = {
    # --- Big Five -----------------------------------------------------------
    "english_premier_league": LeagueConfig(
        slug="english_premier_league",
        name="English Premier League",
        tm_league_url="https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1",
        seasons=["2024/25", "2023/24", "2022/23", "2021/22", "2020/21", "2019/20"],
        tm_season_ids={
            "2024/25": 2024,
            "2023/24": 2023,
            "2022/23": 2022,
            "2021/22": 2021,
            "2020/21": 2020,
            "2019/20": 2019,
        },
        sofa_league_key="Premier League",
        sofa_tournament_id=17,
        sofa_season_map={
            "2024/25": "24/25",
            "2023/24": "23/24",
            "2022/23": "22/23",
            "2021/22": "21/22",
            "2020/21": "20/21",
            "2019/20": "19/20",
        },
        processed_dataset_pattern=str(
            PROCESSED_DIR / "english_premier_league_{season_slug}_with_sofa.csv"
        ),
    ),
    "spanish_la_liga": LeagueConfig(
        slug="spanish_la_liga",
        name="LaLiga",
        tm_league_url="https://www.transfermarkt.com/primera-division/startseite/wettbewerb/ES1",
        seasons=["2024/25", "2023/24", "2022/23", "2021/22", "2020/21", "2019/20"],
        tm_season_ids={
            "2024/25": 2024,
            "2023/24": 2023,
            "2022/23": 2022,
            "2021/22": 2021,
            "2020/21": 2020,
            "2019/20": 2019,
        },
        sofa_league_key="LaLiga",
        sofa_tournament_id=8,
        sofa_season_map={
            "2024/25": "24/25",
            "2023/24": "23/24",
            "2022/23": "22/23",
            "2021/22": "21/22",
            "2020/21": "20/21",
            "2019/20": "19/20",
        },
        processed_dataset_pattern=str(
            PROCESSED_DIR / "spanish_la_liga_{season_slug}_with_sofa.csv"
        ),
    ),
    "german_bundesliga": LeagueConfig(
        slug="german_bundesliga",
        name="Bundesliga",
        tm_league_url="https://www.transfermarkt.de/bundesliga/startseite/wettbewerb/L1",
        seasons=["2024/25", "2023/24", "2022/23", "2021/22", "2020/21", "2019/20"],
        tm_season_ids={
            "2024/25": 2024,
            "2023/24": 2023,
            "2022/23": 2022,
            "2021/22": 2021,
            "2020/21": 2020,
            "2019/20": 2019,
        },
        sofa_league_key="Bundesliga",
        sofa_tournament_id=35,
        sofa_season_map={
            "2024/25": "24/25",
            "2023/24": "23/24",
            "2022/23": "22/23",
            "2021/22": "21/22",
            "2020/21": "20/21",
            "2019/20": "19/20",
        },
        processed_dataset_pattern=str(
            PROCESSED_DIR / "german_bundesliga_{season_slug}_with_sofa.csv"
        ),
    ),
    "italian_serie_a": LeagueConfig(
        slug="italian_serie_a",
        name="Serie A",
        tm_league_url="https://www.transfermarkt.com/serie-a/startseite/wettbewerb/IT1",
        seasons=["2024/25", "2023/24", "2022/23", "2021/22", "2020/21", "2019/20"],
        tm_season_ids={
            "2024/25": 2024,
            "2023/24": 2023,
            "2022/23": 2022,
            "2021/22": 2021,
            "2020/21": 2020,
            "2019/20": 2019,
        },
        sofa_league_key="Serie A",
        sofa_tournament_id=23,
        sofa_season_map={
            "2024/25": "24/25",
            "2023/24": "23/24",
            "2022/23": "22/23",
            "2021/22": "21/22",
            "2020/21": "20/21",
            "2019/20": "19/20",
        },
        processed_dataset_pattern=str(
            PROCESSED_DIR / "italian_serie_a_{season_slug}_with_sofa.csv"
        ),
    ),
    "french_ligue_1": LeagueConfig(
        slug="french_ligue_1",
        name="Ligue 1",
        tm_league_url="https://www.transfermarkt.com/ligue-1/startseite/wettbewerb/FR1",
        seasons=["2024/25", "2023/24", "2022/23", "2021/22", "2020/21", "2019/20"],
        tm_season_ids={
            "2024/25": 2024,
            "2023/24": 2023,
            "2022/23": 2022,
            "2021/22": 2021,
            "2020/21": 2020,
            "2019/20": 2019,
        },
        sofa_league_key="Ligue 1",
        sofa_tournament_id=34,
        sofa_season_map={
            "2024/25": "24/25",
            "2023/24": "23/24",
            "2022/23": "22/23",
            "2021/22": "21/22",
            "2020/21": "20/21",
            "2019/20": "19/20",
        },
        processed_dataset_pattern=str(
            PROCESSED_DIR / "french_ligue_1_{season_slug}_with_sofa.csv"
        ),
    ),

    # --- Austria & Switzerland --------------------------------------------
    "austrian_bundesliga": LeagueConfig(
        slug="austrian_bundesliga",
        name="Austrian Bundesliga",
        tm_league_url="https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/A1",
        seasons=["2024/25", "2023/24"],
        tm_season_ids={"2024/25": 2024, "2023/24": 2023},
        sofa_league_key="Austrian Bundesliga",
        sofa_tournament_id=45,
        sofa_season_map={"2024/25": "24/25", "2023/24": "23/24"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "austrian_bundesliga_{season_slug}_with_sofa.csv"
        ),
    ),
    "swiss_super_league": LeagueConfig(
        slug="swiss_super_league",
        name="Swiss Super League",
        tm_league_url="https://www.transfermarkt.com/super-league/startseite/wettbewerb/C1",
        seasons=["2024/25", "2023/24"],
        tm_season_ids={"2024/25": 2024, "2023/24": 2023},
        sofa_league_key="Super League",
        sofa_tournament_id=55,
        sofa_season_map={"2024/25": "24/25", "2023/24": "23/24"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "swiss_super_league_{season_slug}_with_sofa.csv"
        ),
    ),

    # --- Scandinavia -------------------------------------------------------
    "danish_superliga": LeagueConfig(
        slug="danish_superliga",
        name="Danish Superliga",
        tm_league_url="https://www.transfermarkt.com/superligaen/startseite/wettbewerb/DK1",
        seasons=["2024/25", "2023/24"],
        tm_season_ids={"2024/25": 2024, "2023/24": 2023},
        sofa_league_key="Superliga",
        sofa_tournament_id=31,
        sofa_season_map={"2024/25": "2024/25", "2023/24": "2023/24"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "danish_superliga_{season_slug}_with_sofa.csv"
        ),
    ),
    "swedish_allsvenskan": LeagueConfig(
        slug="swedish_allsvenskan",
        name="Allsvenskan",
        tm_league_url="https://www.transfermarkt.com/allsvenskan/startseite/wettbewerb/SE1",
        seasons=["2024", "2023"],
        tm_season_ids={"2024": 2024, "2023": 2023},
        sofa_league_key="Allsvenskan",
        sofa_tournament_id=120,
        sofa_season_map={"2024": "2024", "2023": "2023"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "swedish_allsvenskan_{season_slug}_with_sofa.csv"
        ),
    ),
    "norwegian_eliteserien": LeagueConfig(
        slug="norwegian_eliteserien",
        name="Norwegian Eliteserien",
        tm_league_url="https://www.transfermarkt.com/eliteserien/startseite/wettbewerb/NO1",
        seasons=["2024", "2023"],
        tm_season_ids={"2024": 2024, "2023": 2023},
        sofa_league_key="Eliteserien",
        sofa_tournament_id=36,
        sofa_season_map={"2024": "2024", "2023": "2023"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "norwegian_eliteserien_{season_slug}_with_sofa.csv"
        ),
    ),
    "finnish_veikkausliiga": LeagueConfig(
        slug="finnish_veikkausliiga",
        name="Finnish Veikkausliiga",
        tm_league_url="https://www.transfermarkt.com/veikkausliiga/startseite/wettbewerb/FI1",
        seasons=["2024", "2023"],
        tm_season_ids={"2024": 2024, "2023": 2023},
        sofa_league_key="Veikkausliiga",
        sofa_tournament_id=124,
        sofa_season_map={"2024": "2024", "2023": "2023"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "finnish_veikkausliiga_{season_slug}_with_sofa.csv"
        ),
    ),

    # --- Eastern Europe ----------------------------------------------------
    "czech_first_league": LeagueConfig(
        slug="czech_first_league",
        name="Czech Fortuna Liga",
        tm_league_url="https://www.transfermarkt.com/fortuna-liga/startseite/wettbewerb/TS1",
        seasons=["2024/25"],
        tm_season_ids={"2024/25": 2024},
        sofa_league_key="Fortuna Liga",
        sofa_tournament_id=238,
        sofa_season_map={"2024/25": "24/25"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "czech_first_league_{season_slug}_with_sofa.csv"
        ),
    ),
    "polish_ekstraklasa": LeagueConfig(
        slug="polish_ekstraklasa",
        name="Polish Ekstraklasa",
        tm_league_url="https://www.transfermarkt.com/ekstraklasa/startseite/wettbewerb/PL1",
        seasons=["2024/25"],
        tm_season_ids={"2024/25": 2024},
        sofa_league_key="Ekstraklasa",
        sofa_tournament_id=59,
        sofa_season_map={"2024/25": "24/25"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "polish_ekstraklasa_{season_slug}_with_sofa.csv"
        ),
    ),
    "croatian_hnl": LeagueConfig(
        slug="croatian_hnl",
        name="Croatian HNL",
        tm_league_url="https://www.transfermarkt.com/1-hnl/startseite/wettbewerb/KR1",
        seasons=["2024/25"],
        tm_season_ids={"2024/25": 2024},
        sofa_league_key="HNL",
        sofa_tournament_id=117,
        sofa_season_map={"2024/25": "24/25"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "croatian_hnl_{season_slug}_with_sofa.csv"
        ),
    ),
    "serbian_super_liga": LeagueConfig(
        slug="serbian_super_liga",
        name="Serbian SuperLiga",
        tm_league_url="https://www.transfermarkt.com/super-liga/startseite/wettbewerb/RS1",
        seasons=["2024/25"],
        tm_season_ids={"2024/25": 2024},
        sofa_league_key="SuperLiga",
        sofa_tournament_id=205,
        sofa_season_map={"2024/25": "24/25"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "serbian_super_liga_{season_slug}_with_sofa.csv"
        ),
    ),
    "estonian_meistriliiga": LeagueConfig(
        slug="estonian_meistriliiga",
        name="Estonian Meistriliiga",
        tm_league_url="https://www.transfermarkt.com/premium-liiga/startseite/wettbewerb/EST1",
        seasons=["2025", "2024"],
        tm_season_ids={"2025": 2025, "2024": 2024},
        sofa_league_key="Premium Liiga",
        sofa_tournament_id=178,
        sofa_season_map={"2025": "2025", "2024": "2024"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "estonian_meistriliiga_{season_slug}_with_sofa.csv"
        ),
    ),

    # --- South America -----------------------------------------------------
    "brazil_serie_a": LeagueConfig(
        slug="brazil_serie_a",
        name="Brazil Série A",
        tm_league_url="https://www.transfermarkt.com/serie-a/startseite/wettbewerb/BRA1",
        seasons=["2024", "2023"],
        tm_season_ids={"2024": 2024, "2023": 2023},
        sofa_league_key="Serie A",
        sofa_tournament_id=325,
        sofa_season_map={"2024": "2024", "2023": "2023"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "brazil_serie_a_{season_slug}_with_sofa.csv"
        ),
    ),
    "brazil_serie_b": LeagueConfig(
        slug="brazil_serie_b",
        name="Brazil Série B",
        tm_league_url="https://www.transfermarkt.com/serie-b/startseite/wettbewerb/BRA2",
        seasons=["2024", "2023"],
        tm_season_ids={"2024": 2024, "2023": 2023},
        sofa_league_key="Serie B",
        sofa_tournament_id=327,
        sofa_season_map={"2024": "2024", "2023": "2023"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "brazil_serie_b_{season_slug}_with_sofa.csv"
        ),
    ),
    "argentina_primera_division": LeagueConfig(
        slug="argentina_primera_division",
        name="Argentina Primera División",
        tm_league_url="https://www.transfermarkt.com/liga-profesional-de-futbol/startseite/wettbewerb/AR1",
        seasons=["2024", "2023"],
        tm_season_ids={"2024": 2024, "2023": 2023},
        sofa_league_key="Liga Profesional",
        sofa_tournament_id=116,
        sofa_season_map={"2024": "2024", "2023": "2023"},
        processed_dataset_pattern=str(
            PROCESSED_DIR / "argentina_primera_division_{season_slug}_with_sofa.csv"
        ),
    ),
}


def list_leagues() -> List[LeagueConfig]:
    return sorted(LEAGUES.values(), key=lambda league: league.name.lower())


def get_league(slug: str) -> LeagueConfig:
    try:
        return LEAGUES[slug]
    except KeyError as exc:
        raise KeyError(
            f"Unknown league slug '{slug}'. Available options: "
            + ", ".join(sorted(LEAGUES))
        ) from exc


def ensure_slug(value: str) -> str:
    return slugify(value)
