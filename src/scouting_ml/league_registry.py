from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from scouting_ml.paths import PROCESSED_DIR


def slugify(value: str) -> str:
    """Return a filesystem-safe slug."""
    import re

    slug = value.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    slug = slug.replace("__", "_")
    return slug


def season_slug(season: str | None) -> str:
    if not season:
        return ""
    return season.replace("/", "-").replace(" ", "_")


@dataclass(frozen=True)
class LeagueConfig:
    slug: str
    name: str
    tm_league_url: str | None = None
    tm_season_label: str | None = None
    tm_season_id: int | None = None
    sofa_league_key: str | None = None
    sofa_tournament_id: int | None = None
    sofa_season_label: str | None = None
    processed_dataset: Path | None = None
    processed_dataset_pattern: str | None = None

    def guess_processed_dataset(self, season: str | None = None) -> Optional[Path]:
        """Return the most likely processed dataset for this league/season."""
        if self.processed_dataset:
            return self.processed_dataset
        if self.processed_dataset_pattern:
            ctx = {
                "slug": self.slug,
                "season": season or self.tm_season_label or "",
                "season_slug": season_slug(season or self.tm_season_label or ""),
            }
            formatted = self.processed_dataset_pattern.format(**ctx)
            return Path(formatted)
        if season:
            candidate = PROCESSED_DIR / f"{self.slug}_{season_slug(season)}_with_sofa.csv"
            if candidate.exists():
                return candidate
        if self.tm_season_label:
            candidate = PROCESSED_DIR / f"{self.slug}_{season_slug(self.tm_season_label)}_with_sofa.csv"
            if candidate.exists():
                return candidate
        return None

    def with_overrides(
        self,
        *,
        season: str | None = None,
        tm_season_id: int | None = None,
        sofa_season_label: str | None = None,
    ) -> "LeagueConfig":
        """Return a copy of the config with runtime overrides."""
        return LeagueConfig(
            slug=self.slug,
            name=self.name,
            tm_league_url=self.tm_league_url,
            tm_season_label=season or self.tm_season_label,
            tm_season_id=tm_season_id or self.tm_season_id,
            sofa_league_key=self.sofa_league_key,
            sofa_tournament_id=self.sofa_tournament_id,
            sofa_season_label=sofa_season_label or self.sofa_season_label,
            processed_dataset=self.processed_dataset,
            processed_dataset_pattern=self.processed_dataset_pattern,
        )


LEAGUES: Dict[str, LeagueConfig] = {
    "austrian_bundesliga": LeagueConfig(
        slug="austrian_bundesliga",
        name="Austrian Bundesliga",
        tm_league_url="https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/A1",
        tm_season_label="2025/26",
        tm_season_id=2025,
        sofa_league_key="Austrian Bundesliga",
        sofa_tournament_id=45,
        sofa_season_label="24/25",
        processed_dataset=PROCESSED_DIR / "Austrian Clubs" / "austrian_bundesliga_2025-26_with_sofa.csv",
        processed_dataset_pattern=str(
            PROCESSED_DIR / "Austrian Clubs" / "austrian_bundesliga_{season_slug}_with_sofa.csv"
        ),
    ),
    "estonian_meistriliiga": LeagueConfig(
        slug="estonian_meistriliiga",
        name="Estonian Meistriliiga",
        tm_league_url="https://www.transfermarkt.com/premium-liiga/startseite/wettbewerb/EST1",
        tm_season_label="2025",
        tm_season_id=2025,
        sofa_league_key="Premium Liiga",
        sofa_tournament_id=178,
        sofa_season_label="2025",
        processed_dataset=PROCESSED_DIR / "estonian_meistriliiga_2025_with_sofa.csv",
        processed_dataset_pattern=str(
            PROCESSED_DIR / "estonian_meistriliiga_{season_slug}_with_sofa.csv"
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
    """Return the slug if registry entry exists, otherwise build a slug from the value."""
    slug = slugify(value)
    return slug
