from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


FILE_PATTERN = "*_with_sofa.csv"
FILE_RE = re.compile(r"^(?P<league_slug>.+?)_(?P<season>\d{4}(?:-\d{2,4})?)_with_sofa\.csv$")

LEAGUE_COUNTRY_BY_SLUG = {
    "english_premier_league": "england",
    "spanish_la_liga": "spain",
    "german_bundesliga": "germany",
    "italian_serie_a": "italy",
    "french_ligue_1": "france",
    "portuguese_primeira_liga": "portugal",
    "dutch_eredivisie": "netherlands",
    "belgian_pro_league": "belgium",
    "turkish_super_lig": "turkey",
    "scottish_premiership": "scotland",
    "greek_super_league": "greece",
    "austrian_bundesliga": "austria",
    "estonian_meistriliiga": "estonia",
    "swiss_super_league": "switzerland",
    "danish_superliga": "denmark",
    "allsvenskan": "sweden",
    "norwegian_eliteserien": "norway",
    "finnish_veikkausliiga": "finland",
    "czech_fortuna_liga": "czechia",
    "polish_ekstraklasa": "poland",
    "croatian_hnl": "croatia",
    "serbian_superliga": "serbia",
}


@dataclass(frozen=True)
class CanonicalFile:
    basename: str
    source_path: Path
    league_slug: str
    season: str
    country: str
    duplicate_count: int
    duplicate_collision: bool


def _norm_season(token: str) -> str:
    token = token.strip()
    if "-" not in token:
        return token
    start, end = token.split("-", 1)
    if len(end) == 4:
        end = end[-2:]
    return f"{start}-{end}"


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _preferred_path(path: Path, source_dir: Path, combined_dir: Path) -> tuple[int, int, int]:
    rel = path.resolve().relative_to(source_dir.resolve())
    rel_lower = "/".join(rel.parts).lower()

    # Priority: canonical combined folder > root-level files > staging "_with_sofa" > everything else
    if _is_under(path, combined_dir):
        bucket = 0
    elif path.parent.resolve() == source_dir.resolve():
        bucket = 1
    elif "_with_sofa/" in f"{rel_lower}/":
        bucket = 2
    else:
        bucket = 3
    return (bucket, len(rel.parts), len(rel_lower))


def _parse_filename(name: str) -> tuple[str, str] | None:
    m = FILE_RE.match(name)
    if not m:
        return None
    return m.group("league_slug").strip().lower(), _norm_season(m.group("season"))


def _country_for_slug(league_slug: str) -> str:
    return LEAGUE_COUNTRY_BY_SLUG.get(league_slug, "unknown")


def _safe_slug(text: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return out or "unknown"


def _collect_candidates(source_dir: Path, season_root: Path, country_root: Path) -> list[Path]:
    files = sorted(source_dir.rglob(FILE_PATTERN))
    out: list[Path] = []
    for p in files:
        if _is_under(p, season_root) or _is_under(p, country_root):
            continue
        parsed = _parse_filename(p.name)
        if parsed is None:
            continue
        out.append(p)
    return out


def _canonicalize(
    source_dir: Path,
    combined_dir: Path,
    season_root: Path,
    country_root: Path,
) -> list[CanonicalFile]:
    files = _collect_candidates(source_dir=source_dir, season_root=season_root, country_root=country_root)
    if not files:
        return []

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[path.name].append(path)

    canonical: list[CanonicalFile] = []
    for basename, candidates in sorted(groups.items()):
        parsed = _parse_filename(basename)
        if parsed is None:
            continue
        league_slug, season = parsed

        chosen = min(candidates, key=lambda p: _preferred_path(p, source_dir=source_dir, combined_dir=combined_dir))
        collision = False
        if len(candidates) > 1:
            hashes = {_hash_file(p) for p in candidates}
            collision = len(hashes) > 1

        canonical.append(
            CanonicalFile(
                basename=basename,
                source_path=chosen,
                league_slug=league_slug,
                season=season,
                country=_country_for_slug(league_slug),
                duplicate_count=len(candidates),
                duplicate_collision=collision,
            )
        )
    return canonical


def _sync_file(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return "reused"
    shutil.copy2(src, dst)
    return "copied"


def _cleanup_stale(root: Path, keep: set[Path]) -> int:
    if not root.exists():
        return 0
    removed = 0
    for path in root.rglob(FILE_PATTERN):
        if path.resolve() not in keep:
            path.unlink()
            removed += 1
    return removed


def organize_processed_files(
    source_dir: str = "data/processed",
    combined_dir: str = "data/processed/Clubs combined",
    country_root: str = "data/processed/by_country",
    season_root: str = "data/processed/by_season",
    manifest_path: str = "data/processed/organization_manifest.csv",
    clean_targets: bool = True,
) -> None:
    src_root = Path(source_dir)
    out_combined = Path(combined_dir)
    out_country = Path(country_root)
    out_season = Path(season_root)
    out_manifest = Path(manifest_path)

    canonical = _canonicalize(
        source_dir=src_root,
        combined_dir=out_combined,
        season_root=out_season,
        country_root=out_country,
    )
    if not canonical:
        raise ValueError(f"No canonical files found under {src_root}")

    counts = Counter()
    keep_combined: set[Path] = set()
    keep_country: set[Path] = set()
    keep_season: set[Path] = set()

    rows: list[dict[str, str | int | bool]] = []
    country_counts = Counter()
    season_counts = Counter()
    collision_count = 0
    duplicate_groups = 0

    for item in canonical:
        duplicate_groups += 1 if item.duplicate_count > 1 else 0
        collision_count += 1 if item.duplicate_collision else 0

        combined_dst = out_combined / item.basename
        country_dst = out_country / _safe_slug(item.country) / item.basename
        season_dst = out_season / item.season / item.basename

        counts[f"combined_{_sync_file(item.source_path, combined_dst)}"] += 1
        counts[f"country_{_sync_file(item.source_path, country_dst)}"] += 1
        counts[f"season_{_sync_file(item.source_path, season_dst)}"] += 1

        keep_combined.add(combined_dst.resolve())
        keep_country.add(country_dst.resolve())
        keep_season.add(season_dst.resolve())

        country_counts[item.country] += 1
        season_counts[item.season] += 1

        rows.append(
            {
                "basename": item.basename,
                "league_slug": item.league_slug,
                "season": item.season,
                "country": item.country,
                "source_path": str(item.source_path),
                "combined_path": str(combined_dst),
                "country_path": str(country_dst),
                "season_path": str(season_dst),
                "duplicate_count": item.duplicate_count,
                "duplicate_collision": item.duplicate_collision,
            }
        )

    removed_combined = _cleanup_stale(out_combined, keep_combined) if clean_targets else 0
    removed_country = _cleanup_stale(out_country, keep_country) if clean_targets else 0
    removed_season = _cleanup_stale(out_season, keep_season) if clean_targets else 0

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "basename",
        "league_slug",
        "season",
        "country",
        "source_path",
        "combined_path",
        "country_path",
        "season_path",
        "duplicate_count",
        "duplicate_collision",
    ]
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[organize] source dir: {src_root}")
    print(f"[organize] canonical league-season files: {len(canonical)}")
    print(f"[organize] duplicate groups: {duplicate_groups}")
    print(f"[organize] duplicate collision groups: {collision_count}")
    print(
        "[organize] writes: "
        f"combined copied={counts['combined_copied']} reused={counts['combined_reused']} | "
        f"country copied={counts['country_copied']} reused={counts['country_reused']} | "
        f"season copied={counts['season_copied']} reused={counts['season_reused']}"
    )
    if clean_targets:
        print(
            "[organize] stale removed: "
            f"combined={removed_combined} | country={removed_country} | season={removed_season}"
        )

    print(f"[organize] country folders: {len(country_counts)}")
    print(
        "[organize] by-country counts: "
        + ", ".join(f"{k}:{v}" for k, v in sorted(country_counts.items()))
    )
    print(
        "[organize] by-season counts: "
        + ", ".join(f"{k}:{v}" for k, v in sorted(season_counts.items()))
    )
    print(f"[organize] manifest written -> {out_manifest}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize processed league CSVs into canonical folders: "
            "Clubs combined, by_country, by_season."
        )
    )
    parser.add_argument("--source-dir", default="data/processed", help="Root directory containing processed CSVs.")
    parser.add_argument("--combined-dir", default="data/processed/Clubs combined", help="Canonical combined folder.")
    parser.add_argument("--country-root", default="data/processed/by_country", help="Output root for per-country folders.")
    parser.add_argument("--season-root", default="data/processed/by_season", help="Output root for per-season folders.")
    parser.add_argument(
        "--manifest",
        default="data/processed/organization_manifest.csv",
        help="Output CSV manifest with source and canonical paths.",
    )
    parser.add_argument(
        "--no-clean-targets",
        action="store_true",
        help="Do not remove stale *_with_sofa.csv files from target folders.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    organize_processed_files(
        source_dir=args.source_dir,
        combined_dir=args.combined_dir,
        country_root=args.country_root,
        season_root=args.season_root,
        manifest_path=args.manifest,
        clean_targets=not args.no_clean_targets,
    )
