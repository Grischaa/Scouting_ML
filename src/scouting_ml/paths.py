from pathlib import Path

# This file lives at <repo>/src/scouting_ml/paths.py
# Repo root is 2 levels up (paths.py -> scouting_ml -> src -> <repo>)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
LOGS_DIR = PROJECT_ROOT / "logs"
TMP_DIR = PROJECT_ROOT / "tmp"

# Sources
TM_RAW = RAW_DIR / "tm"
SOFA_RAW = RAW_DIR / "sofascore"


def ensure_dirs() -> None:
    """Create all required project directories if they don't exist."""
    for p in [
        DATA_DIR,
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        EXTERNAL_DIR,
        LOGS_DIR,
        TMP_DIR,
        TM_RAW,
        SOFA_RAW,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def tm_html(filename: str) -> Path:
    """Where to save a Transfermarkt HTML page."""
    return TM_RAW / filename


def sofa_json(filename: str) -> Path:
    """Where to save a Sofascore JSON dump."""
    return SOFA_RAW / filename


if __name__ == "__main__":
    ensure_dirs()
    print("Directories ready at:", DATA_DIR)
