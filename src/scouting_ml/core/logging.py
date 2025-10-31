from pathlib import Path
from loguru import logger
import datetime

# Define log folder
BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure rotating logs (1 per day)
date_str = datetime.date.today().isoformat()
log_path = LOG_DIR / f"tm_scraper_{date_str}.log"

logger.remove()  # remove default
logger.add(
    log_path,
    rotation="1 day",
    retention="14 days",
    compression="zip",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
)

def get_logger():
    return logger
