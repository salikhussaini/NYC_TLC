from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
import random
from datetime import datetime
import sys
import logging
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Base URL for NYC TLC trip data
BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# Get python file path
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
# create parent dir
parent_dir = os.path.dirname(script_dir)

# create output directory
OUTPUT = Path(parent_dir) / "data"
OUTPUT.mkdir(exist_ok=True)

# create logs directory
LOGS_DIR = Path(parent_dir) / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Logging Setup
# -----------------------------
log_file = LOGS_DIR / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
MAX_WORKERS = 4              # Conservative
REQUESTS_PER_SECOND = 2      # Global limit
TIMEOUT = 180                 # Reduced from 120 to prevent long hangs
MAX_RETRIES = 3              # Reduced from 5 to fail faster

# -----------------------------
# Global Rate Limiter
# -----------------------------
class RateLimiter:
    def __init__(self, rate):
        self.interval = 1.0 / rate
        self.lock = threading.Lock()
        self.next_time = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()

            if now < self.next_time:
                time.sleep(self.next_time - now)

            self.next_time = max(now, self.next_time) + self.interval


limiter = RateLimiter(REQUESTS_PER_SECOND)

# -----------------------------
# HTTP Session
# -----------------------------
session = requests.Session()

retry = Retry(
    total=MAX_RETRIES,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,
)

adapter = HTTPAdapter(
    max_retries=retry,
    pool_connections=MAX_WORKERS,
    pool_maxsize=MAX_WORKERS,
)

session.mount("https://", adapter)
session.mount("http://", adapter)

# -----------------------------
# Download Function
# -----------------------------
def download(url):
    filename = url.rsplit("/", 1)[-1]
    outfile = OUTPUT / filename

    if outfile.exists():
        logger.info(f"Skipped (exists): {filename}")
        return

    logger.info(f"Attempting: {filename}")
    
    limiter.wait()

    # Small random delay to avoid synchronized requests
    time.sleep(random.uniform(0.1, 0.4))

    try:
        with session.get(url, stream=True, timeout=TIMEOUT) as r:

            if r.status_code in (404, 403):
                logger.warning(f"Skipped (not found): {filename}")
                return

            r.raise_for_status()

            with open(outfile, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        logger.info(f"Downloaded: {filename}")

    except Exception as e:
        logger.error(f"Failed: {filename} ({type(e).__name__}: {e})")

# -----------------------------
# Build URL List
# -----------------------------
urls = []

# Only download data up to the current month
now = datetime.now()
current_year = now.year
current_month = now.month

count = 0
for dataset in ["yellow", "green", "fhv", "fhvhv"]:
    for year in range(2000, current_year + 1):
        max_month = current_month if year == current_year else 12
        for month in range(1, max_month + 1):
            urls.append(
                f"{BASE}/{dataset}_tripdata_{year}-{month:02d}.parquet"
            )

# -----------------------------
# Download
# -----------------------------
logger.info(f"Starting download of {len(urls)} files...")

try:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        # Use map with timeout to prevent infinite hangs
        # Each file gets TIMEOUT + overhead, multiply by max possible retries
        results = pool.map(download, urls, timeout=TIMEOUT * (MAX_RETRIES + 2))
        # Consume the iterator to trigger execution
        list(results)
except TimeoutError as e:
    logger.error(f"Download timed out after {TIMEOUT * (MAX_RETRIES + 2)}s")
    logger.warning("Some files may not have completed.")
except KeyboardInterrupt:
    logger.warning("Download interrupted by user.")
except Exception as e:
    logger.error(f"{type(e).__name__}: {e}")

logger.info("Finished.")