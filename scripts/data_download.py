from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
import random
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

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

# -----------------------------
# Configuration
# -----------------------------
MAX_WORKERS = 4              # Conservative
REQUESTS_PER_SECOND = 2      # Global limit
TIMEOUT = 120

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
    total=5,
    backoff_factor=2,
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
        return

    limiter.wait()

    # Small random delay to avoid synchronized requests
    time.sleep(random.uniform(0.1, 0.4))

    try:
        with session.get(url, stream=True, timeout=TIMEOUT) as r:

            if r.status_code in (404, 403):
                return

            r.raise_for_status()

            with open(outfile, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        print(f"Downloaded: {filename}")

    except Exception as e:
        print(f"Failed: {filename} ({e})")

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
    for year in range(2010, current_year + 1):
        max_month = current_month if year == current_year else 12
        for month in range(1, max_month + 1):
            urls.append(
                f"{BASE}/{dataset}_tripdata_{year}-{month:02d}.parquet"
            )

# -----------------------------
# Download
# -----------------------------
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    list(pool.map(download, urls))

print("Finished.")