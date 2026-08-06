# NYC TLC Trip Record Data Repository

This repository hosts datasets from the New York City Taxi and Limousine Commission (TLC), capturing trip records for yellow taxis, green taxis, and For-Hire Vehicles (FHV). The datasets encompass various fields such as pick-up and drop-off dates/times, locations, trip distances, fares, rate types, payment methods, and more.

## Data Sources

### Yellow Taxi Trips
Yellow taxis have been serving New York City since 2009 and are hailed via street signals or e-hail apps like Curb or Arro. They are permitted to respond to street hails across all five boroughs.

- **Data Fields**: pick-up and drop-off dates/times, locations, trip distances, itemized fares, rate types, payment types, and passenger counts.
- **Data Dictionary**: [PDF Download](#https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)

### Green Taxi Trips
Introduced in August 2013, green taxis serve the boroughs of NYC and specific areas in Manhattan.

- **Data Fields**: pick-up and drop-off dates/times, locations, trip distances, itemized fares, rate types, payment types, and passenger counts.
- **Data Dictionary**: [PDF Download](#https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf)

### For-Hire Vehicle (FHV) Data
FHV data includes trip records from high-volume for-hire vehicle bases, community livery bases, luxury limousine bases, and black car bases.

- **Data Fields**: dispatching base number, pick-up/drop-off dates/times, locations, trip distances, fare details, and more.
- **Data Dictionary**: [PDF Download](#)
- **FHV Data Dictionary**: [PDF Download](#https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_fhv.pdf)
- **High Volume Data Dictionary**: [PDF Download](#https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf)

#### Note on Shared Rides
Shared rides information, like Lyft Line and Uber Pool, is included if the trip was specially reserved with one of these services.

#### Matching to FHV Bases
To identify the base that dispatched the trip, join the `dispatching_base_num` with the `License Number`. Note that high-volume bases might not match the commonly recognized company name.

- **Key for HVFHS Companies**: Juno, Lyft, Uber, and Via

**Disclaimer**: The TLC does not create the trip data and makes no representations regarding its accuracy.

## Data Engineering Pipeline

The `scripts/data_engineer.py` script processes NYC TLC trip data (CSV or Parquet) and engineers 20+ new features for analysis and modeling.

### Features Engineered

**Temporal Features**:
- `pickup_hour`, `pickup_day_of_week`, `pickup_date`
- `is_weekend`, `is_peak_hour`

**Trip Duration & Speed**:
- `trip_duration_minutes`, `trip_duration_seconds`
- `trip_speed_mph`, `distance_category` (Short/Medium/Long/Very Long)

**Financial Metrics**:
- `cost_per_mile`, `cost_per_minute`, `revenue_per_mile`
- `tip_percentage`, `payment_type_name`, `is_cash_payment`
- `total_surcharges`, `surcharge_percentage` (yellow/green only)

**Location Features**:
- `is_airport_trip` (pickup/dropoff at airport locations)

**Taxi-Specific Features**:
- **Yellow**: `per_passenger_fare`, `per_passenger_cost`
- **Green**: `trip_type_name`, `per_passenger_fare`, `per_passenger_cost`
- **FHVHV**: `request_to_pickup_minutes`, `driver_earnings_per_mile`, `platform_commission`, `license_type`, accessibility flags
- **FHV**: Basic temporal and location features

**Data Quality Flags**:
- `zero_fare_flag`, `zero_distance_flag`, `negative_duration_flag`, `excessive_speed_flag`

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Basic Usage
```bash
python scripts/data_engineer.py
```
Processes all files in `data/` folder, outputs CSV files to `data/engineered/`.

#### Output Format
```bash
# Output as Parquet (10-100x faster for large files)
python scripts/data_engineer.py --format parquet

# Output as CSV with gzip compression
python scripts/data_engineer.py --format csv --compress gzip
```

#### Parallel Processing
```bash
# Auto-detect CPU count (e.g., uses 8 workers on 8-core CPU)
python scripts/data_engineer.py --format parquet

# Use specific number of workers
python scripts/data_engineer.py --format parquet --workers 4

# Combine all options
python scripts/data_engineer.py --format parquet --compress snappy --workers 8 --rerun
```

#### Force Reprocessing
```bash
# Reprocess all files even if engineered versions already exist
python scripts/data_engineer.py --rerun
```

### Running in Background (Linux/macOS)

#### Option 1: Using `nohup` (Simplest)
```bash
nohup python scripts/data_engineer.py --format parquet --workers 8 > output.log 2>&1 &
```
- Continues running after terminal closes
- Output saved to `output.log`
- Check progress: `tail -f output.log`
- Kill process: `pkill -f data_engineer.py`

#### Option 2: Using `screen` (Interactive monitoring)
```bash
# Create a new screen session
screen -S tlc_engineering

# Run the script
python scripts/data_engineer.py --format parquet --workers 8

# Detach: Ctrl+A then D
# Reattach: screen -r tlc_engineering
# List sessions: screen -ls
```

#### Option 3: Using `tmux` (Modern terminal multiplexer)
```bash
# Create a new tmux session
tmux new-session -d -s tlc_engineering

# Run the script in the session
tmux send-keys -t tlc_engineering "cd /path/to/NYC_TLC && python scripts/data_engineer.py --format parquet --workers 8" Enter

# Monitor: tmux attach -t tlc_engineering
# Detach: Ctrl+B then D
```

#### Option 4: Using `systemd` (For long-running services)
```bash
# Create service file
sudo nano /etc/systemd/system/tlc-engineering.service
```

Add this content:
```ini
[Unit]
Description=NYC TLC Data Engineering Pipeline
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/NYC_TLC
ExecStart=/usr/bin/python3 scripts/data_engineer.py --format parquet --workers 8
Restart=on-failure
RestartSec=60

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl start tlc-engineering
sudo systemctl status tlc-engineering
sudo journalctl -u tlc-engineering -f  # View logs
```

### Performance Notes

**File Size**: Processing 360 files (23M+ rows each)
- **Sequential (1 worker)**: ~11 seconds per file
- **Parallel (8 workers)**: ~2-3 seconds per file (6-8x speedup)

**Output Format**:
- **Parquet**: 10-100x faster than CSV for large datasets
- **Compression**: snappy (Parquet) balances speed and file size

### Example Log Output

```
2026-08-04 21:39:13,131 - INFO - Processing 360 files. Output: /mnt/external/NYC_TLC/data/engineered
2026-08-04 21:39:13,131 - INFO - Output format: PARQUET
2026-08-04 21:39:19,102 - INFO - Loaded fhv_tripdata_2019-01.parquet (23159064 rows)
2026-08-04 21:39:30,615 - INFO - Engineered features for fhv_tripdata_2019-01.parquet (17 columns)

===== PROCESSING SUMMARY =====
Total files: 360
Newly processed: 342
Already existed (skipped): 15
Failed: 0
Skipped (non-CSV/Parquet): 3
```
