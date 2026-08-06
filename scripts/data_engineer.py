import polars as pl
import os
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
PAYMENT_TYPE_MAP = {
    1: 'Credit Card', 
    2: 'Cash', 
    3: 'No Charge', 
    4: 'Dispute',
    5: 'Unknown'
}

AIRPORT_LOCATION_IDS = [132, 138]
PEAK_HOURS = [7, 8, 9, 17, 18, 19]

# Taxi type configurations with source column mappings
TAXI_CONFIG = {
    'yellow': {
        'pickup_col': 'tpep_pickup_datetime',
        'dropoff_col': 'tpep_dropoff_datetime',
        'distance_col': 'trip_distance',
        'extra_surcharges': ['extra', 'Airport_fee'],
        'has_trip_type': False,
    },
    'green': {
        'pickup_col': 'lpep_pickup_datetime',
        'dropoff_col': 'lpep_dropoff_datetime',
        'distance_col': 'trip_distance',
        'extra_surcharges': ['ehail_fee'],
        'has_trip_type': True,
    },
    'fhvhv': {
        'pickup_col': 'pickup_datetime',
        'dropoff_col': 'dropoff_datetime',
        'distance_col': 'trip_miles',
        'extra_surcharges': ['sales_tax'],
        'has_trip_type': False,
        'is_fhvhv': True,
    },
    'fhv': {
        'pickup_col': 'pickup_datetime',
        'dropoff_col': 'dropoff_datetime',  # May not exist in FHV
        'distance_col': 'trip_miles',  # May not exist in FHV
        'extra_surcharges': [],
        'has_trip_type': False,
        'is_fhv': True,  # Standard FHV (different from FHVHV)
    }
}

def engineer_data(df: pl.DataFrame, taxi_type: str) -> pl.DataFrame:
    """
    Engineer features for taxi data (yellow, green, fhvhv, or fhv).
    
    Args:
        df: Input dataframe
        taxi_type: One of 'yellow', 'green', 'fhvhv', 'fhv'
    
    Returns:
        DataFrame with engineered features
    
    Raises:
        ValueError: If taxi_type is invalid or required columns are missing
        TypeError: If data conversion fails
    """
    if taxi_type not in TAXI_CONFIG:
        raise ValueError(f"Unknown taxi type: {taxi_type}. Must be one of {list(TAXI_CONFIG.keys())}")
    
    config = TAXI_CONFIG[taxi_type]
    
    try:
        # For FHV files: only pickup_datetime is required (dropoff/distance may not exist)
        if taxi_type == 'fhv':
            required_cols = [config['pickup_col']]
        else:
            required_cols = [config['pickup_col'], config['dropoff_col'], config['distance_col']]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
        
        # ===== TEMPORAL FEATURES =====
        # Prepare datetime columns in a single batch
        temporal_cols = {}
        
        # Handle pickup datetime
        pickup_col_data = df[config['pickup_col']]
        if pickup_col_data.dtype in [pl.Datetime('us'), pl.Datetime('ns'), pl.Datetime('ms')]:
            temporal_cols['pickup_datetime'] = pl.col(config['pickup_col'])
        else:
            temporal_cols['pickup_datetime'] = pl.col(config['pickup_col']).str.to_datetime(format=None)
        
        # Handle dropoff datetime
        if config['dropoff_col'] in df.columns:
            dropoff_col_data = df[config['dropoff_col']]
            if dropoff_col_data.dtype in [pl.Datetime('us'), pl.Datetime('ns'), pl.Datetime('ms')]:
                temporal_cols['dropoff_datetime'] = pl.col(config['dropoff_col'])
            else:
                temporal_cols['dropoff_datetime'] = pl.col(config['dropoff_col']).str.to_datetime(format=None)
        else:
            temporal_cols['dropoff_datetime'] = pl.lit(None).cast(pl.Datetime('us'))
        
        # Add derived temporal features in single batch
        temporal_cols['pickup_hour'] = pl.col('pickup_datetime').dt.hour()
        temporal_cols['pickup_day_of_week'] = pl.col('pickup_datetime').dt.weekday()
        temporal_cols['pickup_date'] = pl.col('pickup_datetime').dt.date()
        temporal_cols['is_weekend'] = pl.col('pickup_day_of_week').is_in([5, 6]).cast(pl.Int32)
        temporal_cols['is_peak_hour'] = pl.col('pickup_hour').is_in(PEAK_HOURS).cast(pl.Int32)
        
        df = df.with_columns(**temporal_cols)

        # ===== TRIP DURATION & SPEED/DISTANCE METRICS =====
        # Batch calculate all derived metrics
        derived_metrics = {}
        
        # Trip duration
        if config['dropoff_col'] in df.columns:
            derived_metrics['trip_duration_minutes'] = (pl.col('dropoff_datetime') - pl.col('pickup_datetime')).dt.total_seconds() / 60
            derived_metrics['trip_duration_seconds'] = (pl.col('dropoff_datetime') - pl.col('pickup_datetime')).dt.total_seconds()
        else:
            derived_metrics['trip_duration_minutes'] = pl.lit(None).cast(pl.Float64)
            derived_metrics['trip_duration_seconds'] = pl.lit(None).cast(pl.Float64)
        
        # Speed and distance metrics
        if config['distance_col'] in df.columns:
            derived_metrics['trip_speed_mph'] = pl.when(pl.col('trip_duration_minutes') > 0).then(
                (pl.col(config['distance_col']) / pl.col('trip_duration_minutes')) * 60
            ).otherwise(0)
            
            derived_metrics['distance_category'] = (
                pl.when(pl.col(config['distance_col']) <= 2).then(pl.lit('Short'))
                .when(pl.col(config['distance_col']) <= 5).then(pl.lit('Medium'))
                .when(pl.col(config['distance_col']) <= 10).then(pl.lit('Long'))
                .otherwise(pl.lit('Very Long'))
            )
        else:
            derived_metrics['trip_speed_mph'] = pl.lit(None).cast(pl.Float64)
            derived_metrics['distance_category'] = pl.lit(None).cast(pl.Utf8)
        
        df = df.with_columns(**derived_metrics)

        # ===== TAXI-TYPE SPECIFIC FEATURES =====
        if taxi_type == 'yellow':
            df = _engineer_yellow_features(df)
        elif taxi_type == 'green':
            df = _engineer_green_features(df)
        elif taxi_type == 'fhvhv':
            df = _engineer_fhvhv_features(df)
        elif taxi_type == 'fhv':
            df = _engineer_fhv_features(df)

        # ===== COMMON FINANCIAL METRICS =====
        df = _engineer_financial_metrics(df, taxi_type, config)

        # ===== LOCATION FEATURES =====
        if 'PULocationID' in df.columns and 'DOLocationID' in df.columns:
            df = df.with_columns(
                (pl.col('PULocationID').is_in(AIRPORT_LOCATION_IDS) | 
                 pl.col('DOLocationID').is_in(AIRPORT_LOCATION_IDS)).cast(pl.Int32).alias('is_airport_trip')
            )

        # ===== DATA QUALITY FLAGS =====
        df = _engineer_quality_flags(df, taxi_type, config)

        return df
    
    except Exception as e:
        logger.error(f"Error engineering {taxi_type} data: {str(e)}")
        raise


def _engineer_fhv_features(df: pl.DataFrame) -> pl.DataFrame:
    """Engineer standard FHV (non-FHVHV) specific features."""
    # FHV has minimal supplemental columns
    # Mainly just pickup/dropoff locations and basic info
    # Check for dispatching info if it exists
    if 'on_scene_datetime' in df.columns:
        col_data = df['on_scene_datetime']
        if col_data.dtype not in [pl.Datetime('us'), pl.Datetime('ns'), pl.Datetime('ms')]:
            df = df.with_columns(pl.col('on_scene_datetime').str.to_datetime())
    return df


def _engineer_yellow_features(df: pl.DataFrame) -> pl.DataFrame:
    """Engineer yellow taxi specific features."""
    cols = {}
    if 'fare_amount' in df.columns and 'passenger_count' in df.columns:
        cols['per_passenger_fare'] = pl.col('fare_amount') / pl.col('passenger_count').fill_null(1)
        cols['per_passenger_cost'] = pl.col('total_amount') / pl.col('passenger_count').fill_null(1)
    
    if cols:
        df = df.with_columns(**cols)
    return df


def _engineer_green_features(df: pl.DataFrame) -> pl.DataFrame:
    """Engineer green taxi specific features."""
    cols = {}
    
    if 'trip_type' in df.columns:
        cols['trip_type_name'] = (
            pl.when(pl.col('trip_type') == 1).then(pl.lit('Street-hail'))
            .when(pl.col('trip_type') == 2).then(pl.lit('Dispatch'))
            .otherwise(None)
        )
    
    if 'fare_amount' in df.columns and 'passenger_count' in df.columns:
        cols['per_passenger_fare'] = pl.col('fare_amount') / pl.col('passenger_count').fill_null(1)
        cols['per_passenger_cost'] = pl.col('total_amount') / pl.col('passenger_count').fill_null(1)
    
    if cols:
        df = df.with_columns(**cols)
    return df


def _engineer_fhvhv_features(df: pl.DataFrame) -> pl.DataFrame:
    """Engineer FHVHV (for-hire high-volume) specific features."""
    cols = {}
    
    # Request response time metrics
    if 'request_datetime' in df.columns and 'on_scene_datetime' in df.columns:
        request_col_data = df['request_datetime']
        if request_col_data.dtype not in [pl.Datetime('us'), pl.Datetime('ns'), pl.Datetime('ms')]:
            df = df.with_columns(pl.col('request_datetime').str.to_datetime())
        
        onscene_col_data = df['on_scene_datetime']
        if onscene_col_data.dtype not in [pl.Datetime('us'), pl.Datetime('ns'), pl.Datetime('ms')]:
            df = df.with_columns(pl.col('on_scene_datetime').str.to_datetime())
        
        cols['request_to_pickup_minutes'] = (pl.col('pickup_datetime') - pl.col('request_datetime')).dt.total_seconds() / 60
        cols['request_to_onscene_minutes'] = (pl.col('on_scene_datetime') - pl.col('request_datetime')).dt.total_seconds() / 60

    # Driver performance metrics
    if 'driver_pay' in df.columns and 'trip_miles' in df.columns:
        driver_earnings = pl.col('driver_pay').fill_null(0)
        cols['driver_earnings'] = driver_earnings
        cols['driver_earnings_per_mile'] = pl.when(pl.col('trip_miles') > 0).then(
            driver_earnings / pl.col('trip_miles')
        ).otherwise(0)
        cols['driver_earnings_per_minute'] = pl.when(pl.col('trip_duration_minutes') > 0).then(
            driver_earnings / pl.col('trip_duration_minutes')
        ).otherwise(0)

    # Platform commission
    if 'base_passenger_fare' in df.columns and 'driver_pay' in df.columns and 'total_passenger_cost' in df.columns:
        cols['platform_commission'] = pl.col('total_passenger_cost') - pl.col('driver_earnings')
        cols['commission_percentage'] = pl.when(pl.col('total_passenger_cost') > 0).then(
            (pl.col('platform_commission') / pl.col('total_passenger_cost')) * 100
        ).otherwise(0)

    # License type mapping
    if 'hvfhs_license_num' in df.columns:
        cols['license_type'] = (
            pl.when(pl.col('hvfhs_license_num') == 'HV0003').then(pl.lit('Uber'))
            .when(pl.col('hvfhs_license_num') == 'HV0005').then(pl.lit('Lyft'))
            .when(pl.col('hvfhs_license_num') == 'HV0004').then(pl.lit('Via'))
            .when(pl.col('hvfhs_license_num') == 'HV0002').then(pl.lit('Juno'))
            .when(pl.col('hvfhs_license_num') == 'HV0001').then(pl.lit('Uber'))
            .otherwise(pl.lit('Other'))
        )

    if cols:
        df = df.with_columns(**cols)

    # Special service flags - batch these separately
    flag_cols_dict = {}
    flag_cols = ['shared_request_flag', 'shared_match_flag', 'access_a_ride_flag', 
                 'wav_request_flag', 'wav_match_flag']
    for col in flag_cols:
        if col in df.columns:
            new_col = col.replace('_flag', '')
            flag_cols_dict[f'is_{new_col}'] = (pl.col(col) == 'Y').cast(pl.Int32)
    
    # Add accessibility flag
    if 'is_wav_match' in flag_cols_dict or 'is_wav_match' in df.columns:
        flag_cols_dict['is_accessibility_trip'] = pl.col('is_wav_match').cast(pl.Int32) if 'is_wav_match' in flag_cols_dict else pl.lit(0).cast(pl.Int32)
    else:
        flag_cols_dict['is_accessibility_trip'] = pl.lit(0).cast(pl.Int32)
    
    if flag_cols_dict:
        df = df.with_columns(**flag_cols_dict)

    return df


def _engineer_financial_metrics(df: pl.DataFrame, taxi_type: str, config: Dict) -> pl.DataFrame:
    """Engineer common financial metrics."""
    # Skip financial metrics for FHV (likely doesn't have fare/payment data)
    if taxi_type == 'fhv':
        return df
    
    if 'fare_amount' not in df.columns:
        return df

    cols = {}
    
    # Common financial calculations
    if 'total_amount' in df.columns and config['distance_col'] in df.columns:
        cols['cost_per_mile'] = pl.when(pl.col(config['distance_col']) > 0).then(
            pl.col('total_amount') / pl.col(config['distance_col'])
        ).otherwise(0)
    
    if 'total_amount' in df.columns and 'trip_duration_minutes' in df.columns:
        cols['cost_per_minute'] = pl.when(pl.col('trip_duration_minutes') > 0).then(
            pl.col('total_amount') / pl.col('trip_duration_minutes')
        ).otherwise(0)

    if 'fare_amount' in df.columns and config['distance_col'] in df.columns:
        cols['revenue_per_mile'] = pl.when(pl.col(config['distance_col']) > 0).then(
            pl.col('fare_amount') / pl.col(config['distance_col'])
        ).otherwise(0)

    # Tip percentage
    if 'tip_amount' in df.columns:
        cols['tip_percentage'] = pl.when(pl.col('fare_amount') > 0).then(
            (pl.col('tip_amount') / pl.col('fare_amount')) * 100
        ).otherwise(0)

    # Payment type
    if 'payment_type' in df.columns:
        cols['payment_type_name'] = (
            pl.when(pl.col('payment_type') == 1).then(pl.lit('Credit Card'))
            .when(pl.col('payment_type') == 2).then(pl.lit('Cash'))
            .when(pl.col('payment_type') == 3).then(pl.lit('No Charge'))
            .when(pl.col('payment_type') == 4).then(pl.lit('Dispute'))
            .when(pl.col('payment_type') == 5).then(pl.lit('Unknown'))
            .otherwise(None)
        )
        cols['is_cash_payment'] = (pl.col('payment_type') == 2).cast(pl.Int32)

    if cols:
        df = df.with_columns(**cols)

    # Surcharge metrics - only for non-fhvhv types
    if taxi_type != 'fhvhv':
        df = _calculate_surcharges(df, taxi_type)
    
    return df


def _calculate_surcharges(df: pl.DataFrame, taxi_type: str) -> pl.DataFrame:
    """Calculate surcharge-related metrics."""
    surcharge_cols = []
    
    if taxi_type == 'yellow':
        surcharge_cols = ['extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge',
                         'congestion_surcharge']
        optional_cols = ['Airport_fee', 'cbd_congestion_fee']
    elif taxi_type == 'green':
        surcharge_cols = ['mta_tax', 'tolls_amount', 'improvement_surcharge', 'congestion_surcharge']
        optional_cols = ['cbd_congestion_fee', 'ehail_fee']
    else:
        return df

    # Filter to only existing columns
    surcharge_cols = [col for col in surcharge_cols if col in df.columns]
    optional_cols = [col for col in optional_cols if col in df.columns]
    
    all_surcharge_cols = surcharge_cols + optional_cols
    
    if all_surcharge_cols:
        # Sum all surcharge columns (filling nulls with 0)
        total_surcharges_expr = sum(
            pl.col(col).fill_null(0) for col in all_surcharge_cols
        )
        
        cols = {'total_surcharges': total_surcharges_expr}
        
        if 'fare_amount' in df.columns:
            cols['surcharge_percentage'] = pl.when(pl.col('fare_amount') > 0).then(
                (pl.col('total_surcharges') / pl.col('fare_amount')) * 100
            ).otherwise(0)
        
        df = df.with_columns(**cols)
    
    return df


def _engineer_quality_flags(df: pl.DataFrame, taxi_type: str, config: Dict) -> pl.DataFrame:
    """Engineer data quality flags."""
    cols = {}
    
    if 'fare_amount' in df.columns:
        cols['zero_fare_flag'] = (pl.col('fare_amount') == 0).cast(pl.Int32)
    
    if config['distance_col'] in df.columns:
        cols['zero_distance_flag'] = (pl.col(config['distance_col']) == 0).cast(pl.Int32)
    
    if 'trip_duration_minutes' in df.columns:
        cols['negative_duration_flag'] = (pl.col('trip_duration_minutes') < 0).cast(pl.Int32)
    
    if 'trip_speed_mph' in df.columns:
        cols['excessive_speed_flag'] = (pl.col('trip_speed_mph') > 100).cast(pl.Int32)
    
    if taxi_type == 'fhvhv' and 'request_to_pickup_minutes' in df.columns:
        cols['negative_request_response'] = (pl.col('request_to_pickup_minutes') < 0).cast(pl.Int32)
    
    if cols:
        df = df.with_columns(**cols)
    
    return df


def engineer_yellow(df_yellow):
    """Wrapper for backward compatibility. Use engineer_data() instead."""
    return engineer_data(df_yellow, 'yellow')


def engineer_green(df_green):
    """Wrapper for backward compatibility. Use engineer_data() instead."""
    return engineer_data(df_green, 'green')


def engineer_fhvhv(df_fhvhv):
    """Wrapper for backward compatibility. Use engineer_data() instead."""
    return engineer_data(df_fhvhv, 'fhvhv')

def crawl_folder(folder_path: str) -> List[str]:
    """
    Crawl a folder and return a list of all files using os.walk().
    
    Args:
        folder_path: Path to the folder
    
    Returns:
        List of file paths
    
    Raises:
        FileNotFoundError: If folder doesn't exist
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    
    logger.info(f"Found {len(file_list)} files in {folder_path}")
    return file_list


def engineer_files(file_path: str, output_format: str = 'csv') -> Tuple[bool, pl.DataFrame, str, str]:
    """
    Read and engineer a taxi data file (CSV or Parquet).
    
    Args:
        file_path: Path to the CSV or Parquet file
        output_format: Output format ('csv' or 'parquet')
    
    Returns:
        Tuple of (success: bool, dataframe: pl.DataFrame or None, message: str, output_filename: str)
    """
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()
    
    # Check if file is CSV or Parquet
    if file_ext not in ['.csv', '.parquet']:
        return False, None, f"Skipping non-CSV/Parquet file: {file_name}", ""
    
    # Determine taxi type - check in order: 'fhv_' before 'fhvhv' to avoid mismatches
    taxi_type = None
    filename_lower = file_path.lower()
    
    # Priority order: fhv_ (but not fhvhv), then fhvhv, then yellow, then green
    if 'fhv_' in filename_lower and 'fhvhv' not in filename_lower:
        taxi_type = 'fhv'
    else:
        for key in ['fhvhv', 'yellow', 'green']:
            if key in filename_lower:
                taxi_type = key
                break
    
    if not taxi_type:
        return False, None, f"Could not determine taxi type for: {file_name}", ""
    
    try:
        if not os.path.exists(file_path):
            return False, None, f"File not found: {file_path}", ""
        
        # Read file based on extension
        if file_ext == '.csv':
            df = pl.read_csv(file_path)
            base_filename = file_name
        else:  # .parquet
            df = pl.read_parquet(file_path)
            # Remove .parquet extension for base filename
            base_filename = file_name.replace('.parquet', '')
        
        # Engineer features
        eng_df = engineer_data(df, taxi_type)
        
        # Set output filename based on format
        if output_format == 'parquet':
            output_filename = base_filename + '.parquet'
        else:  # csv
            output_filename = base_filename + '.csv'
        
        return True, eng_df, f"Successfully processed {file_name}", output_filename
    
    except Exception as e:
        return False, None, f"Error processing {file_name}: {str(e)}", ""


def process_and_save_file(args: Tuple) -> Dict:
    """
    Process a single file and save engineered version.
    
    Args:
        args: Tuple of (file_path, output_format, engineer_folder, rerun)
    
    Returns:
        Dictionary with processing stats
    """
    file_path, output_format, engineer_folder, rerun, compression = args
    file_name = os.path.basename(file_path)
    
    success, eng_df, message, output_filename = engineer_files(file_path, output_format=output_format)
    
    result = {'file': file_name, 'success': success, 'message': message}
    
    if not success:
        result['status'] = 'failed' if 'Error' in message else 'skipped'
        return result
    
    output_path = os.path.join(engineer_folder, output_filename)
    
    # Skip if file already exists and rerun is False
    if os.path.exists(output_path) and not rerun:
        result['status'] = 'exists'
        return result
    
    try:
        if output_format == 'parquet':
            eng_df.write_parquet(output_path, compression=compression or 'snappy')
        else:
            eng_df.write_csv(output_path)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        result['status'] = 'saved'
        result['size_mb'] = file_size_mb
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result


def engineer_all(file_list: List[str], engineer_folder: str, rerun: bool = False, 
                 output_format: str = 'csv', compression: Optional[str] = None, 
                 num_workers: int = 0) -> Dict:
    """
    Process all files and save engineered versions using multiprocessing.
    
    Args:
        file_list: List of file paths to process
        engineer_folder: Output directory for engineered files
        rerun: If True, reprocess all files even if they exist
        output_format: Output format ('csv' or 'parquet')
        compression: Compression to use ('gzip' for csv; 'snappy', 'gzip', 'brotli', 'lz4' for parquet)
        num_workers: Number of parallel workers (0 = auto-detect CPU count)
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total': len(file_list),
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'already_exists': 0,
        'errors': []
    }
    
    if not file_list:
        logger.warning("No files to process")
        return stats
    
    os.makedirs(engineer_folder, exist_ok=True)
    
    # Set number of workers
    if num_workers <= 0:
        num_workers = cpu_count()
    
    logger.info(f"Processing {len(file_list)} files using {num_workers} workers. Output: {engineer_folder}")
    logger.info(f"Output format: {output_format.upper()}" + (f", Compression: {compression}" if compression else ""))
    
    # Prepare arguments for parallel processing
    process_args = [
        (file_path, output_format, engineer_folder, rerun, compression)
        for file_path in file_list
    ]
    
    # Process files in parallel
    with Pool(num_workers) as pool:
        results = pool.map(process_and_save_file, process_args)
    
    # Collect statistics
    for result in results:
        status = result.get('status', 'unknown')
        
        if status == 'saved':
            stats['successful'] += 1
            size = result.get('size_mb', 0)
            logger.info(f"Saved: {result['file']} ({size:.2f} MB)")
        elif status == 'exists':
            stats['already_exists'] += 1
        elif status == 'error':
            stats['failed'] += 1
            error_msg = f"Failed to save {result['file']}: {result.get('error', 'Unknown error')}"
            stats['errors'].append(error_msg)
            logger.error(error_msg)
        elif status == 'skipped':
            stats['skipped'] += 1
        elif status == 'failed':
            stats['failed'] += 1
            stats['errors'].append(result.get('message', 'Processing failed'))
    
    # Summary
    logger.info(f"\n===== PROCESSING SUMMARY =====")
    logger.info(f"Total files: {stats['total']}")
    logger.info(f"Newly processed: {stats['successful']}")
    logger.info(f"Already existed (skipped): {stats['already_exists']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped (non-CSV/Parquet): {stats['skipped']}")
    
    if stats['errors']:
        logger.warning(f"Errors encountered:")
        for error in stats['errors']:
            logger.warning(f"  - {error}")
    
    return stats


def main(rerun: bool = False, output_format: str = 'csv', compression: Optional[str] = None, num_workers: int = 0):
    """
    Main entry point for data engineering pipeline.
    
    Args:
        rerun: If True, reprocess all files even if engineered versions exist
        output_format: Output format ('csv' or 'parquet')
        compression: Compression to use ('gzip' for csv; 'snappy', 'gzip', 'brotli' for parquet)
        num_workers: Number of parallel workers (0 = auto-detect CPU count)
    """
    try:
        # Define data folders - use absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(os.path.dirname(script_dir), 'data')
        engineer_folder = os.path.join(data_folder, 'engineered')
        
        logger.info(f"Starting data engineering pipeline")
        if rerun:
            logger.info(f"Mode: RERUN (reprocessing all files)")
        else:
            logger.info(f"Mode: NORMAL (skipping existing files)")
        logger.info(f"Data folder: {data_folder}")
        logger.info(f"Output folder: {engineer_folder}")
        
        # Validate data folder exists
        if not os.path.isdir(data_folder):
            logger.error(f"Data folder not found: {data_folder}")
            return False
        
        # Crawl and process files
        file_list = crawl_folder(data_folder)
        
        if not file_list:
            logger.warning("No files found to process")
            return False
        
        stats = engineer_all(file_list, engineer_folder, rerun=rerun, 
                           output_format=output_format, compression=compression, num_workers=num_workers)
        
        # Exit with success/failure code
        if stats['failed'] > 0:
            logger.error(f"Pipeline completed with {stats['failed']} errors")
            return False
        else:
            logger.info("Pipeline completed successfully")
            return True
    
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {str(e)}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NYC TLC Data Engineering Pipeline (Yellow, Green, FHV, FHVHV)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_engineer.py                          # CSV output, no compression
  python data_engineer.py --format parquet         # Parquet output (faster, smaller)
  python data_engineer.py --format csv --compress gzip   # CSV with gzip compression
  python data_engineer.py --format parquet --compress snappy --rerun   # Parquet snappy, reprocess all

Supported file types:
  - yellow_tripdata_*.csv/parquet
  - green_tripdata_*.csv/parquet
  - fhv_tripdata_*.csv/parquet
  - fhvhv_tripdata_*.csv/parquet

Performance notes:
  - Parquet format is 10-100x faster for large files (23M+ rows)
  - CSV compression reduces file size by 60-80% but adds processing time
  - Parquet with snappy compression balances speed and file size
        """
    )
    parser.add_argument(
        '--rerun',
        action='store_true',
        help='Reprocess all files even if engineered versions already exist'
    )
    parser.add_argument(
        '--format',
        choices=['csv', 'parquet'],
        default='csv',
        help='Output file format (default: csv)'
    )
    parser.add_argument(
        '--compress',
        choices=['gzip', 'snappy', 'brotli', 'lz4'],
        default=None,
        help='Compression method (gzip for CSV; snappy/gzip/brotli/lz4 for Parquet)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Number of parallel workers (0 = auto-detect CPU count, default: 0)'
    )
    
    args = parser.parse_args()
    
    # Validate compression for format
    if args.format == 'csv' and args.compress and args.compress not in ['gzip']:
        parser.error(f"CSV format only supports 'gzip' compression, got '{args.compress}'")
    
    success = main(rerun=args.rerun, output_format=args.format, compression=args.compress, num_workers=args.workers)
    exit(0 if success else 1)