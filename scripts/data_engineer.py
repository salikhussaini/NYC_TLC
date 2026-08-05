import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple

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
    }
}

def engineer_data(df: pd.DataFrame, taxi_type: str) -> pd.DataFrame:
    """
    Engineer features for taxi data (yellow, green, or fhvhv).
    
    Args:
        df: Input dataframe
        taxi_type: One of 'yellow', 'green', 'fhvhv'
    
    Returns:
        DataFrame with engineered features
    
    Raises:
        ValueError: If taxi_type is invalid or required columns are missing
        TypeError: If data conversion fails
    """
    if taxi_type not in TAXI_CONFIG:
        raise ValueError(f"Unknown taxi type: {taxi_type}. Must be one of {list(TAXI_CONFIG.keys())}")
    
    df = df.copy()
    config = TAXI_CONFIG[taxi_type]
    
    try:
        # Validate required columns
        required_cols = [config['pickup_col'], config['dropoff_col'], config['distance_col']]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
        
        # ===== TEMPORAL FEATURES =====
        df['pickup_datetime'] = pd.to_datetime(df[config['pickup_col']], errors='coerce')
        df['dropoff_datetime'] = pd.to_datetime(df[config['dropoff_col']], errors='coerce')
        
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['pickup_date'] = df['pickup_datetime'].dt.date
        df['is_weekend'] = df['pickup_day_of_week'].isin([5, 6]).astype(int)
        df['is_peak_hour'] = df['pickup_hour'].isin(PEAK_HOURS).astype(int)

        # ===== TRIP DURATION =====
        df['trip_duration_minutes'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60
        df['trip_duration_seconds'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()

        # ===== SPEED & DISTANCE METRICS =====
        df['trip_speed_mph'] = np.where(
            df['trip_duration_minutes'] > 0,
            (df[config['distance_col']] / df['trip_duration_minutes']) * 60,
            0
        )
        df['distance_category'] = pd.cut(df[config['distance_col']], 
                                         bins=[0, 2, 5, 10, 100], 
                                         labels=['Short', 'Medium', 'Long', 'Very Long'])

        # ===== TAXI-TYPE SPECIFIC FEATURES =====
        if taxi_type == 'yellow':
            _engineer_yellow_features(df)
        elif taxi_type == 'green':
            _engineer_green_features(df)
        elif taxi_type == 'fhvhv':
            _engineer_fhvhv_features(df)

        # ===== COMMON FINANCIAL METRICS =====
        _engineer_financial_metrics(df, taxi_type, config)

        # ===== LOCATION FEATURES =====
        df['is_airport_trip'] = ((df['PULocationID'].isin(AIRPORT_LOCATION_IDS)) | 
                                 (df['DOLocationID'].isin(AIRPORT_LOCATION_IDS))).astype(int)

        # ===== DATA QUALITY FLAGS =====
        _engineer_quality_flags(df, taxi_type, config)

        return df
    
    except Exception as e:
        logger.error(f"Error engineering {taxi_type} data: {str(e)}")
        raise


def _engineer_yellow_features(df: pd.DataFrame) -> None:
    """Engineer yellow taxi specific features."""
    # Yellow has 'extra' field and airport fee
    if 'fare_amount' in df.columns and 'passenger_count' in df.columns:
        df['per_passenger_fare'] = df['fare_amount'] / df['passenger_count'].replace(0, 1)
        df['per_passenger_cost'] = df['total_amount'] / df['passenger_count'].replace(0, 1)


def _engineer_green_features(df: pd.DataFrame) -> None:
    """Engineer green taxi specific features."""
    if 'trip_type' in df.columns:
        df['trip_type_name'] = df['trip_type'].map({
            1: 'Street-hail', 
            2: 'Dispatch'
        })
    
    if 'fare_amount' in df.columns and 'passenger_count' in df.columns:
        df['per_passenger_fare'] = df['fare_amount'] / df['passenger_count'].replace(0, 1)
        df['per_passenger_cost'] = df['total_amount'] / df['passenger_count'].replace(0, 1)


def _engineer_fhvhv_features(df: pd.DataFrame) -> None:
    """Engineer FHVHV (for-hire high-volume) specific features."""
    # Request response time metrics
    if 'request_datetime' in df.columns and 'on_scene_datetime' in df.columns:
        df['request_datetime'] = pd.to_datetime(df['request_datetime'], errors='coerce')
        df['on_scene_datetime'] = pd.to_datetime(df['on_scene_datetime'], errors='coerce')
        df['request_to_pickup_minutes'] = (df['pickup_datetime'] - df['request_datetime']).dt.total_seconds() / 60
        df['request_to_onscene_minutes'] = (df['on_scene_datetime'] - df['request_datetime']).dt.total_seconds() / 60

    # Driver performance metrics
    if 'driver_pay' in df.columns and 'trip_miles' in df.columns:
        df['driver_earnings'] = df['driver_pay'].fillna(0)
        df['driver_earnings_per_mile'] = np.where(
            df['trip_miles'] > 0,
            df['driver_earnings'] / df['trip_miles'],
            0
        )
        df['driver_earnings_per_minute'] = np.where(
            df['trip_duration_minutes'] > 0,
            df['driver_earnings'] / df['trip_duration_minutes'],
            0
        )

    # Platform commission
    if 'base_passenger_fare' in df.columns and 'driver_pay' in df.columns and 'total_passenger_cost' in df.columns:
        df['platform_commission'] = df['total_passenger_cost'] - df['driver_earnings']
        df['commission_percentage'] = np.where(
            df['total_passenger_cost'] > 0,
            (df['platform_commission'] / df['total_passenger_cost']) * 100,
            0
        )

    # Special service flags
    flag_cols = ['shared_request_flag', 'shared_match_flag', 'access_a_ride_flag', 
                 'wav_request_flag', 'wav_match_flag']
    for col in flag_cols:
        if col in df.columns:
            new_col = col.replace('_flag', '')
            df[f'is_{new_col}'] = (df[col] == 'Y').astype(int)

    df['is_accessibility_trip'] = df['is_wav_match'].astype(int) if 'is_wav_match' in df.columns else 0

    # License type mapping
    if 'hvfhs_license_num' in df.columns:
        df['license_type'] = df['hvfhs_license_num'].map({
            'HV0003': 'Uber',
            'HV0005': 'Lyft',
            'HV0004': 'Via',
            'HV0002': 'Juno',
            'HV0001': 'Uber',
        }).fillna('Other')


def _engineer_financial_metrics(df: pd.DataFrame, taxi_type: str, config: Dict) -> None:
    """Engineer common financial metrics."""
    if 'fare_amount' not in df.columns:
        return

    # Common financial calculations
    if 'total_amount' in df.columns and config['distance_col'] in df.columns:
        df['cost_per_mile'] = np.where(
            df[config['distance_col']] > 0,
            df['total_amount'] / df[config['distance_col']],
            0
        )
    
    if 'total_amount' in df.columns:
        df['cost_per_minute'] = np.where(
            df['trip_duration_minutes'] > 0,
            df['total_amount'] / df['trip_duration_minutes'],
            0
        )

    if 'fare_amount' in df.columns and config['distance_col'] in df.columns:
        df['revenue_per_mile'] = np.where(
            df[config['distance_col']] > 0,
            df['fare_amount'] / df[config['distance_col']],
            0
        )

    # Tip percentage
    if 'tip_amount' in df.columns:
        df['tip_percentage'] = np.where(
            df['fare_amount'] > 0,
            (df['tip_amount'] / df['fare_amount']) * 100,
            0
        )

    # Payment type
    if 'payment_type' in df.columns:
        df['payment_type_name'] = df['payment_type'].map(PAYMENT_TYPE_MAP)
        df['is_cash_payment'] = (df['payment_type'] == 2).astype(int)

    # Surcharge metrics - only for non-fhvhv types
    if taxi_type != 'fhvhv':
        _calculate_surcharges(df, taxi_type)


def _calculate_surcharges(df: pd.DataFrame, taxi_type: str) -> None:
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
        return

    # Filter to only existing columns
    surcharge_cols = [col for col in surcharge_cols if col in df.columns]
    optional_cols = [col for col in optional_cols if col in df.columns]
    
    if surcharge_cols or optional_cols:
        df['total_surcharges'] = sum(df[col].fillna(0) if col in optional_cols else df[col] 
                                     for col in surcharge_cols + optional_cols)
        
        if 'fare_amount' in df.columns:
            df['surcharge_percentage'] = np.where(
                df['fare_amount'] > 0,
                (df['total_surcharges'] / df['fare_amount']) * 100,
                0
            )


def _engineer_quality_flags(df: pd.DataFrame, taxi_type: str, config: Dict) -> None:
    """Engineer data quality flags."""
    if 'fare_amount' in df.columns:
        df['zero_fare_flag'] = (df['fare_amount'] == 0).astype(int)
    
    if config['distance_col'] in df.columns:
        df['zero_distance_flag'] = (df[config['distance_col']] == 0).astype(int)
    
    df['negative_duration_flag'] = (df['trip_duration_minutes'] < 0).astype(int)
    df['excessive_speed_flag'] = (df['trip_speed_mph'] > 100).astype(int)
    
    if taxi_type == 'fhvhv' and 'request_to_pickup_minutes' in df.columns:
        df['negative_request_response'] = (df['request_to_pickup_minutes'] < 0).astype(int)


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


def engineer_files(file_path: str) -> Tuple[bool, pd.DataFrame, str]:
    """
    Read and engineer a taxi data file.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        Tuple of (success: bool, dataframe: pd.DataFrame or None, message: str)
    """
    file_name = os.path.basename(file_path)
    
    # Determine taxi type
    taxi_type = None
    for key in TAXI_CONFIG.keys():
        if key in file_path.lower():
            taxi_type = key
            break
    
    if not taxi_type:
        return False, None, f"Could not determine taxi type for: {file_name}"
    
    try:
        if not os.path.exists(file_path):
            return False, None, f"File not found: {file_path}"
        
        if not file_path.lower().endswith('.csv'):
            return False, None, f"Skipping non-CSV file: {file_name}"
        
        # Read CSV
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {file_name} ({len(df)} rows)")
        
        # Engineer features
        eng_df = engineer_data(df, taxi_type)
        logger.info(f"Engineered features for {file_name} ({eng_df.shape[1]} columns)")
        
        return True, eng_df, f"Successfully processed {file_name}"
    
    except pd.errors.EmptyDataError:
        return False, None, f"Empty CSV file: {file_name}"
    except pd.errors.ParserError as e:
        return False, None, f"CSV parsing error in {file_name}: {str(e)}"
    except ValueError as e:
        return False, None, f"Value error in {file_name}: {str(e)}"
    except Exception as e:
        return False, None, f"Unexpected error processing {file_name}: {str(e)}"


def engineer_all(file_list: List[str], engineer_folder: str) -> Dict:
    """
    Process all files and save engineered versions.
    
    Args:
        file_list: List of file paths to process
        engineer_folder: Output directory for engineered files
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total': len(file_list),
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'errors': []
    }
    
    if not file_list:
        logger.warning("No files to process")
        return stats
    
    os.makedirs(engineer_folder, exist_ok=True)
    logger.info(f"Processing {len(file_list)} files. Output: {engineer_folder}")
    
    for file_path in file_list:
        success, eng_df, message = engineer_files(file_path)
        
        if not success:
            if 'non-CSV' in message:
                stats['skipped'] += 1
            else:
                stats['failed'] += 1
                stats['errors'].append(message)
            logger.warning(message)
            continue
        
        try:
            file_name = os.path.basename(file_path)
            output_path = os.path.join(engineer_folder, file_name)
            
            # Check for existing file
            if os.path.exists(output_path):
                logger.warning(f"Overwriting existing file: {output_path}")
            
            eng_df.to_csv(output_path, index=False)
            stats['successful'] += 1
            logger.info(f"Saved engineered data to {output_path}")
        
        except Exception as e:
            stats['failed'] += 1
            error_msg = f"Failed to save {file_name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(error_msg)
    
    # Summary
    logger.info(f"\n===== PROCESSING SUMMARY =====")
    logger.info(f"Total files: {stats['total']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    
    if stats['errors']:
        logger.warning(f"Errors encountered:")
        for error in stats['errors']:
            logger.warning(f"  - {error}")
    
    return stats


def main():
    """Main entry point for data engineering pipeline."""
    try:
        # Define data folders - use absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(os.path.dirname(script_dir), 'data')
        engineer_folder = os.path.join(data_folder, 'engineered')
        
        logger.info(f"Starting data engineering pipeline")
        logger.info(f"Data folder: {data_folder}")
        logger.info(f"Output folder: {engineer_folder}")
        
        # Validate data folder exists
        if not os.path.isdir(data_folder):
            logger.error(f"Data folder not found: {data_folder}")
            return
        
        # Crawl and process files
        file_list = crawl_folder(data_folder)
        
        if not file_list:
            logger.warning("No files found to process")
            return
        
        stats = engineer_all(file_list, engineer_folder)
        
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
    success = main()
    exit(0 if success else 1)