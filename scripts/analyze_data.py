import pandas as pd
import os
from typing import Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Define data path
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_SCRIPT_PATH = os.path.dirname(SCRIPT_PATH)
DATA_PATH = os.path.join(PARENT_SCRIPT_PATH, "data")
output_path = os.path.join(PARENT_SCRIPT_PATH, "metadata")
os.makedirs(output_path, exist_ok=True)

def load_data_file(file_path: str, file_type: str = "parquet") -> pd.DataFrame:
    """
    Load data file from parquet or CSV format.
    
    Args:
        file_path: Path to the data file
        file_type: File type ('parquet' or 'csv')
    
    Returns:
        DataFrame with the loaded data
    """
    if file_type == "parquet":
        return pd.read_parquet(file_path)
    elif file_type == "csv":
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def crawl_folder(folder_path: str) -> list:
    """
    Crawl a folder and return a list of all files using multi-threading.
    
    Args:
        folder_path: Path to the folder
    
    Returns:
        List of file paths
    """
    file_list = []
    
    def walk_directory(directory: str) -> list:
        """Walk a single directory and return file paths."""
        files = []
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    files.append(item_path)
                elif os.path.isdir(item_path):
                    files.extend(walk_directory(item_path))
        except PermissionError:
            pass
        return files
    
    # Use multi-threading for directory traversal
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Get immediate subdirectories to parallelize
        try:
            root_items = os.listdir(folder_path)
            futures = {}
            
            for item in root_items:
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    futures[executor.submit(walk_directory, item_path)] = item_path
                elif os.path.isfile(item_path):
                    file_list.append(item_path)
            
            # Collect results as they complete
            for future in as_completed(futures):
                file_list.extend(future.result())
        except Exception as e:
            print(f"Error crawling folder: {e}")
    
    return file_list

def analyze_data_folder(folder_path: str) -> pd.DataFrame:
    """
    Analyze all data files in a folder and return a summary DataFrame.
    Uses multi-threading for parallel file processing.
    
    Args:
        folder_path: Path to the folder containing data files
    
    Returns:
        DataFrame with columns ["file_name", "file_path", "row_count", "column_count"]
    """
    data_files = crawl_folder(folder_path)
    df_base = pd.DataFrame(columns=["file_name", "file_path", "row_count", "column_count"])
    
    def process_file(file_path: str) -> pd.DataFrame:
        """Process a single file and return its analysis as a DataFrame."""
        try:
            file_name = os.path.basename(file_path)
            file_name_split = file_name.split('_')
            data_type = file_name_split[0]
            data_date = file_name_split[2]
            data_date_year = data_date.split('-')[0]
            data_date_month = data_date.split('-')[1].split('.')[0]

            df = load_data_file(file_path)
            df_temp = pd.DataFrame(
                {
                    "file_name": [file_name],
                    "file_path": [file_path],
                    "file_type": [data_type],
                    "data_date_year": [data_date_year],
                    "data_date_month": [data_date_month],
                    "row_count": [df.shape[0]],
                    "column_count": [df.shape[1]]
                }
            )
            return df_temp
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return pd.DataFrame()
    
    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_file, file_path): file_path for file_path in data_files}
        
        for future in as_completed(futures):
            try:
                df_temp = future.result()
                if not df_temp.empty:
                    df_base = pd.concat([df_base, df_temp], ignore_index=True)
            except Exception as e:
                print(f"Error in thread: {e}")
    
    return df_base

def main():
    # Analyze 
    df_base = analyze_data_folder(DATA_PATH)
    # sort for ease of read
    df_base = df_base.sort_values('file_name')
    # export csv
    output_name = os.path.join(output_path, "data_column_and_row_check.csv")
    df_base.to_csv(output_name, index=False)

    # print agg 
    print('=' * 60)
    print('SUMMARY STATISTICS'.center(60))
    print('=' * 60)
    
    # Overall metrics
    print('\n📊 OVERALL METRICS:')
    print(f'  Total Files: {len(df_base)}')
    print(f'  Total Rows: {df_base["row_count"].sum():,}')
    print(f'  Total Columns: {df_base["column_count"].sum()}')
    print(f'  Avg Rows per File: {df_base["row_count"].mean():,.0f}')
    print(f'  Avg Columns per File: {df_base["column_count"].mean():.1f}')
    
    # By file type
    print('\n📁 BY FILE TYPE (Row Count):')
    agg_by_type = df_base.groupby('file_type')['row_count'].agg(['sum', 'count', 'mean'])
    agg_by_type.columns = ['Total Rows', 'File Count', 'Avg Rows']
    agg_by_type['Total Rows'] = agg_by_type['Total Rows'].apply(lambda x: f'{x:,}')
    agg_by_type['Avg Rows'] = agg_by_type['Avg Rows'].apply(lambda x: f'{x:,.0f}')
    print(agg_by_type.to_string())
    
    # By year
    print('\n📅 BY YEAR (Row Count):')
    agg_by_year = df_base.groupby('data_date_year')['row_count'].agg(['sum', 'count', 'mean'])
    agg_by_year.columns = ['Total Rows', 'File Count', 'Avg Rows']
    agg_by_year['Total Rows'] = agg_by_year['Total Rows'].apply(lambda x: f'{x:,}')
    agg_by_year['Avg Rows'] = agg_by_year['Avg Rows'].apply(lambda x: f'{x:,.0f}')
    print(agg_by_year.to_string())
    
    # By type and year
    print('\n🔍 BY YEAR & FILE TYPE (Row Count):')
    agg_by_type_year = df_base.groupby(['data_date_year', 'file_type'])['row_count'].sum().unstack(fill_value=0)
    # Format with proper alignment
    agg_by_type_year_formatted = agg_by_type_year.map(lambda x: f'{int(x):>15,}' if x > 0 else '-'.rjust(15))
    print(agg_by_type_year_formatted.to_string())
    
    print('\n' + '=' * 60)

if __name__ == "__main__":
    main()