import pandas as pd
import os
from typing import Dict, Set, Tuple

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
    Crawl a folder and return a list of all files.
    
    Args:
        folder_path: Path to the folder
    
    Returns:
        List of file paths
    """
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def analyze_data_folder(folder_path: str) -> pd.DataFrame:
    """
    Analyze all data files in a folder and return a summary DataFrame.
    
    Args:
        folder_path: Path to the folder containing data files
    
    Returns:
        DataFrame with columns ["file_name", "file_path", "row_count", "column_count"]
    """
    data_files = crawl_folder(folder_path)
    df_base = pd.DataFrame(columns=["file_name", "file_path", "row_count", "column_count"])
    for file_path in data_files:
        file_name = os.path.basename(file_path)
        df = load_data_file(file_path)
        df_temp = pd.DataFrame(
            {
                "file_name": [file_name],
                "file_path": [file_path],
                "row_count": [df.shape[0]],
                "column_count": [df.shape[1]]
            }
        )
        df_base = pd.concat([df_base, df_temp], ignore_index=True)
    return df_base

def main():
    # Analyze 
    df_base = analyze_data_folder(DATA_PATH)

    # export csv
    output_name = os.path.join(output_path, "data_column_and_row_check.csv")
    df_base.to_csv(output_name, index=False)

if __name__ == "__main__":
    main()