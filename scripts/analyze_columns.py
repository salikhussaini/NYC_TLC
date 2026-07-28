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


def load_multiple_files(file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load multiple data files.
    
    Args:
        file_paths: Dictionary with {name: file_path}
    
    Returns:
        Dictionary with {name: DataFrame}
    """
    dataframes = {}
    for name, path in file_paths.items():
        print(f"Loading {name}...")
        df = load_data_file(path)
        dataframes[name] = df
        print(f"  ✓ {len(df)} rows, {len(df.columns)} columns\n")
    return dataframes


def find_common_columns(dataframes: Dict[str, pd.DataFrame]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Find common columns across all dataframes and unique columns per dataframe.
    
    Args:
        dataframes: Dictionary with {name: DataFrame}
    
    Returns:
        Tuple of (common_columns, unique_columns_per_df)
    """
    column_sets = {name: set(df.columns) for name, df in dataframes.items()}
    
    # Find common columns
    common = set.intersection(*column_sets.values()) if column_sets else set()
    
    # Find unique columns
    unique = {name: cols - common for name, cols in column_sets.items()}
    
    return common, unique, column_sets


def create_sample_csv(df: pd.DataFrame, output_path: str, sample_size: int = 5, columns: list = None) -> None:
    """
    Create a sample CSV file from a dataframe.
    
    Args:
        df: Input DataFrame
        output_path: Path to save the CSV file
        sample_size: Number of rows to sample
        columns: Specific columns to include (None = all)
    """
    if columns:
        sample = df[columns].head(sample_size).copy()
    else:
        sample = df.head(sample_size).copy()
    
    sample.to_csv(output_path, index=False)
    print(f"✓ Created {os.path.basename(output_path)} ({len(sample)} rows, {len(sample.columns)} columns)")


def save_column_analysis(output_path: str, column_sets: Dict[str, Set[str]], 
                         common_cols: Set[str], unique_cols: Dict[str, Set[str]]) -> None:
    """
    Save column analysis to a text file.
    
    Args:
        output_path: Path to save the analysis file
        column_sets: Dictionary of {name: column_set}
        common_cols: Set of common columns
        unique_cols: Dictionary of {name: unique_columns}
    """
    with open(output_path, "w") as f:
        f.write("NYC TLC TAXI DATA - COLUMN ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        

        # Write common columns
        f.write(f"COMMON COLUMNS ({len(common_cols)}):\n")
        f.write("-" * 60 + "\n")
        for col in sorted(common_cols):
            f.write(f"  • {col}\n")
        f.write("\n")
            
    print(f"✓ Created {os.path.basename(output_path)}")


def main():
    """Main execution function."""
    # Define files to load - diverse random samples across years and months
    files_to_load = {
        "green_2019_01": os.path.join(DATA_PATH, "green_tripdata_2019-01.parquet"),
        "green_2020_06": os.path.join(DATA_PATH, "green_tripdata_2020-06.parquet"),
        "green_2021_12": os.path.join(DATA_PATH, "green_tripdata_2021-12.parquet"),
        "green_2022_03": os.path.join(DATA_PATH, "green_tripdata_2022-03.parquet"),
        "green_2023_09": os.path.join(DATA_PATH, "green_tripdata_2023-09.parquet"),
        "green_2024_01": os.path.join(DATA_PATH, "green_tripdata_2024-01.parquet"),
        "green_2025_06": os.path.join(DATA_PATH, "green_tripdata_2025-06.parquet"),
        "green_2026_01": os.path.join(DATA_PATH, "green_tripdata_2026-01.parquet"),
        "yellow_2019_06": os.path.join(DATA_PATH, "yellow_tripdata_2019-06.parquet"),
        "yellow_2020_12": os.path.join(DATA_PATH, "yellow_tripdata_2020-12.parquet"),
        "yellow_2021_03": os.path.join(DATA_PATH, "yellow_tripdata_2021-03.parquet"),
        "yellow_2022_09": os.path.join(DATA_PATH, "yellow_tripdata_2022-09.parquet"),
        "yellow_2023_01": os.path.join(DATA_PATH, "yellow_tripdata_2023-01.parquet"),
        "yellow_2024_06": os.path.join(DATA_PATH, "yellow_tripdata_2024-06.parquet"),
        "yellow_2025_12": os.path.join(DATA_PATH, "yellow_tripdata_2025-12.parquet"),
        "yellow_2026_02": os.path.join(DATA_PATH, "yellow_tripdata_2026-02.parquet"),
    }

    

    # Load all files
    print("=" * 60)
    print("LOADING DATA FILES")
    print("=" * 60 + "\n")
    dataframes = load_multiple_files(files_to_load)
    
    # Find common and unique columns
    print("=" * 60)
    print("ANALYZING COLUMNS")
    print("=" * 60 + "\n")
    common_cols, unique_cols, column_sets = find_common_columns(dataframes)
    
    print(f"Common columns found: {len(common_cols)}")
    print(f"  {sorted(common_cols)}\n")
    
    for name, cols in unique_cols.items():
        if cols:
            print(f"{name.upper()}-only columns: {len(cols)}")
            print(f"  {sorted(cols)}\n")
    
    
    # Create sample with common columns only
    if common_cols and dataframes:
        first_df = next(iter(dataframes.values()))
        output_file = os.path.join(output_path, "sample_common_columns.csv")
        #create_sample_csv(first_df, output_file, sample_size=5, columns=sorted(common_cols))
    
    print()
    
    # Save analysis report
    print("=" * 60)
    print("SAVING ANALYSIS REPORT")
    print("=" * 60 + "\n")
    
    analysis_file = os.path.join(output_path, "column_analysis.txt")
    save_column_analysis(analysis_file, column_sets, common_cols, unique_cols)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Common columns: {len(common_cols)}")


if __name__ == "__main__":
    main()
