from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import RAW_DATA_DIR


def list_csv_files(data_dir: Path | None = None) -> list[Path]:
    """
    Return all CSV files in the raw data directory.
    """
    directory = data_dir or RAW_DATA_DIR
    return sorted(directory.glob("*.csv"))


def load_csv(file_path: Path, low_memory: bool = False) -> pd.DataFrame:
    """
    Load a single CSV file into a pandas DataFrame.
    """
    return pd.read_csv(file_path, low_memory=low_memory)


def load_all_csvs(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Load all CSV files from the raw data directory and return them
    as a dictionary: {stem_name: dataframe}.
    """
    directory = data_dir or RAW_DATA_DIR
    dataframes: dict[str, pd.DataFrame] = {}

    for csv_file in list_csv_files(directory):
        dataframes[csv_file.stem] = load_csv(csv_file)

    return dataframes


def dataframe_overview(df: pd.DataFrame) -> dict:
    """
    Produce a compact overview of a DataFrame.
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def print_dataset_summary(dataframes: dict[str, pd.DataFrame]) -> None:
    """
    Print a concise summary for each loaded DataFrame.
    """
    for name, df in dataframes.items():
        print("=" * 80)
        print(f"TABLE: {name}")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print("Columns:")
        for col in df.columns:
            print(f"  - {col}")
        print()