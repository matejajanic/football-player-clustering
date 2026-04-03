from __future__ import annotations

import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """
    return df.drop_duplicates().copy()


def fill_numeric_missing(
    df: pd.DataFrame,
    columns: list[str],
    value: float = 0.0,
) -> pd.DataFrame:
    """
    Fill missing values in numeric columns with a given value.
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(value)

    return df


def fill_categorical_missing(
    df: pd.DataFrame,
    columns: list[str],
    value: str = "Unknown",
) -> pd.DataFrame:
    """
    Fill missing values in categorical columns with a placeholder.
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(value)

    return df


def cast_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Safely cast selected columns to numeric.
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def select_relevant_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Keep only relevant columns that exist in the dataframe.
    """
    existing_columns = [col for col in columns if col in df.columns]
    return df[existing_columns].copy()