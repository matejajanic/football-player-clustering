from __future__ import annotations

import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def fill_numeric_missing(
    df: pd.DataFrame,
    columns: list[str],
    value: float = 0.0,
) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(value)

    return df


def fill_numeric_missing_with_median(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    return df


def fill_categorical_missing(
    df: pd.DataFrame,
    columns: list[str],
    value: str = "Unknown",
) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(value)

    return df


def cast_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def select_relevant_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing_columns = [col for col in columns if col in df.columns]
    return df[existing_columns].copy()


def drop_rows_with_missing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing_columns = [col for col in columns if col in df.columns]
    return df.dropna(subset=existing_columns).copy()

def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [col for col in columns if col in df.columns]
    return pd.get_dummies(df, columns=existing, drop_first=True)