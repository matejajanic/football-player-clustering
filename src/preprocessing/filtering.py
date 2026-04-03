from __future__ import annotations

import pandas as pd


def filter_players_by_minutes(df: pd.DataFrame, min_minutes: int = 900) -> pd.DataFrame:
    """
    Keep only players with at least min_minutes played.
    """
    return df[df["total_minutes"] >= min_minutes].copy()


def filter_players_by_matches(df: pd.DataFrame, min_matches: int = 10) -> pd.DataFrame:
    """
    Keep only players with at least min_matches appearances.
    """
    return df[df["matches_played"] >= min_matches].copy()


def remove_zero_minute_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove players with zero total minutes.
    """
    return df[df["total_minutes"] > 0].copy()


def drop_players_without_core_features(
    df: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    """
    Drop rows with missing values in required columns.
    """
    existing_required = [col for col in required_columns if col in df.columns]
    return df.dropna(subset=existing_required).copy()