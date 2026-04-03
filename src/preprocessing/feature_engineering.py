from __future__ import annotations

import pandas as pd


def aggregate_appearances(appearances: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate match-level appearance data into player-level performance features.
    """
    grouped = (
        appearances.groupby("player_id")
        .agg(
            matches_played=("appearance_id", "count"),
            total_minutes=("minutes_played", "sum"),
            total_goals=("goals", "sum"),
            total_assists=("assists", "sum"),
            total_yellow=("yellow_cards", "sum"),
            total_red=("red_cards", "sum"),
        )
        .reset_index()
    )

    return add_per90_features(grouped)


def add_per90_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-90-minute features for player performance statistics.
    """
    df = df.copy()

    minutes = df["total_minutes"].replace(0, pd.NA)

    df["goals_per_90"] = (df["total_goals"] / minutes) * 90
    df["assists_per_90"] = (df["total_assists"] / minutes) * 90
    df["yellow_per_90"] = (df["total_yellow"] / minutes) * 90
    df["red_per_90"] = (df["total_red"] / minutes) * 90

    df = df.fillna(0)

    return df


def aggregate_valuations(player_valuations: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate valuation history into player-level market value features.
    """
    return (
        player_valuations.groupby("player_id")
        .agg(
            avg_market_value=("market_value_in_eur", "mean"),
            max_market_value=("market_value_in_eur", "max"),
            valuation_records=("market_value_in_eur", "count"),
        )
        .reset_index()
    )


def add_age_feature(df: pd.DataFrame, reference_year: int = 2025) -> pd.DataFrame:
    """
    Compute an approximate player age using date_of_birth.
    Age is computed against a fixed reference year for reproducibility.
    """
    df = df.copy()

    if "date_of_birth" not in df.columns:
        return df

    dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"] = reference_year - dob.dt.year

    return df


def select_player_profile_columns(players: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the most relevant player profile columns for downstream analysis.
    """
    desired_columns = [
        "player_id",
        "name",
        "position",
        "sub_position",
        "foot",
        "height_in_cm",
        "date_of_birth",
        "country_of_birth",
        "country_of_citizenship",
        "current_club_id",
        "current_club_name",
        "current_club_domestic_competition_id",
        "market_value_in_eur",
        "highest_market_value_in_eur",
        "international_caps",
        "international_goals",
    ]

    existing_columns = [col for col in desired_columns if col in players.columns]
    return players[existing_columns].copy()


def build_player_dataset(
    players: pd.DataFrame,
    appearances: pd.DataFrame,
    valuations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a player-level dataset by combining:
    - aggregated performance data from appearances
    - selected player profile attributes
    - aggregated valuation history
    """
    player_profiles = select_player_profile_columns(players)
    appearance_features = aggregate_appearances(appearances)
    valuation_features = aggregate_valuations(valuations)

    df = appearance_features.merge(player_profiles, on="player_id", how="left")
    df = df.merge(valuation_features, on="player_id", how="left")
    df = add_age_feature(df)

    return df