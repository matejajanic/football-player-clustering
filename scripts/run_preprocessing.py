from __future__ import annotations

from src.config import PROCESSED_DATA_DIR
from src.data_loading.loader import load_all_csvs
from src.preprocessing.feature_engineering import build_player_dataset
from src.preprocessing.filtering import (
    filter_players_by_matches,
    filter_players_by_minutes,
    remove_zero_minute_players,
    drop_players_without_core_features,
)
from src.preprocessing.cleaning import (
    cast_numeric_columns,
    fill_categorical_missing,
    fill_numeric_missing,
    remove_duplicates,
)


def main() -> None:
    data = load_all_csvs()

    df = build_player_dataset(
        players=data["players"],
        appearances=data["appearances"],
        valuations=data["player_valuations"],
    )

    print("Initial shape:", df.shape)

    numeric_columns = [
        "matches_played",
        "total_minutes",
        "total_goals",
        "total_assists",
        "total_yellow",
        "total_red",
        "goals_per_90",
        "assists_per_90",
        "yellow_per_90",
        "red_per_90",
        "height_in_cm",
        "market_value_in_eur",
        "highest_market_value_in_eur",
        "avg_market_value",
        "max_market_value",
        "valuation_records",
        "international_caps",
        "international_goals",
        "age",
    ]

    categorical_columns = [
        "position",
        "sub_position",
        "foot",
        "country_of_birth",
        "country_of_citizenship",
        "current_club_name",
        "current_club_domestic_competition_id",
        "name",
    ]

    df = cast_numeric_columns(df, numeric_columns)
    df = remove_duplicates(df)
    df = remove_zero_minute_players(df)
    df = filter_players_by_minutes(df, min_minutes=900)
    df = filter_players_by_matches(df, min_matches=10)

    df = fill_numeric_missing(df, numeric_columns, value=0.0)
    df = fill_categorical_missing(df, categorical_columns, value="Unknown")

    df = drop_players_without_core_features(
        df,
        required_columns=[
            "position",
            "height_in_cm",
            "age",
            "avg_market_value",
        ],
    )

    print("Processed shape:", df.shape)
    print("\nProcessed dataset preview:")
    print(df.head())

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "player_profiles.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved processed dataset to: {output_path}")


if __name__ == "__main__":
    main()