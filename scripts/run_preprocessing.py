from __future__ import annotations

from src.config import PROCESSED_DATA_DIR
from src.data_loading.loader import load_all_csvs
from src.preprocessing.feature_engineering import build_player_dataset
from src.preprocessing.filtering import (
    filter_players_by_matches,
    filter_players_by_minutes,
    remove_zero_minute_players,
)
from src.preprocessing.cleaning import (
    cast_numeric_columns,
    fill_categorical_missing,
    fill_numeric_missing,
    fill_numeric_missing_with_median,
    remove_duplicates,
    drop_rows_with_missing,
    one_hot_encode,
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

    # 1) kategorijske kolone
    df = fill_categorical_missing(df, categorical_columns, value="Unknown")

    # 2) kolone gde je 0 smislen
    zero_fill_columns = [
        "international_caps",
        "international_goals",
    ]
    df = fill_numeric_missing(df, zero_fill_columns, value=0.0)

    # 3) kolone gde je bolja mediana nego 0
    median_fill_columns = [
        "market_value_in_eur",
        "highest_market_value_in_eur",
        "avg_market_value",
        "max_market_value",
        "valuation_records",
    ]
    df = fill_numeric_missing_with_median(df, median_fill_columns)

    # 4) height i age NE punimo nulom -> izbacujemo ako fale
    df = drop_rows_with_missing(df, ["height_in_cm", "age"])

    # one-hot encoding
    df = one_hot_encode(df, ["position", "foot"])

    print("Processed shape:", df.shape)
    print("\nProcessed dataset preview:")
    print(df.head())

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "player_profiles.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved processed dataset to: {output_path}")


if __name__ == "__main__":
    main()