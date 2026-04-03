from src.data_loading.loader import load_all_csvs
from src.preprocessing.feature_engineering import build_player_dataset


def main():
    data = load_all_csvs()

    df = build_player_dataset(
        players=data["players"],
        appearances=data["appearances"],
        valuations=data["player_valuations"],
    )

    print(df.shape)
    print(df.head())


if __name__ == "__main__":
    main()