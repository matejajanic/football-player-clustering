from __future__ import annotations

from src.data_loading.loader import load_all_csvs


def main() -> None:
    dataframes = load_all_csvs()

    if not dataframes:
        print("No CSV files found in data/raw/.")
        print("Please place the dataset files inside data/raw/ and try again.")
        return

    print(f"Loaded {len(dataframes)} CSV files.\n")

    for name, df in dataframes.items():
        print("=" * 100)
        print(f"TABLE: {name}")
        print(f"SHAPE: {df.shape}")
        print("COLUMNS:")
        for col in df.columns:
            print(f"  - {col}")

        print("\nMISSING VALUES (top 15):")
        missing = df.isna().sum().sort_values(ascending=False).head(15)
        for col, value in missing.items():
            print(f"  - {col}: {value}")

        print("\nHEAD:")
        print(df.head(3).to_string())
        print()

    print("=" * 100)
    print("Inspection complete.")


if __name__ == "__main__":
    main()