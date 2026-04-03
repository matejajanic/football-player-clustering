from __future__ import annotations

import pandas as pd


def merge_player_core_data(
    players: pd.DataFrame,
    appearances: pd.DataFrame,
    player_valuations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge player-level information with appearance-level data.

    This function is intentionally conservative in the initial version.
    It assumes that:
    - players contains a player identifier
    - appearances contains match-by-match player records
    - player_valuations may optionally contain market value history
    """
    if "player_id" not in appearances.columns:
        raise ValueError("appearances must contain 'player_id' column")

    if "player_id" not in players.columns and "id" not in players.columns:
        raise ValueError("players must contain 'player_id' or 'id' column")

    players_copy = players.copy()
    if "id" in players_copy.columns and "player_id" not in players_copy.columns:
        players_copy = players_copy.rename(columns={"id": "player_id"})

    merged = appearances.merge(
        players_copy,
        on="player_id",
        how="left",
        suffixes=("", "_player"),
    )

    if player_valuations is not None and "player_id" in player_valuations.columns:
        valuation_agg = (
            player_valuations
            .groupby("player_id", as_index=False)
            .agg(
                avg_market_value=("market_value_in_eur", "mean"),
                max_market_value=("market_value_in_eur", "max"),
                valuation_records=("market_value_in_eur", "count"),
            )
        )

        merged = merged.merge(valuation_agg, on="player_id", how="left")

    return merged