from collections import defaultdict

import pandas as pd

from .utils import convert_default_dict


def build_participation(participation_df: pd.DataFrame) -> dict:
    participation: dict = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: {
                    "snap_count": 0,
                    "offense_snaps": 0,
                    "defense_snaps": 0,
                    "formation_counts": defaultdict(int),
                    "offense_personnel_counts": defaultdict(int),
                    "defense_personnel_counts": defaultdict(int),
                }
            )
        )
    )

    df = participation_df[participation_df["possession_team"].notna()]
    df = df[df["n_offense"] == 11]
    df = df[df["n_defense"] == 11]

    for _, play in df.iterrows():
        offense_team = play["possession_team"]
        defense_team = play["defending_team"]
        season = play["season"]

        formation = play["offense_formation"]
        offense_personnel = play["offense_personnel"]
        defense_personnel = play["defense_personnel"]

        # Offensive players
        for player_id in play["offense_players"]:
            if pd.isna(player_id):
                continue

            player = participation[offense_team][season][player_id]

            player["snap_count"] += 1
            player["offense_snaps"] += 1

            if formation:
                player["formation_counts"][formation] += 1

            if offense_personnel:
                player["offense_personnel_counts"][offense_personnel] += 1

        # Defensive players
        for player_id in play["defense_players"]:
            if pd.isna(player_id):
                continue

            player = participation[defense_team][season][player_id]

            player["snap_count"] += 1
            player["defense_snaps"] += 1

            if defense_personnel:
                player["defense_personnel_counts"][defense_personnel] += 1

    # Convert nested defaultdicts into normal dicts
    return convert_default_dict(participation)
