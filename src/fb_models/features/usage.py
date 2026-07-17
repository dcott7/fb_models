"""
Output dict from build_usage() should look something like this
{
    "player_id": {
        2024: {
            "rushing": {
                "overall": {
                    "carries": 100,
                    "opportunities": 200,
                    "share": 0.50
                },
                "personnel": {...},
                "formation": {...},
                "field_position": {...}
            },
            "receiving": {
                "overall": {
                    "targets": 80,
                    "opportunities": 500,
                    "share": 0.16
                },
                ...
            }
        }
    }
}
"""


from collections import defaultdict

import pandas as pd

from .utils import convert_default_dict


_PARTICIPATION_COLS = [
    "nflverse_game_id",
    "play_id",
    "offense_players",
    "offense_formation",
    "offense_personnel",
]


def _prepare_usage_data(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame
) -> pd.DataFrame:

    pbp = pbp_df[
        (pbp_df["down"].notna())
        & (pbp_df["season_type"] == "REG")
    ]

    part = participation_df[_PARTICIPATION_COLS]

    df = part.merge(
        pbp,
        left_on=["nflverse_game_id", "play_id"],
        right_on=["game_id", "play_id"],
        how="inner"
    )

    df = df.explode("offense_players")

    df.rename(columns={"offense_players": "player_id"}, inplace=True)

    return df


def _create_usage_template(stat_name: str) -> dict:
    return {
        "overall": {
            stat_name: 0,
            "opportunities": 0,
        },
        "personnel": defaultdict(
            lambda: {
                stat_name: 0,
                "opportunities": 0,
            }
        ),
        "formation": defaultdict(
            lambda: {
                stat_name: 0,
                "opportunities": 0,
            }
        ),
        "field_position": defaultdict(
            lambda: {
                stat_name: 0,
                "opportunities": 0,
            }
        )
    }


def _calculate_usage_shares(usage, numerator: str):
    for _, seasons in usage.items():
        for _, data in seasons.items():

            overall = data["overall"]

            overall["share"] = (
                overall[numerator] / overall["opportunities"]
                if overall["opportunities"]
                else 0
            )

            for category in ["personnel", "formation", "field_position"]:
                for _, value in data[category].items():
                    value["share"] = (
                        value[numerator] / value["opportunities"]
                        if value["opportunities"]
                        else 0
                    )

    return usage


def _build_rushing_usage(df: pd.DataFrame):

    usage: defaultdict = defaultdict(
        lambda: defaultdict(
            lambda: _create_usage_template("carries")
        )
    )

    rushes = df[
        (df["rush"] == True)
        & (df["rush_attempt"] == True)
    ]

    for _, play in rushes.iterrows():

        player_id = play["player_id"]
        season = play["season"]

        entry = usage[player_id][season]

        personnel = play["offense_personnel"]
        formation = play["offense_formation"]
        field_position = play["field_position"]

        entry["overall"]["opportunities"] += 1

        if pd.notna(personnel):
            entry["personnel"][personnel]["opportunities"] += 1

        if pd.notna(formation):
            entry["formation"][formation]["opportunities"] += 1

        entry["field_position"][field_position]["opportunities"] += 1

        if play["rusher_player_id"] == player_id:

            entry["overall"]["carries"] += 1

            if pd.notna(personnel):
                entry["personnel"][personnel]["carries"] += 1

            if pd.notna(formation):
                entry["formation"][formation]["carries"] += 1

            entry["field_position"][field_position]["carries"] += 1

    return _calculate_usage_shares(usage, "carries")


def _build_receiving_usage(df: pd.DataFrame):

    usage: defaultdict = defaultdict(
        lambda: defaultdict(
            lambda: _create_usage_template("targets")
        )
    )

    passes = df[(df["pass"] == True) & (df["pass_attempt"] == True)]

    for _, play in passes.iterrows():

        player_id = play["player_id"]
        season = play["season"]

        entry = usage[player_id][season]

        personnel = play["offense_personnel"]
        formation = play["offense_formation"]
        field_position = play["field_position"]

        entry["overall"]["opportunities"] += 1

        if pd.notna(personnel):
            entry["personnel"][personnel]["opportunities"] += 1

        if pd.notna(formation):
            entry["formation"][formation]["opportunities"] += 1

        entry["field_position"][field_position]["opportunities"] += 1

        # Actual target
        if play["receiver_player_id"] == player_id:

            entry["overall"]["targets"] += 1

            if pd.notna(personnel):
                entry["personnel"][personnel]["targets"] += 1

            if pd.notna(formation):
                entry["formation"][formation]["targets"] += 1

            entry["field_position"][field_position]["targets"] += 1

    return _calculate_usage_shares(usage, "targets")


def build_usage(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame,
) -> dict:

    df = _prepare_usage_data(pbp_df, participation_df)

    rushing = _build_rushing_usage(df)
    receiving = _build_receiving_usage(df)

    usage: defaultdict = defaultdict(lambda: defaultdict(dict))
    players = set(rushing.keys()) | set(receiving.keys())

    for player_id in players:

        seasons = (
            set(rushing[player_id].keys())
            | set(receiving[player_id].keys())
        )

        for season in seasons:

            usage[player_id][season] = {
                "rushing": rushing[player_id].get(season, {},),
                "receiving": receiving[player_id].get(season, {},),
            }

    return convert_default_dict(usage)