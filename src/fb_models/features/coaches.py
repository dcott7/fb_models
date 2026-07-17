from collections import defaultdict

import pandas as pd

from .utils import convert_default_dict

_BLITZ_PASS_RUSHERS = 5

_PARTICIPATION_COLS = [
    "nflverse_game_id",
    "play_id",
    "defense_man_zone_type",
    "was_pressure",
    "number_of_pass_rushers",
]


def _prepare_coach_data(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame,
) -> pd.DataFrame:
    pbp = pbp_df[
        (pbp_df["down"].notna())
        & (pbp_df["season_type"] == "REG")
    ]

    part = participation_df[_PARTICIPATION_COLS]

    return pbp.merge(
        part,
        left_on=["game_id", "play_id"],
        right_on=["nflverse_game_id", "play_id"],
        how="left",
    )


def _rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def _build_offensive_tendencies(df: pd.DataFrame) -> dict:
    offense: defaultdict = defaultdict(dict)

    for (coach, season), plays in df.groupby(["offense_coach", "season"]):
        if not isinstance(coach, str):
            continue

        snaps = plays[plays["is_go_for_it"]]
        early_down = snaps[snaps["down"].isin([1, 2])]
        neutral = snaps[snaps["neutral_script"]]
        third_down = snaps[snaps["down"] == 3]
        red_zone = snaps[snaps["red_zone"]]
        goal_line = snaps[snaps["goal_line"]]
        two_minute = snaps[snaps["two_minute"]]
        four_minute = snaps[snaps["four_minute"]]

        fourth_down = plays[plays["down"] == 4]
        fourth_down_own = fourth_down[fourth_down["yardline_100"] > 50]
        fourth_down_opp = fourth_down[fourth_down["yardline_100"] <= 50]

        offense[coach][season] = {
            "plays": len(plays),
            "pass_rate": _rate(snaps["is_pass"]),
            "early_down_pass_rate": _rate(early_down["is_pass"]),
            "neutral_pass_rate": _rate(neutral["is_pass"]),
            "shotgun_rate": _rate(snaps["shotgun"]),
            "no_huddle_rate": _rate(snaps["no_huddle"]),
            "situational": {
                "red_zone": {
                    "pass_rate": _rate(red_zone["is_pass"]),
                    "rush_rate": _rate(red_zone["is_rush"]),
                },
                "goal_line": {
                    "pass_rate": _rate(goal_line["is_pass"]),
                    "rush_rate": _rate(goal_line["is_rush"]),
                },
                "third_down": {
                    "pass_rate": _rate(third_down["is_pass"]),
                    "shotgun_rate": _rate(third_down["shotgun"]),
                },
                "two_minute": {
                    "pass_rate": _rate(two_minute["is_pass"]),
                    "no_huddle_rate": _rate(two_minute["no_huddle"]),
                },
                "four_minute": {
                    "rush_rate": _rate(four_minute["is_rush"]),
                },
            },
            "fourth_down": {
                "go_for_it_rate": _rate(fourth_down["is_go_for_it"]),
                "go_for_it_rate_own_territory": _rate(fourth_down_own["is_go_for_it"]),
                "go_for_it_rate_opponent_territory": _rate(fourth_down_opp["is_go_for_it"]),
            },
        }

    return offense


def _build_defensive_tendencies(df: pd.DataFrame) -> dict:
    defense: defaultdict = defaultdict(dict)

    for (coach, season), plays in df.groupby(["defense_coach", "season"]):
        if not isinstance(coach, str):
            continue

        snaps = plays[plays["is_go_for_it"]]
        early_down = snaps[snaps["down"].isin([1, 2])]
        neutral = snaps[snaps["neutral_script"]]
        third_down = snaps[snaps["down"] == 3]
        red_zone = snaps[snaps["red_zone"]]
        goal_line = snaps[snaps["goal_line"]]
        two_minute = snaps[snaps["two_minute"]]

        pass_rush_known = plays[plays["number_of_pass_rushers"].notna()]
        pressure_known = plays[plays["was_pressure"].notna()]
        coverage_known = plays[plays["defense_man_zone_type"].notna()]

        fourth_down_faced = plays[(plays["down"] == 4) & plays["is_go_for_it"]]

        defense[coach][season] = {
            "plays": len(plays),
            "pass_rate_allowed": _rate(snaps["is_pass"]),
            "early_down_pass_rate_allowed": _rate(early_down["is_pass"]),
            "neutral_pass_rate_allowed": _rate(neutral["is_pass"]),
            "blitz_rate": _rate(
                pass_rush_known["number_of_pass_rushers"] >= _BLITZ_PASS_RUSHERS
            ),
            "pressure_rate": _rate(pressure_known["was_pressure"].astype(bool)),
            "man_rate": _rate(
                coverage_known["defense_man_zone_type"] == "MAN_COVERAGE"
            ),
            "situational": {
                "red_zone": {"pass_rate_allowed": _rate(red_zone["is_pass"])},
                "goal_line": {"pass_rate_allowed": _rate(goal_line["is_pass"])},
                "third_down": {"pass_rate_allowed": _rate(third_down["is_pass"])},
                "two_minute": {"pass_rate_allowed": _rate(two_minute["is_pass"])},
            },
            "fourth_down_stop_rate": _rate(fourth_down_faced["fourth_down_failed"] == 1),
        }

    return defense


def build_coaches(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame,
) -> dict:
    # TODO - Try and get play caller data so that we can attribute the playcalls to the
    # proper play caller
    df = _prepare_coach_data(pbp_df, participation_df)

    offense = _build_offensive_tendencies(df)
    defense = _build_defensive_tendencies(df)

    coaches: defaultdict = defaultdict(lambda: defaultdict(dict))
    coach_names = set(offense.keys()) | set(defense.keys())

    for coach in coach_names:
        seasons = set(offense.get(coach, {}).keys()) | set(defense.get(coach, {}).keys())

        for season in seasons:
            coaches[coach][season] = {
                "offense": offense.get(coach, {}).get(season, {}),
                "defense": defense.get(coach, {}).get(season, {}),
            }

    return convert_default_dict(coaches)
