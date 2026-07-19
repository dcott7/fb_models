import pandas as pd

_VALID_PLAY_TYPES = {"run", "pass"}

FORMATION_NUMERIC_COLS: list[str] = [
    "down",
    "ydstogo",
    "yardline_100",
    "goal_to_go",
    "qtr",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "red_zone",
    "goal_line",
    "two_minute",
    "four_minute",
    "passing_down",
    "short_yardage",
    "neutral_script",
    "previous_yards_gained",
    "previous_first_down",
    "previous_turnover",
    "off_pass_rate_hist",
    "off_early_down_pass_rate_hist",
    "off_red_zone_pass_rate_hist",
    "off_goal_line_pass_rate_hist",
    "off_neutral_pass_rate_hist",
    "off_fourth_down_go_for_it_rate_hist",
    "off_shotgun_rate_hist",
    "off_under_center_rate_hist",
    "off_pistol_rate_hist",
    "off_no_huddle_rate_hist",
    "def_pass_rate_allowed_hist",
    "def_blitz_rate_hist",
    "def_pressure_rate_hist",
    "def_man_rate_hist",
    "posteam_spread_line",
    "total_line",
    "div_game",
    "posteam_rest",
    "defteam_rest",
    "temp",
    "wind",
]

FORMATION_CATEGORICAL_COLS: list[str] = [
    "score_margin_bucket",
    "yardage_bucket",
    "previous_play_type",
    "roof",
    "play_type",
]

FORMATION_FEATURE_COLS: list[str] = FORMATION_NUMERIC_COLS + FORMATION_CATEGORICAL_COLS


def select_formation_features(plays_df: pd.DataFrame) -> pd.DataFrame:
    # offense_formation is basically never populated for punt/field_goal
    # (confirmed <0.05% on real data) -- it's a run/pass-only concept, and
    # play_type is already known by the time formation is decided.
    df = plays_df[plays_df["play_type"].isin(_VALID_PLAY_TYPES)]
    df = df[df["offense_formation"].notna()].copy()

    for col in FORMATION_CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    return df[FORMATION_FEATURE_COLS + ["offense_formation"]]
