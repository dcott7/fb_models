import pandas as pd

PLAY_TYPE_NUMERIC_COLS: list[str] = [
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

PLAY_TYPE_CATEGORICAL_COLS: list[str] = [
    "score_margin_bucket",
    "yardage_bucket",
    "previous_play_type",
    "roof",
]

PLAY_TYPE_FEATURE_COLS: list[str] = PLAY_TYPE_NUMERIC_COLS + PLAY_TYPE_CATEGORICAL_COLS


def select_play_type_features(plays_df: pd.DataFrame) -> pd.DataFrame:
    df = plays_df.copy()

    for col in PLAY_TYPE_CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    return df[PLAY_TYPE_FEATURE_COLS + ["play_type"]]
