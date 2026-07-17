import pandas as pd

_SITUATION_COLS: list[str] = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "qtr",
    "game_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "goal_to_go",
    "drive",
    "drive_play_count",
    "drive_first_downs",
    "red_zone",
    "goal_line",
    "leading",
    "trailing",
    "one_score_game",
    "score_margin_bucket",
    "two_minute",
    "early_down",
    "passing_down",
    "short_yardage",
    "yardage_bucket",
]


def build_situation(pbp_df: pd.DataFrame) -> dict:
    df = pbp_df[
        (pbp_df["down"].notna())
        & (pbp_df["season_type"] == "REG")
    ]

    return (
        df.set_index(["game_id", "play_id"])[_SITUATION_COLS]
        .to_dict("index")
    )
