import numpy as np
import pandas as pd

FEATURE_COLS: list[str] = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "qtr",
    "game_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "goal_to_go",
]

PLAY_TYPES: list[str] = ["field_goal", "pass", "punt", "run"]

OUTCOME_COLS: list[str] = [
    "play_type",
    "yards_gained",
    "complete_pass",
    "incomplete_pass",
    "interception",
    "fumble",
    "fumble_lost",
    "play_duration",
]


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLS].to_numpy(dtype=np.float64)
