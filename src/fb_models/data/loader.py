from pathlib import Path

import numpy as np
import pandas as pd

_NEEDED_COLS = [
    "game_id", "order_sequence", "season_type", "play_type", "qtr",
    "down", "ydstogo", "yardline_100", "score_differential",
    "game_seconds_remaining", "play_clock",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "goal_to_go",
    "yards_gained", "complete_pass", "incomplete_pass",
    "interception", "fumble", "fumble_lost",
]

_VALID_PLAY_TYPES = {"run", "pass", "punt", "field_goal"}
_PLAY_TYPE_REMAP = {"qb_kneel": "run", "qb_spike": "pass"}

_OUTPUT_COLS = [
    "play_type", "down", "ydstogo", "yardline_100", "score_differential",
    "qtr", "game_seconds_remaining", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "goal_to_go", "yards_gained",
    "complete_pass", "incomplete_pass", "interception", "fumble",
    "fumble_lost", "play_duration",
]


def load_plays(data_dir: Path, min_season: int = 2016) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("pbp_*.parquet")):
        season = int(path.stem.split("_")[1])
        if season < min_season:
            continue
        frames.append(pd.read_parquet(path, columns=_NEEDED_COLS))

    if not frames:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["season_type"].isin({"REG", "POST"})]
    df = df[df["down"].notna()]

    spike_mask = df["play_type"] == "qb_spike"
    df.loc[spike_mask, "complete_pass"] = 0
    df.loc[spike_mask, "incomplete_pass"] = 1

    df["play_type"] = df["play_type"].replace(_PLAY_TYPE_REMAP)
    df = df[df["play_type"].isin(_VALID_PLAY_TYPES)]

    df = df.sort_values(["game_id", "order_sequence"])
    df["play_duration"] = _compute_play_duration(df)
    df = df.dropna(subset=["play_duration"])

    return df[_OUTPUT_COLS].reset_index(drop=True)


def _compute_play_duration(df: pd.DataFrame) -> pd.Series:
    next_gsr = df.groupby("game_id")["game_seconds_remaining"].shift(-1)
    next_play_clock = df.groupby("game_id")["play_clock"].shift(-1)
    next_qtr = df.groupby("game_id")["qtr"].shift(-1)

    same_qtr = df["qtr"] == next_qtr
    valid = same_qtr & next_play_clock.notna()

    total_elapsed = df["game_seconds_remaining"] - next_gsr
    between_play = 40.0 - next_play_clock
    duration = (total_elapsed - between_play).where(valid)

    return duration.clip(1.0, 15.0)
