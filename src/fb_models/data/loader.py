from pathlib import Path

import pandas as pd

_NEEDED_COLS = [
    "game_id", "order_sequence", "season_type", "play_type", "qtr",
    "down", "ydstogo", "yardline_100", "score_differential",
    "game_seconds_remaining",
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
    "fumble_lost",
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

    return df[_OUTPUT_COLS].reset_index(drop=True)
