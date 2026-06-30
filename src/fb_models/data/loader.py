from pathlib import Path

import pandas as pd

_NEEDED_COLS: list[str] = [
    "game_id", "order_sequence", "season_type", "play_type", "qtr",
    "down", "ydstogo", "yardline_100", "score_differential",
    "game_seconds_remaining",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "goal_to_go",
    "yards_gained", "complete_pass", "incomplete_pass",
    "interception", "fumble", "fumble_lost",
]

_VALID_PLAY_TYPES: set[str] = {"run", "pass", "punt", "field_goal"}
_PLAY_TYPE_REMAP: dict[str, str] = {"qb_kneel": "run", "qb_spike": "pass"}

_OUTPUT_COLS: list[str] = [
    "play_type", "down", "ydstogo", "yardline_100", "score_differential",
    "qtr", "game_seconds_remaining", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "goal_to_go", "yards_gained",
    "complete_pass", "incomplete_pass", "interception", "fumble",
    "fumble_lost",
]


def load_plays(
    data_dir: Path, min_season: int = 2016, max_season: int | None = None
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(data_dir.glob("pbp_*.parquet")):
        season = int(path.stem.split("_")[1])
        if season < min_season:
            continue
        if max_season is not None and season > max_season:
            continue
        frames.append(pd.read_parquet(path, columns=_NEEDED_COLS))

    if not frames:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = pd.concat(frames, ignore_index=True)
    # only want REG/POST season as teams don't care about preseason
    df = df[df["season_type"].isin(["REG", "POST"])]
    df = df[df["down"].notna()]

    spike_mask = df["play_type"] == "qb_spike"
    df.loc[spike_mask, "complete_pass"] = 0
    df.loc[spike_mask, "incomplete_pass"] = 1

    df["play_type"] = df["play_type"].replace(_PLAY_TYPE_REMAP)
    df = df[df["play_type"].isin(list(_VALID_PLAY_TYPES))]

    df = df.sort_values(["game_id", "order_sequence"])

    return df[_OUTPUT_COLS].reset_index(drop=True)
