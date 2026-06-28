from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from fb_models.data.loader import load_plays


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    pq.write_table(pa.Table.from_pandas(df), path)


def _base_row() -> dict:
    return {
        "game_id": "2024_01_KC_SF",
        "order_sequence": 1,
        "season_type": "REG",
        "play_type": "run",
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 50,
        "score_differential": 0,
        "qtr": 1,
        "game_seconds_remaining": 3600,
        "play_clock": 15,
        "posteam_timeouts_remaining": 3,
        "defteam_timeouts_remaining": 3,
        "goal_to_go": 0,
        "yards_gained": 5,
        "complete_pass": 0,
        "incomplete_pass": 0,
        "interception": 0,
        "fumble": 0,
        "fumble_lost": 0,
    }


def _make_consecutive_plays(tmp_path: Path, season: int) -> Path:
    """Two plays in the same game/quarter so play_duration can be computed."""
    row1 = {**_base_row(), "game_id": f"game_{season}", "order_sequence": 1, "game_seconds_remaining": 3600, "play_clock": 15}
    row2 = {**_base_row(), "game_id": f"game_{season}", "order_sequence": 2, "game_seconds_remaining": 3540, "play_clock": 25}
    df = pd.DataFrame([row1, row2])
    path = tmp_path / f"pbp_{season}.parquet"
    _write_parquet(path, df)
    return path


def test_load_plays_returns_dataframe(tmp_path):
    _make_consecutive_plays(tmp_path, 2020)
    df = load_plays(tmp_path, min_season=2020)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_plays_filters_by_min_season(tmp_path):
    _make_consecutive_plays(tmp_path, 2015)
    _make_consecutive_plays(tmp_path, 2020)

    df_2015_included = load_plays(tmp_path, min_season=2015)
    df_2015_excluded = load_plays(tmp_path, min_season=2016)

    # 2020 plays appear in both loads; 2015 plays only in the first
    assert len(df_2015_included) > len(df_2015_excluded)
    assert len(df_2015_excluded) > 0


def test_load_plays_excludes_preseason(tmp_path):
    row1 = {**_base_row(), "season_type": "PRE", "order_sequence": 1, "game_seconds_remaining": 3600, "play_clock": 15}
    row2 = {**_base_row(), "season_type": "REG", "order_sequence": 2, "game_seconds_remaining": 3540, "play_clock": 25}
    path = tmp_path / "pbp_2020.parquet"
    _write_parquet(path, pd.DataFrame([row1, row2]))
    df = load_plays(tmp_path, min_season=2020)
    # Both rows needed for play_duration but PRE row filtered before compute
    # Result should have 0 rows (no consecutive REG plays in same game/qtr)
    assert len(df) == 0


def test_load_plays_remaps_qb_kneel_to_run(tmp_path):
    row1 = {**_base_row(), "play_type": "qb_kneel", "order_sequence": 1, "game_seconds_remaining": 3600, "play_clock": 15}
    row2 = {**_base_row(), "play_type": "qb_kneel", "order_sequence": 2, "game_seconds_remaining": 3540, "play_clock": 25}
    path = tmp_path / "pbp_2020.parquet"
    _write_parquet(path, pd.DataFrame([row1, row2]))
    df = load_plays(tmp_path, min_season=2020)
    assert (df["play_type"] == "run").all()


def test_load_plays_remaps_qb_spike_to_pass_incomplete(tmp_path):
    row1 = {**_base_row(), "play_type": "qb_spike", "order_sequence": 1, "game_seconds_remaining": 3600, "play_clock": 15, "complete_pass": 0, "incomplete_pass": 0}
    row2 = {**_base_row(), "play_type": "qb_spike", "order_sequence": 2, "game_seconds_remaining": 3540, "play_clock": 25, "complete_pass": 0, "incomplete_pass": 0}
    path = tmp_path / "pbp_2020.parquet"
    _write_parquet(path, pd.DataFrame([row1, row2]))
    df = load_plays(tmp_path, min_season=2020)
    assert (df["play_type"] == "pass").all()
    assert (df["incomplete_pass"] == 1).all()
    assert (df["complete_pass"] == 0).all()


def test_load_plays_computes_play_duration(tmp_path):
    # play1: gsr=3600, play_clock at next snap=15 → duration=(3600-3540)-(40-15)=60-25=35... clipped to 15
    # With play_clock=25 at next snap: duration=(3600-3540)-(40-25)=60-15=45... clipped to 15
    row1 = {**_base_row(), "order_sequence": 1, "game_seconds_remaining": 3600, "play_clock": 15}
    row2 = {**_base_row(), "order_sequence": 2, "game_seconds_remaining": 3560, "play_clock": 35}
    path = tmp_path / "pbp_2020.parquet"
    _write_parquet(path, pd.DataFrame([row1, row2]))
    df = load_plays(tmp_path, min_season=2020)
    # duration = (3600-3560) - (40-35) = 40 - 5 = 35, clipped to 15
    assert len(df) == 1
    assert df["play_duration"].iloc[0] == pytest.approx(15.0)


def test_load_plays_output_columns(tmp_path):
    _make_consecutive_plays(tmp_path, 2020)
    df = load_plays(tmp_path, min_season=2020)
    expected = {
        "play_type", "down", "ydstogo", "yardline_100", "score_differential",
        "qtr", "game_seconds_remaining", "posteam_timeouts_remaining",
        "defteam_timeouts_remaining", "goal_to_go", "yards_gained",
        "complete_pass", "incomplete_pass", "interception", "fumble",
        "fumble_lost", "play_duration",
    }
    assert set(df.columns) == expected
