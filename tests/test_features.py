import numpy as np
import pytest
from fb_models.data.features import (
    FEATURE_COLS,
    OUTCOME_COLS,
    PLAY_TYPES,
    build_feature_matrix,
)


def test_feature_cols_length():
    assert len(FEATURE_COLS) == 9


def test_feature_cols_names():
    expected = {
        "down", "ydstogo", "yardline_100", "score_differential", "qtr",
        "game_seconds_remaining", "posteam_timeouts_remaining",
        "defteam_timeouts_remaining", "goal_to_go",
    }
    assert set(FEATURE_COLS) == expected


def test_play_types_sorted():
    assert PLAY_TYPES == sorted(PLAY_TYPES)
    assert set(PLAY_TYPES) == {"run", "pass", "punt", "field_goal"}


def test_outcome_cols_contains_required():
    required = {
        "play_type", "yards_gained", "complete_pass", "incomplete_pass",
        "interception", "fumble", "fumble_lost", "play_duration",
    }
    assert required.issubset(set(OUTCOME_COLS))


def test_build_feature_matrix_shape(sample_plays):
    X = build_feature_matrix(sample_plays)
    assert X.shape == (len(sample_plays), 9)


def test_build_feature_matrix_dtype(sample_plays):
    X = build_feature_matrix(sample_plays)
    assert X.dtype == np.float64


def test_build_feature_matrix_column_order(sample_plays):
    X = build_feature_matrix(sample_plays)
    # First column should match first feature col value
    first_col = FEATURE_COLS[0]
    np.testing.assert_array_equal(X[:, 0], sample_plays[first_col].to_numpy(dtype=float))
