import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from fb_models.models.knn import SECONDS_ELAPSED_RANGES, build_knn_index, query_knn


def test_build_knn_index_returns_correct_types(sample_plays):
    nn, scaler, subset = build_knn_index(sample_plays, "run", k=5)
    assert isinstance(nn, NearestNeighbors)
    assert isinstance(scaler, StandardScaler)
    assert len(subset) == (sample_plays["play_type"] == "run").sum()


def test_build_knn_index_subset_has_outcome_cols(sample_plays):
    _, _, subset = build_knn_index(sample_plays, "pass", k=5)
    required = {"play_type", "yards_gained", "complete_pass", "incomplete_pass",
                "interception", "fumble", "fumble_lost"}
    assert required.issubset(set(subset.columns))


def test_query_knn_returns_all_output_fields(sample_plays):
    knn_index = build_knn_index(sample_plays, "run", k=5)
    game_state = np.array([1, 10, 50, 0, 1, 1800, 3, 3, 0], dtype=np.float64)
    rng = np.random.default_rng(0)
    result = query_knn(knn_index, game_state, rng)
    expected_keys = {
        "play_type", "yards_gained", "is_complete", "is_incomplete",
        "is_intercepted", "is_fumble", "is_turnover", "seconds_elapsed",
    }
    assert set(result.keys()) == expected_keys


def test_query_knn_play_type_matches_index(sample_plays):
    knn_index = build_knn_index(sample_plays, "punt", k=5)
    game_state = np.array([4, 10, 60, 0, 2, 1800, 3, 3, 0], dtype=np.float64)
    rng = np.random.default_rng(1)
    result = query_knn(knn_index, game_state, rng)
    assert result["play_type"] == "punt"


def test_query_knn_is_turnover_is_true_when_interception(sample_plays):
    # Build index from plays where all are interceptions
    df = sample_plays.copy()
    pass_plays = df[df["play_type"] == "pass"].copy()
    pass_plays["interception"] = 1
    pass_plays["fumble_lost"] = 0
    knn_index = build_knn_index(pass_plays, "pass", k=5)
    game_state = np.array([3, 8, 40, -7, 3, 600, 2, 3, 0], dtype=np.float64)
    rng = np.random.default_rng(2)
    result = query_knn(knn_index, game_state, rng)
    assert result["is_turnover"] is True


def test_query_knn_seconds_elapsed_within_play_type_range(sample_plays):
    knn_index = build_knn_index(sample_plays, "punt", k=5)
    game_state = np.array([4, 10, 60, 0, 2, 1800, 3, 3, 0], dtype=np.float64)
    rng = np.random.default_rng(3)
    result = query_knn(knn_index, game_state, rng)
    low, high = SECONDS_ELAPSED_RANGES["punt"]
    assert isinstance(result["seconds_elapsed"], float)
    assert low <= result["seconds_elapsed"] <= high


def test_query_knn_deterministic_with_same_seed(sample_plays):
    knn_index = build_knn_index(sample_plays, "run", k=10)
    game_state = np.array([2, 5, 30, 7, 2, 900, 2, 3, 0], dtype=np.float64)
    result1 = query_knn(knn_index, game_state, np.random.default_rng(99))
    result2 = query_knn(knn_index, game_state, np.random.default_rng(99))
    assert result1 == result2
