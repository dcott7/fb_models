from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fb_models.models.classifier import train_classifier
from fb_models.models.knn import build_knn_index
from fb_models.training.evaluate import (
    discover_seasons,
    evaluate_classifier,
    evaluate_knn_outcomes,
    main,
    parse_args,
)


def _write_season_parquet(tmp_path: Path, season: int, n_games_per_type: int = 15) -> None:
    """15 games per play type (60 games, 120 rows) for a single season."""
    rng = np.random.default_rng(season)
    play_type_cycle = ["run", "pass", "punt", "field_goal"] * n_games_per_type

    rows = []
    for game_idx, pt in enumerate(play_type_cycle):
        for seq in (1, 2):
            rows.append({
                "game_id": f"game_{season}_{game_idx}",
                "order_sequence": seq,
                "season_type": "REG",
                "play_type": pt,
                "down": int(rng.integers(1, 5)),
                "ydstogo": int(rng.integers(1, 21)),
                "yardline_100": int(rng.integers(1, 100)),
                "score_differential": int(rng.integers(-28, 29)),
                "qtr": 1,
                "game_seconds_remaining": 3600 if seq == 1 else 3560,
                "posteam_timeouts_remaining": 3,
                "defteam_timeouts_remaining": 3,
                "goal_to_go": 0,
                "yards_gained": int(rng.integers(-5, 20)),
                "complete_pass": 1 if pt == "pass" else 0,
                "incomplete_pass": 0,
                "interception": 0,
                "fumble": 0,
                "fumble_lost": 0,
            })

    df = pd.DataFrame(rows)
    path = tmp_path / "data" / f"pbp_{season}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), path)


def test_discover_seasons_returns_sorted_seasons(tmp_path):
    _write_season_parquet(tmp_path, 2021)
    _write_season_parquet(tmp_path, 2019)
    _write_season_parquet(tmp_path, 2020)
    assert discover_seasons(tmp_path / "data") == [2019, 2020, 2021]


def test_discover_seasons_empty_directory(tmp_path):
    (tmp_path / "data").mkdir()
    assert discover_seasons(tmp_path / "data") == []


def test_evaluate_classifier_returns_expected_keys(sample_plays):
    clf = train_classifier(sample_plays)
    metrics = evaluate_classifier(clf, sample_plays)
    assert set(metrics.keys()) == {
        "accuracy", "log_loss", "classification_report", "confusion_matrix", "labels",
    }
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert metrics["confusion_matrix"].shape == (len(metrics["labels"]), len(metrics["labels"]))


def test_evaluate_knn_outcomes_returns_expected_keys(sample_plays):
    knn_index = build_knn_index(sample_plays, "run", k=5)
    test_subset = sample_plays[sample_plays["play_type"] == "run"]
    rng = np.random.default_rng(0)
    comparison = evaluate_knn_outcomes(knn_index, test_subset, rng, n_samples=10)

    assert comparison["n_real"] == comparison["n_sampled"]
    assert comparison["n_real"] == min(10, len(test_subset))
    for key in ("yards_gained_mean", "yards_gained_std", "turnover_rate", "complete_rate"):
        assert isinstance(comparison[f"{key}_real"], float)
        assert isinstance(comparison[f"{key}_sampled"], float)


def test_evaluate_knn_outcomes_caps_at_n_samples(sample_plays):
    knn_index = build_knn_index(sample_plays, "run", k=5)
    test_subset = sample_plays[sample_plays["play_type"] == "run"]
    rng = np.random.default_rng(0)
    comparison = evaluate_knn_outcomes(knn_index, test_subset, rng, n_samples=3)
    assert comparison["n_real"] == 3
    assert comparison["n_sampled"] == 3


def test_main_runs_end_to_end_and_prints_report(tmp_path, capsys):
    _write_season_parquet(tmp_path, 2019)
    _write_season_parquet(tmp_path, 2020)

    main(
        data_dir=tmp_path / "data",
        min_season=2019,
        test_season=2020,
        k=5,
        n_eval_samples=10,
        seed=0,
    )

    out = capsys.readouterr().out
    assert "Play Type Classifier" in out
    assert "K-NN Outcome Sampler" in out
    assert "run" in out
    assert "pass" in out


def test_main_defaults_test_season_to_latest_available(tmp_path, capsys):
    _write_season_parquet(tmp_path, 2019)
    _write_season_parquet(tmp_path, 2020)

    main(data_dir=tmp_path / "data", min_season=2019, k=5, n_eval_samples=10, seed=0)

    out = capsys.readouterr().out
    assert "test season 2020" in out


def test_main_raises_when_test_season_not_after_min_season(tmp_path):
    _write_season_parquet(tmp_path, 2020)
    with pytest.raises(ValueError, match="test_season"):
        main(data_dir=tmp_path / "data", min_season=2020, test_season=2020)


def test_main_raises_when_no_data_files(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(ValueError, match="No pbp_"):
        main(data_dir=tmp_path / "data")


def test_parse_args_defaults():
    args = parse_args([])
    assert args.data_dir == Path("data")
    assert args.min_season == 2016
    assert args.test_season is None
    assert args.k == 50
    assert args.n_eval_samples == 2000
    assert args.seed == 0


def test_parse_args_overrides():
    args = parse_args([
        "--data-dir", "other_data",
        "--min-season", "2018",
        "--test-season", "2024",
        "--k", "100",
        "--n-eval-samples", "500",
        "--seed", "7",
    ])
    assert args.data_dir == Path("other_data")
    assert args.min_season == 2018
    assert args.test_season == 2024
    assert args.k == 100
    assert args.n_eval_samples == 500
    assert args.seed == 7


def test_cli_invocation_with_overrides(tmp_path, capsys):
    _write_season_parquet(tmp_path, 2019)
    _write_season_parquet(tmp_path, 2020)

    args = parse_args([
        "--data-dir", str(tmp_path / "data"),
        "--min-season", "2019",
        "--test-season", "2020",
        "--k", "5",
        "--n-eval-samples", "10",
    ])
    main(**vars(args))

    out = capsys.readouterr().out
    assert "test season 2020" in out
