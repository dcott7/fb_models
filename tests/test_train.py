from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import joblib
import lightgbm as lgb
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from fb_models.training.train import main
from fb_models.data.features import PLAY_TYPES


def _write_minimal_parquet(tmp_path: Path) -> None:
    """Write a parquet with 15 games per play type (60 games total, 120 rows).

    Each game has exactly two plays of the same type in the same quarter.
    After loader processing, all 120 plays survive — 30 per play type.
    """
    rng = np.random.default_rng(0)
    play_type_cycle = ["run", "pass", "punt", "field_goal"] * 15  # 60 games

    rows = []
    for game_idx, pt in enumerate(play_type_cycle):
        for seq in (1, 2):
            rows.append({
                "game_id": f"game_{game_idx}",
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
    path = tmp_path / "data" / "pbp_2020.parquet"
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pandas(df), path)


def test_main_creates_all_artifacts(tmp_path):
    _write_minimal_parquet(tmp_path)
    artifacts_dir = tmp_path / "artifacts"

    main(
        data_dir=tmp_path / "data",
        artifacts_dir=artifacts_dir,
        min_season=2020,
        k=5,
    )

    assert (artifacts_dir / "play_type_classifier.joblib").exists()
    for pt in PLAY_TYPES:
        assert (artifacts_dir / f"knn_{pt}.joblib").exists()


def test_main_artifacts_are_loadable(tmp_path):
    _write_minimal_parquet(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    main(data_dir=tmp_path / "data", artifacts_dir=artifacts_dir, min_season=2020, k=5)

    clf = joblib.load(artifacts_dir / "play_type_classifier.joblib")
    assert isinstance(clf, lgb.LGBMClassifier)

    for pt in PLAY_TYPES:
        knn_index = joblib.load(artifacts_dir / f"knn_{pt}.joblib")
        assert isinstance(knn_index["nn"], NearestNeighbors)
        assert isinstance(knn_index["scaler"], StandardScaler)
        assert isinstance(knn_index["outcomes"], pd.DataFrame)
