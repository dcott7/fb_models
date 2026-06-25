# Play Simulation Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and serialize a LightGBM play-type classifier and four K-NN outcome indexes from nflverse play-by-play data, producing artifacts consumable by an external NFL game simulation engine.

**Architecture:** A two-stage pipeline — a GBM classifier estimates `P(play_type | game_state)` over `{run, pass, punt, field_goal}`, then a per-play-type K-NN index samples a real historical play to produce all output fields (yards_gained, is_complete, is_intercepted, is_fumble, is_turnover, seconds_elapsed). Outputs are real plays, so joint correlations are preserved without explicit modeling.

**Tech Stack:** Python 3.11+, LightGBM, scikit-learn (NearestNeighbors, StandardScaler), pandas, pyarrow, joblib, pytest, uv

## Global Constraints

- Python >= 3.11
- `MIN_SEASON = 2016` default (parameterized) — modern NFL era filter
- `K = 50` default neighbors (parameterized)
- Play types: `run`, `pass`, `punt`, `field_goal` only — `qb_kneel` remapped to `run`, `qb_spike` remapped to `pass` (with `incomplete_pass=1`)
- Artifact paths: `artifacts/play_type_classifier.joblib`, `artifacts/knn_{play_type}.joblib`
- Data directory: `data/` — parquet files named `pbp_{season}.parquet`
- No simulation layer — this project produces artifacts only
- TDD: write failing test before any implementation

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Add dependencies, build system, pytest config |
| `src/fb_models/__init__.py` | Package marker |
| `src/fb_models/data/__init__.py` | Package marker |
| `src/fb_models/data/loader.py` | Load parquets, filter, remap play types, compute `play_duration` |
| `src/fb_models/data/features.py` | `FEATURE_COLS`, `PLAY_TYPES`, `OUTCOME_COLS` constants; `build_feature_matrix()` |
| `src/fb_models/models/__init__.py` | Package marker |
| `src/fb_models/models/classifier.py` | `train_classifier()`, `predict_play_type_probs()` |
| `src/fb_models/models/knn.py` | `build_knn_index()`, `query_knn()` |
| `src/fb_models/training/__init__.py` | Package marker |
| `src/fb_models/training/train.py` | `main()` — end-to-end load → train → serialize |
| `tests/conftest.py` | Shared `sample_plays` fixture |
| `tests/test_loader.py` | Tests for `load_plays()` |
| `tests/test_features.py` | Tests for `build_feature_matrix()` |
| `tests/test_classifier.py` | Tests for `train_classifier()`, `predict_play_type_probs()` |
| `tests/test_knn.py` | Tests for `build_knn_index()`, `query_knn()` |
| `tests/test_train.py` | Integration test: `main()` produces expected artifact files |

---

## Task 1: Project Setup

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `src/fb_models/__init__.py`
- Create: `src/fb_models/data/__init__.py`
- Create: `src/fb_models/models/__init__.py`
- Create: `src/fb_models/training/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: installable `fb_models` package; `sample_plays` pytest fixture consumed by all test files

- [ ] **Step 1: Add dependencies and build system to `pyproject.toml`**

Replace the entire file with:

```toml
[project]
name = "fb-models"
version = "0.1.0"
description = "NFL play simulation models"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "lightgbm>=4.0",
    "scikit-learn>=1.3",
    "joblib>=1.3",
    "numpy>=1.26",
    "pandas>=2.0",
    "pyarrow>=14.0",
    "requests>=2.32.5",
    "types-requests>=2.32.4.20260107",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/fb_models"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Add `artifacts/` to `.gitignore`**

Append to `.gitignore`:
```
# Trained model artifacts
artifacts/
```

- [ ] **Step 3: Create package directory structure**

```bash
mkdir -p src/fb_models/data src/fb_models/models src/fb_models/training tests
touch src/fb_models/__init__.py
touch src/fb_models/data/__init__.py
touch src/fb_models/models/__init__.py
touch src/fb_models/training/__init__.py
touch tests/__init__.py
```

- [ ] **Step 4: Install dependencies**

```bash
uv sync --dev
```

Expected output ends with: `Installed N packages`

- [ ] **Step 5: Create `tests/conftest.py`**

```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_plays() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    play_types = rng.choice(["run", "pass", "punt", "field_goal"], n)
    complete = np.where(play_types == "pass", rng.integers(0, 2, n), 0)
    incomplete = np.where(play_types == "pass", 1 - complete, 0)
    return pd.DataFrame({
        "play_type": play_types,
        "down": rng.integers(1, 5, n),
        "ydstogo": rng.integers(1, 21, n),
        "yardline_100": rng.integers(1, 100, n),
        "score_differential": rng.integers(-28, 29, n),
        "qtr": rng.integers(1, 6, n),
        "game_seconds_remaining": rng.integers(0, 3601, n),
        "posteam_timeouts_remaining": rng.integers(0, 4, n),
        "defteam_timeouts_remaining": rng.integers(0, 4, n),
        "goal_to_go": rng.integers(0, 2, n),
        "yards_gained": rng.integers(-10, 31, n),
        "complete_pass": complete,
        "incomplete_pass": incomplete,
        "interception": rng.integers(0, 2, n),
        "fumble": rng.integers(0, 2, n),
        "fumble_lost": rng.integers(0, 2, n),
        "play_duration": rng.uniform(1.0, 15.0, n),
    })
```

- [ ] **Step 6: Verify pytest discovers the fixture**

```bash
uv run pytest --collect-only
```

Expected: `selected 0 items` (no tests yet, no errors)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/
git commit -m "chore: project setup — add deps, package structure, test fixture"
```

---

## Task 2: Data Loader

**Files:**
- Create: `src/fb_models/data/loader.py`
- Create: `tests/test_loader.py`

**Interfaces:**
- Consumes: nothing from prior tasks
- Produces:
  - `load_plays(data_dir: Path, min_season: int = 2016) -> pd.DataFrame`
    - Returns DataFrame with columns: `down, ydstogo, yardline_100, score_differential, qtr, game_seconds_remaining, posteam_timeouts_remaining, defteam_timeouts_remaining, goal_to_go, play_type, yards_gained, complete_pass, incomplete_pass, interception, fumble, fumble_lost, play_duration`
    - Rows: regular/postseason scrimmage plays (run/pass/punt/field_goal) from `min_season` onward with non-null `play_duration`

- [ ] **Step 1: Write failing tests**

Create `tests/test_loader.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_loader.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'fb_models'`

- [ ] **Step 3: Implement `src/fb_models/data/loader.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_loader.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/fb_models/data/loader.py tests/test_loader.py
git commit -m "feat: data loader — filter, remap play types, compute play_duration"
```

---

## Task 3: Feature Engineering

**Files:**
- Create: `src/fb_models/data/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes: `sample_plays` fixture (from conftest.py)
- Produces:
  - `FEATURE_COLS: list[str]` — ordered list of 9 feature column names
  - `PLAY_TYPES: list[str]` — `["field_goal", "pass", "punt", "run"]` (sorted, used as class labels)
  - `OUTCOME_COLS: list[str]` — columns extracted from sampled plays
  - `build_feature_matrix(df: pd.DataFrame) -> np.ndarray` — shape `(len(df), 9)`, dtype float64

- [ ] **Step 1: Write failing tests**

Create `tests/test_features.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_features.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'fb_models.data.features'`

- [ ] **Step 3: Implement `src/fb_models/data/features.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_features.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/fb_models/data/features.py tests/test_features.py
git commit -m "feat: feature constants and build_feature_matrix()"
```

---

## Task 4: Play Type Classifier

**Files:**
- Create: `src/fb_models/models/classifier.py`
- Create: `tests/test_classifier.py`

**Interfaces:**
- Consumes:
  - `build_feature_matrix(df) -> np.ndarray` from `fb_models.data.features`
  - `FEATURE_COLS`, `PLAY_TYPES` from `fb_models.data.features`
  - `sample_plays` fixture (all 4 play types present)
- Produces:
  - `train_classifier(df: pd.DataFrame) -> lgb.LGBMClassifier`
  - `predict_play_type_probs(clf: lgb.LGBMClassifier, game_state: np.ndarray) -> dict[str, float]`
    - `game_state`: shape `(9,)` — values in the same order as `FEATURE_COLS`
    - Returns dict with keys exactly matching `PLAY_TYPES`, values summing to 1.0

- [ ] **Step 1: Write failing tests**

Create `tests/test_classifier.py`:

```python
import numpy as np
import pytest
import lightgbm as lgb

from fb_models.models.classifier import predict_play_type_probs, train_classifier
from fb_models.data.features import PLAY_TYPES, FEATURE_COLS


def test_train_classifier_returns_lgbm(sample_plays):
    clf = train_classifier(sample_plays)
    assert isinstance(clf, lgb.LGBMClassifier)


def test_train_classifier_knows_all_play_types(sample_plays):
    clf = train_classifier(sample_plays)
    assert set(clf.classes_) == set(PLAY_TYPES)


def test_predict_returns_dict_with_all_play_types(sample_plays):
    clf = train_classifier(sample_plays)
    game_state = np.array([1, 10, 50, 0, 1, 1800, 3, 3, 0], dtype=np.float64)
    probs = predict_play_type_probs(clf, game_state)
    assert set(probs.keys()) == set(PLAY_TYPES)


def test_predict_probabilities_sum_to_one(sample_plays):
    clf = train_classifier(sample_plays)
    game_state = np.array([1, 10, 50, 0, 1, 1800, 3, 3, 0], dtype=np.float64)
    probs = predict_play_type_probs(clf, game_state)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_all_probabilities_non_negative(sample_plays):
    clf = train_classifier(sample_plays)
    game_state = np.array([4, 15, 65, -14, 4, 120, 1, 3, 0], dtype=np.float64)
    probs = predict_play_type_probs(clf, game_state)
    assert all(p >= 0 for p in probs.values())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_classifier.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'fb_models.models.classifier'`

- [ ] **Step 3: Implement `src/fb_models/models/classifier.py`**

```python
import lightgbm as lgb
import numpy as np
import pandas as pd

from fb_models.data.features import PLAY_TYPES, build_feature_matrix


def train_classifier(df: pd.DataFrame) -> lgb.LGBMClassifier:
    X = build_feature_matrix(df)
    y = df["play_type"]

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(PLAY_TYPES),
        class_weight="balanced",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )
    clf.fit(X, y)
    return clf


def predict_play_type_probs(
    clf: lgb.LGBMClassifier,
    game_state: np.ndarray,
) -> dict[str, float]:
    probs = clf.predict_proba(game_state.reshape(1, -1))[0]
    return dict(zip(clf.classes_, probs))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_classifier.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/fb_models/models/classifier.py tests/test_classifier.py
git commit -m "feat: LightGBM play type classifier — train and predict"
```

---

## Task 5: K-NN Outcome Index

**Files:**
- Create: `src/fb_models/models/knn.py`
- Create: `tests/test_knn.py`

**Interfaces:**
- Consumes:
  - `build_feature_matrix(df) -> np.ndarray` from `fb_models.data.features`
  - `FEATURE_COLS`, `OUTCOME_COLS` from `fb_models.data.features`
  - `sample_plays` fixture
- Produces:
  - `KNNIndex = tuple[NearestNeighbors, StandardScaler, pd.DataFrame]`
  - `build_knn_index(df: pd.DataFrame, play_type: str, k: int = 50) -> KNNIndex`
    - `df`: full plays DataFrame (all play types); function filters internally
    - Returns `(fitted NearestNeighbors, fitted StandardScaler, subset DataFrame with OUTCOME_COLS)`
  - `query_knn(knn_index: KNNIndex, game_state: np.ndarray, rng: np.random.Generator) -> dict[str, object]`
    - `game_state`: shape `(9,)`, same order as `FEATURE_COLS`
    - Returns dict with keys: `play_type, yards_gained, is_complete, is_incomplete, is_intercepted, is_fumble, is_turnover, seconds_elapsed`

- [ ] **Step 1: Write failing tests**

Create `tests/test_knn.py`:

```python
import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from fb_models.models.knn import build_knn_index, query_knn


def test_build_knn_index_returns_correct_types(sample_plays):
    nn, scaler, subset = build_knn_index(sample_plays, "run", k=5)
    assert isinstance(nn, NearestNeighbors)
    assert isinstance(scaler, StandardScaler)
    assert len(subset) == (sample_plays["play_type"] == "run").sum()


def test_build_knn_index_subset_has_outcome_cols(sample_plays):
    _, _, subset = build_knn_index(sample_plays, "pass", k=5)
    required = {"play_type", "yards_gained", "complete_pass", "incomplete_pass",
                "interception", "fumble", "fumble_lost", "play_duration"}
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


def test_query_knn_deterministic_with_same_seed(sample_plays):
    knn_index = build_knn_index(sample_plays, "run", k=10)
    game_state = np.array([2, 5, 30, 7, 2, 900, 2, 3, 0], dtype=np.float64)
    result1 = query_knn(knn_index, game_state, np.random.default_rng(99))
    result2 = query_knn(knn_index, game_state, np.random.default_rng(99))
    assert result1 == result2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_knn.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'fb_models.models.knn'`

- [ ] **Step 3: Implement `src/fb_models/models/knn.py`**

```python
from typing import TypeAlias

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from fb_models.data.features import FEATURE_COLS, OUTCOME_COLS, build_feature_matrix

KNNIndex: TypeAlias = tuple[NearestNeighbors, StandardScaler, pd.DataFrame]


def build_knn_index(
    df: pd.DataFrame,
    play_type: str,
    k: int = 50,
) -> KNNIndex:
    mask = df["play_type"] == play_type
    features = df.loc[mask, FEATURE_COLS].reset_index(drop=True)
    outcomes = df.loc[mask, OUTCOME_COLS].reset_index(drop=True)

    X = features.to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=min(k, len(outcomes)), metric="euclidean", algorithm="ball_tree")
    nn.fit(X_scaled)

    return nn, scaler, outcomes


def query_knn(
    knn_index: KNNIndex,
    game_state: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, object]:
    nn, scaler, outcomes = knn_index
    x_scaled = scaler.transform(game_state.reshape(1, -1))
    _, indices = nn.kneighbors(x_scaled)
    idx = rng.choice(indices[0])
    row = outcomes.iloc[idx]

    return {
        "play_type": str(row["play_type"]),
        "yards_gained": int(row["yards_gained"]),
        "is_complete": bool(row["complete_pass"]),
        "is_incomplete": bool(row["incomplete_pass"]),
        "is_intercepted": bool(row["interception"]),
        "is_fumble": bool(row["fumble"]),
        "is_turnover": bool(row["interception"] or row["fumble_lost"]),
        "seconds_elapsed": float(row["play_duration"]),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_knn.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/fb_models/models/knn.py tests/test_knn.py
git commit -m "feat: K-NN outcome index — build per play type and query"
```

---

## Task 6: End-to-End Training Script

**Files:**
- Create: `src/fb_models/training/train.py`
- Create: `tests/test_train.py`
- Create: `artifacts/.gitkeep`

**Interfaces:**
- Consumes:
  - `load_plays(data_dir, min_season) -> pd.DataFrame` from `fb_models.data.loader`
  - `PLAY_TYPES` from `fb_models.data.features`
  - `train_classifier(df) -> lgb.LGBMClassifier` from `fb_models.models.classifier`
  - `build_knn_index(df, play_type, k) -> KNNIndex` from `fb_models.models.knn`
- Produces:
  - `main(data_dir: Path, artifacts_dir: Path, min_season: int = 2016, k: int = 50) -> None`
  - Artifact files:
    - `{artifacts_dir}/play_type_classifier.joblib`
    - `{artifacts_dir}/knn_run.joblib`
    - `{artifacts_dir}/knn_pass.joblib`
    - `{artifacts_dir}/knn_punt.joblib`
    - `{artifacts_dir}/knn_field_goal.joblib`

- [ ] **Step 1: Write failing test**

Create `tests/test_train.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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

    Each game has exactly two plays of the same type in the same quarter so
    play_duration can be computed for the first play. After loader processing,
    60 plays survive — 15 per play type.
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
                "play_clock": int(rng.integers(10, 36)),
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
        nn, scaler, outcomes = joblib.load(artifacts_dir / f"knn_{pt}.joblib")
        assert isinstance(nn, NearestNeighbors)
        assert isinstance(scaler, StandardScaler)
        assert isinstance(outcomes, pd.DataFrame)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_train.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'fb_models.training.train'`

- [ ] **Step 3: Implement `src/fb_models/training/train.py`**

```python
from pathlib import Path

import joblib

from fb_models.data.features import PLAY_TYPES
from fb_models.data.loader import load_plays
from fb_models.models.classifier import train_classifier
from fb_models.models.knn import build_knn_index


def main(
    data_dir: Path = Path("data"),
    artifacts_dir: Path = Path("artifacts"),
    min_season: int = 2016,
    k: int = 50,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading plays (season >= {min_season})...")
    df = load_plays(data_dir, min_season)
    print(f"Loaded {len(df):,} plays.")

    print("Training play type classifier...")
    clf = train_classifier(df)
    joblib.dump(clf, artifacts_dir / "play_type_classifier.joblib")
    print("  Saved play_type_classifier.joblib")

    for play_type in PLAY_TYPES:
        n = (df["play_type"] == play_type).sum()
        print(f"Building K-NN index for {play_type!r} ({n:,} plays)...")
        knn_index = build_knn_index(df, play_type, k=k)
        joblib.dump(knn_index, artifacts_dir / f"knn_{play_type}.joblib")
        print(f"  Saved knn_{play_type}.joblib")

    print("Training complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `artifacts/.gitkeep`**

```bash
mkdir -p artifacts && touch artifacts/.gitkeep
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_train.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Run the full test suite**

```bash
uv run pytest -v
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/fb_models/training/train.py tests/test_train.py artifacts/.gitkeep
git commit -m "feat: end-to-end training script — load, train, serialize artifacts"
```

---

## Running Training on Real Data

Once all tasks are complete, train on the real dataset:

```bash
uv run python -m fb_models.training.train
```

Expected output:
```
Loading plays (season >= 2016)...
Loaded ~XXX,XXX plays.
Training play type classifier...
  Saved play_type_classifier.joblib
Building K-NN index for 'field_goal' (X,XXX plays)...
  Saved knn_field_goal.joblib
Building K-NN index for 'pass' (XX,XXX plays)...
  Saved knn_pass.joblib
Building K-NN index for 'punt' (X,XXX plays)...
  Saved knn_punt.joblib
Building K-NN index for 'run' (XX,XXX plays)...
  Saved knn_run.joblib
Training complete.
```
