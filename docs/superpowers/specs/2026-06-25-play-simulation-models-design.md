# Play Simulation Models — Design Spec

**Date:** 2026-06-25
**Status:** Approved

---

## Overview

This project builds the data science models that power a play-by-play NFL game simulation engine. Given a game state, the models produce a statistically realistic play outcome drawn from historical NFL play-by-play data (nflverse, 1999–present).

This project owns only the models and serialized artifacts. The simulation engine that calls them lives in a separate project.

---

## Scope

### Play Types Modeled

- `run` — rushing plays, including QB kneels (`qb_kneel` remapped to `run` during preprocessing)
- `pass` — passing plays, including sacks and scrambles; QB spikes (`qb_spike`) remapped to `pass` and treated as incomplete passes during preprocessing
- `punt`
- `field_goal`

**Excluded:** kickoffs (handled by the simulation engine), extra points, two-point conversions.

### Output Fields

Each sampled play produces the following fields, extracted directly from the matching historical play:

| Field | Source |
|---|---|
| `play_type` | `play_type` column (matches the selected type) |
| `yards_gained` | `yards_gained` |
| `is_complete` | `complete_pass` |
| `is_incomplete` | `incomplete_pass` |
| `is_intercepted` | `interception` |
| `is_fumble` | `fumble` |
| `is_turnover` | `interception == 1 OR fumble_lost == 1` |
| `seconds_elapsed` | precomputed `play_duration` (see Data Pipeline) |

Additional output fields can be added later by pulling additional columns from the parquet data during preprocessing — no model retraining required.

### Input Features (Game State)

| Feature | Notes |
|---|---|
| `down` | 1–4 |
| `yards_to_go` | yards to first down |
| `yardline_100` | distance to opponent end zone (1–99) |
| `score_differential` | posteam − defteam score |
| `quarter` | 1–5 (5 = OT) |
| `game_seconds_remaining` | |
| `posteam_timeouts` | 0–3 |
| `defteam_timeouts` | 0–3 |
| `goal_to_go` | binary |

---

## Architecture

The system is two sequential components:

```
game_state + seed
      │
      ▼
┌──────────────────────┐
│  Play Type Classifier │  GBM → P(run), P(pass), P(punt), P(field_goal)
│                       │  sample play_type using seed
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  K-NN Outcome Index  │  One index per play type.
│  (per play type)     │  Find K nearest historical plays,
│                       │  sample one → return all output fields.
└──────────────────────┘
```

**Key design principle:** The classifier handles *which play type*, the K-NN index handles *what happened*. These components are independently trainable and replaceable.

---

## Data Pipeline

### Source

nflverse parquet files: `data/pbp_{season}.parquet`, seasons 1999–2025.

### Filtering

- `play_type` in `{run, pass, punt, field_goal}`
- `season_type` in `{REG, POST}` — excludes preseason
- `down` is not null — eliminates kickoffs, administrative rows, and between-play records
- `season >= MIN_SEASON` (default: **2016**) — focuses on the modern NFL era; parameterized for easy adjustment

### `seconds_elapsed`: Random Draw by Play Type

nflverse does not carry a usable snap-to-whistle duration signal — `play_clock` is uniformly `"0"` in real data, so it cannot be used to derive play duration. Instead, `seconds_elapsed` is generated at K-NN query time as a uniform random draw, with the range conditioned on `play_type`:

| `play_type`   | Range (seconds) |
|---------------|------------------|
| `run`         | 4.0 – 7.0        |
| `pass`        | 5.0 – 8.0        |
| `punt`        | 3.0 – 5.0        |
| `field_goal`  | 2.0 – 4.0        |

This draw uses the same `rng` passed into `query_knn`, so results stay deterministic given a fixed seed. It is not a historical column and is not stored in the loaded play data or the K-NN outcome index — `play_clock` is dropped from the data pipeline entirely.

### Output

A single cleaned DataFrame with all feature columns and all outcome columns required by downstream models.

---

## Play Type Classifier

**Model:** LightGBM multi-class classifier.

**Why GBM:** The relationships are non-linear and interaction-heavy (punt probability spikes sharply on 4th down; field goal probability peaks in the 20–35 yard range; pass rate rises with score deficit late in the game). Gradient boosting handles this naturally, requires no feature scaling, and handles missing values natively.

**Class imbalance:** Field goals and punts are rarer than run/pass. Addressed via `class_weight='balanced'` in LightGBM — preserves the training distribution while preventing the model from ignoring minority classes.

**Sampling at inference:** The model outputs `[p_run, p_pass, p_punt, p_fg]`. The caller samples from this multinomial using the provided seed (via `numpy.random.Generator`), making results deterministic given the same seed and game state.

**Evaluation:** Calibration is the primary metric. A predicted `P(punt) = 0.6` on 4th-and-long at midfield should correspond to a historical punt rate of ~60% in those situations. Validated with calibration curves per play type and spot-checked marginal distributions (e.g., overall pass rate ~58%, punt rate on 4th down) against historical actuals.

**Artifact:** `artifacts/play_type_classifier.joblib`

---

## K-NN Outcome Sampling

**Structure:** One `sklearn.neighbors.NearestNeighbors` index per play type (4 total). Each index is fit only on plays of that type, so a pass play's neighbors are always other pass plays.

**Feature scaling:** K-NN is distance-sensitive. Each index uses a `StandardScaler` fit on its play-type subset's training data. Scalers are saved alongside indexes.

**K:** Default `K=50`. Retrieves 50 nearest neighbors and samples one uniformly using the provided seed. Large enough to introduce variability; small enough to ensure neighbors are genuinely similar situations. Tunable.

**Distance metric:** Euclidean on the standardized feature vector. All 9 features contribute equally. Feature weighting can be revisited if certain features (e.g., `down`, `yards_to_go`) should dominate distance.

**Extensibility:** Adding a new output field requires only pulling an additional column from parquet during preprocessing and extracting it from the sampled row. No index rebuilding, no classifier retraining.

**Artifacts:**
- `artifacts/knn_{play_type}.joblib` — K-NN index
- `artifacts/scaler_{play_type}.joblib` — fitted StandardScaler

---

## Project Layout

```
fb_models/
  src/fb_models/
    data/
      loader.py       # load & filter parquets, compute play_duration
      features.py     # feature constants, feature vector construction
    models/
      classifier.py   # LightGBM train/serialize
      knn.py          # K-NN index build/serialize per play type
    training/
      train.py        # end-to-end: load → train → save artifacts
  artifacts/          # gitignored — serialized model artifacts
  docs/
    superpowers/specs/
  scripts/
    pull_pbp.py
  notebooks/
  tests/
```

---

## Decisions & Rationale

| Decision | Choice | Rationale |
|---|---|---|
| Team identity | Excluded | Model is team-agnostic by design; used as universal default |
| Era handling | `season >= 2016` filter | Modern NFL era; parameterized cutoff for easy adjustment |
| Play type model | LightGBM multi-class | Handles non-linearities and interactions; calibration-focused |
| Outcome model | K-NN sampling | Preserves realistic joint distributions across all output fields |
| Kickoffs | Out of scope | Handled by simulation engine |
| Extra points / 2-pt | Out of scope | Excluded by user decision |
| `seconds_elapsed` | Play duration (snap to whistle) | Computed via play_clock adjustment; inter-play time excluded |
| Simulation layer | Out of scope | Lives in a separate project; this project ships artifacts only |
