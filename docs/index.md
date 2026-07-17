# fb_models

NFL play simulation models. This project owns the data pipeline, feature engineering, and trained model artifacts for a play-by-play NFL game simulator — the simulation engine itself lives in a separate project that depends on `fb_models` as a library.

## Layout

- `src/fb_models/dataset.py` — fetches and caches nflverse play-by-play, participation, roster, player, and schedule/odds data.
- `src/fb_models/features/` — `FeatureStore`: builds lookup dictionaries (situations, coach tendencies, player usage, participation, games) for simulation-time queries.
- `src/fb_models/modeling/` — trainable models. `plays.py` holds feature engineering shared across models (previous-play lag features, expanding coach-tendency rates, game context); each model gets its own subpackage, e.g. `play_type/{features,train,predict}.py`.
- `notebooks/` — data exploration and model training notebooks.
- `examples/` — worked examples showing how a consumer project would call into `fb_models`.

## Getting started

```bash
uv sync
make data    # cache historical nflverse data
make train   # train the play-type classifier
make test
```

See the [design spec](superpowers/specs/2026-06-25-play-simulation-models-design.md) and [implementation plan](superpowers/plans/2026-06-25-play-simulation-models.md) for the original project scope and rationale.
