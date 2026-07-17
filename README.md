# fb_models

NFL play simulation models. This project owns the data pipeline, feature engineering, and trained model artifacts for a play-by-play NFL game simulator. The simulation engine itself lives in a separate project that depends on `fb_models` as a library — see `examples/game_simulation.py` for what that integration looks like.

## Setup

```bash
uv sync
```

## Layout

```
├── data/
│   ├── raw/            <- cached nflverse downloads (pbp, participation, rosters, players, games)
│   ├── interim/
│   ├── processed/
│   └── external/        <- third-party sources outside nflverse (e.g. ESPN)
├── docs/                 <- mkdocs project (`make docs`)
├── models/                <- trained/serialized model artifacts
├── notebooks/              <- data exploration and model training notebooks
├── examples/                <- worked examples of consuming fb_models as a library
├── scripts/                  <- standalone data-pull utilities
├── tests/
└── src/fb_models/
    ├── config.py               <- shared paths (REPO_ROOT, data dirs, MODELS_DIR)
    ├── dataset.py                <- nflverse data loading/caching
    ├── features/                  <- FeatureStore: simulation-time lookup dicts
    └── modeling/                   <- trainable models; plays.py has feature engineering
        └── play_type/                shared across models, one subpackage per model
```

## Common tasks

```bash
make data          # fetch and cache nflverse data
make train          # train the play-type classifier
make test
make lint            # black --check + flake8
make format
make docs             # serve the mkdocs site locally
```

See `docs/superpowers/specs/2026-06-25-play-simulation-models-design.md` for the original design spec.
