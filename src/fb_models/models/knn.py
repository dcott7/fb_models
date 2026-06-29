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
