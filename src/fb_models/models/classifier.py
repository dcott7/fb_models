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
    clf.fit(X, y) # type: ignore
    return clf


def predict_play_type_probs(
    clf: lgb.LGBMClassifier,
    game_state: np.ndarray,
) -> dict[str, float]:
    probs = np.asarray(
        clf.predict_proba(game_state.reshape(1, -1)) # type: ignore
    )[0]
    return dict(zip(clf.classes_, probs))
