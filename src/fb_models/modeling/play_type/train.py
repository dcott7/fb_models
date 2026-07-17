import lightgbm as lgb
import pandas as pd

from ..plays import build_plays
from .features import (
    PLAY_TYPE_CATEGORICAL_COLS,
    PLAY_TYPE_FEATURE_COLS,
    select_play_type_features,
)

_PLAY_TYPES = ["field_goal", "pass", "punt", "run"]


def build_play_type_training_set(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    return select_play_type_features(build_plays(pbp_df, participation_df, games_df))


def train_play_type_classifier(training_df: pd.DataFrame) -> lgb.LGBMClassifier:
    X = training_df[PLAY_TYPE_FEATURE_COLS]
    y = training_df["play_type"]

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(_PLAY_TYPES),
        class_weight="balanced",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )
    clf.fit(X, y, categorical_feature=PLAY_TYPE_CATEGORICAL_COLS)  # type: ignore
    return clf
