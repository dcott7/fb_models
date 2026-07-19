import lightgbm as lgb
import pandas as pd

from ..plays import build_plays
from .features import (
    PERSONNEL_CATEGORICAL_COLS,
    PERSONNEL_FEATURE_COLS,
    select_personnel_features,
)

_PACKAGES = ["11", "12", "21", "13", "22", "01", "10", "02", "OTHER"]


def build_personnel_training_set(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    return select_personnel_features(build_plays(pbp_df, participation_df, games_df))


def train_personnel_classifier(training_df: pd.DataFrame) -> lgb.LGBMClassifier:
    X = training_df[PERSONNEL_FEATURE_COLS]
    y = training_df["offense_personnel_package"]

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(_PACKAGES),
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )
    clf.fit(X, y, categorical_feature=PERSONNEL_CATEGORICAL_COLS)  # type: ignore
    return clf
