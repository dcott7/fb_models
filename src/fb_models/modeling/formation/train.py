import lightgbm as lgb
import pandas as pd

from ..plays import build_plays
from .features import (
    FORMATION_CATEGORICAL_COLS,
    FORMATION_FEATURE_COLS,
    select_formation_features,
)

_FORMATIONS = ["PISTOL", "SHOTGUN", "UNDER CENTER"]


def build_formation_training_set(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    return select_formation_features(build_plays(pbp_df, participation_df, games_df))


def train_formation_classifier(training_df: pd.DataFrame) -> lgb.LGBMClassifier:
    X = training_df[FORMATION_FEATURE_COLS]
    y = training_df["offense_formation"]

    # No class_weight="balanced" here, unlike play_type: empirically it hurts
    # accuracy, log_loss, *and* calibration for this label (0.619->0.720 acc,
    # 0.810->0.660 log_loss when dropped) because PISTOL is a genuinely noisy
    # minority class rather than a structurally-deterministic one the way
    # punt/field_goal are for play_type -- balancing against it just makes
    # the model over-predict it without a compensating calibration benefit.
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(_FORMATIONS),
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )
    clf.fit(X, y, categorical_feature=FORMATION_CATEGORICAL_COLS)  # type: ignore
    return clf
