import lightgbm as lgb
import pandas as pd

from .features import PLAY_TYPE_FEATURE_COLS


def predict_play_type_probs(
    clf: lgb.LGBMClassifier, game_state: pd.DataFrame
) -> dict[str, float]:
    probs = clf.predict_proba(game_state[PLAY_TYPE_FEATURE_COLS])[0]  # type: ignore
    return dict(zip(clf.classes_, probs))  # type: ignore
