import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb

from fb_models.models.classifier import predict_play_type_probs, train_classifier
from fb_models.data.features import PLAY_TYPES


def test_train_classifier_returns_lgbm(sample_plays: pd.DataFrame) -> None:
    clf = train_classifier(sample_plays)
    assert isinstance(clf, lgb.LGBMClassifier)


def test_train_classifier_knows_all_play_types(sample_plays: pd.DataFrame) -> None:
    clf = train_classifier(sample_plays)
    assert set(clf.classes_) == set(PLAY_TYPES)


def test_predict_returns_dict_with_all_play_types(sample_plays: pd.DataFrame) -> None:
    clf = train_classifier(sample_plays)
    game_state = np.array([1, 10, 50, 0, 1, 1800, 3, 3, 0], dtype=np.float64)
    probs = predict_play_type_probs(clf, game_state)
    assert set(probs.keys()) == set(PLAY_TYPES)


def test_predict_probabilities_sum_to_one(sample_plays: pd.DataFrame) -> None:
    clf = train_classifier(sample_plays)
    game_state = np.array([1, 10, 50, 0, 1, 1800, 3, 3, 0], dtype=np.float64)
    probs = predict_play_type_probs(clf, game_state)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_all_probabilities_non_negative(sample_plays: pd.DataFrame) -> None:
    clf = train_classifier(sample_plays)
    game_state = np.array([4, 15, 65, -14, 4, 120, 1, 3, 0], dtype=np.float64)
    probs = predict_play_type_probs(clf, game_state)
    assert all(p >= 0 for p in probs.values())
