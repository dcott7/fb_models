from pathlib import Path
from typing import TypedDict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, log_loss

from fb_models.data.features import PLAY_TYPES, build_feature_matrix
from fb_models.data.loader import load_plays
from fb_models.models.classifier import train_classifier
from fb_models.models.knn import KNNIndex, build_knn_index, query_knn


class ClassifierMetrics(TypedDict):
    accuracy: float
    log_loss: float
    classification_report: str
    confusion_matrix: np.ndarray
    labels: list[str]


class OutcomeComparison(TypedDict):
    n_real: int
    n_sampled: int
    yards_gained_mean_real: float
    yards_gained_mean_sampled: float
    yards_gained_std_real: float
    yards_gained_std_sampled: float
    turnover_rate_real: float
    turnover_rate_sampled: float
    complete_rate_real: float
    complete_rate_sampled: float


def discover_seasons(data_dir: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in data_dir.glob("pbp_*.parquet"))


def evaluate_classifier(clf: lgb.LGBMClassifier, test_df: pd.DataFrame) -> ClassifierMetrics:
    X = build_feature_matrix(test_df)
    y_true = test_df["play_type"].to_numpy()
    y_pred = clf.predict(X)
    y_proba = np.asarray(clf.predict_proba(X))  # type: ignore

    labels = [str(c) for c in clf.classes_]  # type: ignore
    accuracy = float((y_pred == y_true).mean())
    loss = float(log_loss(y_true, y_proba, labels=labels))
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": accuracy,
        "log_loss": loss,
        "classification_report": str(report),
        "confusion_matrix": cm,
        "labels": labels,
    }


def evaluate_knn_outcomes(
    knn_index: KNNIndex,
    test_subset: pd.DataFrame,
    rng: np.random.Generator,
    n_samples: int = 2000,
) -> OutcomeComparison:
    if len(test_subset) > n_samples:
        seed = int(rng.integers(0, 2**31 - 1))
        test_subset = test_subset.sample(n=n_samples, random_state=seed)

    X = build_feature_matrix(test_subset)
    sampled = [query_knn(knn_index, X[i], rng) for i in range(len(test_subset))]
    sampled_df = pd.DataFrame(sampled)

    real_turnover = (test_subset["interception"] == 1) | (test_subset["fumble_lost"] == 1)

    return {
        "n_real": len(test_subset),
        "n_sampled": len(sampled_df),
        "yards_gained_mean_real": float(test_subset["yards_gained"].mean()),
        "yards_gained_mean_sampled": float(sampled_df["yards_gained"].mean()),
        "yards_gained_std_real": float(test_subset["yards_gained"].std()),
        "yards_gained_std_sampled": float(sampled_df["yards_gained"].std()),
        "turnover_rate_real": float(real_turnover.mean()),
        "turnover_rate_sampled": float(sampled_df["is_turnover"].mean()),
        "complete_rate_real": float(test_subset["complete_pass"].mean()),
        "complete_rate_sampled": float(sampled_df["is_complete"].mean()),
    }


def _print_report(
    test_season: int,
    classifier_metrics: ClassifierMetrics,
    outcome_comparisons: dict[str, OutcomeComparison],
) -> None:
    print(f"\n=== Play Type Classifier (test season {test_season}) ===")
    print(f"Accuracy: {classifier_metrics['accuracy']:.3f}")
    print(f"Log loss: {classifier_metrics['log_loss']:.3f}")
    print("\nClassification report:")
    print(classifier_metrics["classification_report"])

    labels = classifier_metrics["labels"]
    cm_df = pd.DataFrame(classifier_metrics["confusion_matrix"], index=labels, columns=labels)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm_df.to_string())

    print("\n=== K-NN Outcome Sampler ===")
    for play_type, comp in outcome_comparisons.items():
        print(f"\n--- {play_type} (n_real={comp['n_real']}, n_sampled={comp['n_sampled']}) ---")
        rows = [
            ("yards_gained mean", comp["yards_gained_mean_real"], comp["yards_gained_mean_sampled"]),
            ("yards_gained std", comp["yards_gained_std_real"], comp["yards_gained_std_sampled"]),
            ("turnover rate", comp["turnover_rate_real"], comp["turnover_rate_sampled"]),
        ]
        if play_type == "pass":
            rows.append(("completion rate", comp["complete_rate_real"], comp["complete_rate_sampled"]))
        table = pd.DataFrame(rows, columns=["metric", "real", "sampled"]).set_index("metric")
        print(table.to_string(float_format=lambda x: f"{x:.3f}"))


def main(
    data_dir: Path = Path("data"),
    min_season: int = 2016,
    test_season: int | None = None,
    k: int = 50,
    n_eval_samples: int = 2000,
    seed: int = 0,
) -> None:
    seasons = discover_seasons(data_dir)
    if not seasons:
        raise ValueError(f"No pbp_*.parquet files found in {data_dir}")
    if test_season is None:
        test_season = max(seasons)
    if test_season <= min_season:
        raise ValueError(
            f"test_season ({test_season}) must be greater than min_season ({min_season})"
        )

    print(f"Train: seasons {min_season}-{test_season - 1}, Test: season {test_season}")

    train_df = load_plays(data_dir, min_season=min_season, max_season=test_season - 1)
    test_df = load_plays(data_dir, min_season=test_season, max_season=test_season)
    print(f"Train plays: {len(train_df):,}, Test plays: {len(test_df):,}")

    print("Training classifier on train split...")
    clf = train_classifier(train_df)
    classifier_metrics = evaluate_classifier(clf, test_df)

    rng = np.random.default_rng(seed)
    outcome_comparisons: dict[str, OutcomeComparison] = {}
    for play_type in PLAY_TYPES:
        if (train_df["play_type"] == play_type).sum() == 0:
            continue
        test_subset = test_df[test_df["play_type"] == play_type]
        if len(test_subset) == 0:
            continue
        print(f"Building K-NN index for {play_type!r} on train split...")
        knn_index = build_knn_index(train_df, play_type, k=k)
        outcome_comparisons[play_type] = evaluate_knn_outcomes(
            knn_index, test_subset, rng, n_eval_samples
        )

    _print_report(test_season, classifier_metrics, outcome_comparisons)


if __name__ == "__main__":
    main()
