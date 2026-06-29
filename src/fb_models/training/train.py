from pathlib import Path

import joblib

from fb_models.data.features import PLAY_TYPES
from fb_models.data.loader import load_plays
from fb_models.models.classifier import train_classifier
from fb_models.models.knn import build_knn_index


def main(
    data_dir: Path = Path("data"),
    artifacts_dir: Path = Path("artifacts"),
    min_season: int = 2016,
    k: int = 50,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading plays (season >= {min_season})...")
    df = load_plays(data_dir, min_season)
    print(f"Loaded {len(df):,} plays.")

    print("Training play type classifier...")
    clf = train_classifier(df)
    joblib.dump(clf, artifacts_dir / "play_type_classifier.joblib")
    print("  Saved play_type_classifier.joblib")

    for play_type in PLAY_TYPES:
        n = (df["play_type"] == play_type).sum()
        print(f"Building K-NN index for {play_type!r} ({n:,} plays)...")
        knn_index = build_knn_index(df, play_type, k=k)
        joblib.dump(knn_index, artifacts_dir / f"knn_{play_type}.joblib")
        print(f"  Saved knn_{play_type}.joblib")

    print("Training complete.")


if __name__ == "__main__":
    main()
