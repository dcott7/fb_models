"""
Example: running the play-type model outside of this project.

This project owns only the models and serialized artifacts (see
docs/superpowers/specs/2026-06-25-play-simulation-models-design.md); the
simulation engine itself is expected to live in a separate project that
depends on fb_models as a library rather than re-deriving its data loading
or feature engineering. This script shows what that integration looks
like end to end, starting with the historical data needed to build the
coach-tendency and game-context lookups the model needs at inference time.
"""

from fb_models.dataset import (
    load_games_dataset,
    load_participation_dataset,
    load_pbp_dataset,
)

SEASONS = list(range(2019, 2025))


def load_simulation_data() -> tuple:
    """Load the historical pbp/participation/games data needed to build
    the coach-tendency and game-context lookups the play-type model
    requires (see fb_models.modeling.plays.build_plays).
    """
    pbp = load_pbp_dataset(seasons=SEASONS)
    participation = load_participation_dataset(seasons=SEASONS)
    games = load_games_dataset()

    return pbp, participation, games


if __name__ == "__main__":
    pbp, participation, games = load_simulation_data()
    print(f"pbp: {len(pbp):,} plays")
    print(f"participation: {len(participation):,} rows")
    print(f"games: {len(games):,} games")
