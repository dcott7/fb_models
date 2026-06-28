import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_plays() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    play_types = rng.choice(["run", "pass", "punt", "field_goal"], n)
    complete = np.where(play_types == "pass", rng.integers(0, 2, n), 0)
    incomplete = np.where(play_types == "pass", 1 - complete, 0)
    return pd.DataFrame({
        "play_type": play_types,
        "down": rng.integers(1, 5, n),
        "ydstogo": rng.integers(1, 21, n),
        "yardline_100": rng.integers(1, 100, n),
        "score_differential": rng.integers(-28, 29, n),
        "qtr": rng.integers(1, 6, n),
        "game_seconds_remaining": rng.integers(0, 3601, n),
        "posteam_timeouts_remaining": rng.integers(0, 4, n),
        "defteam_timeouts_remaining": rng.integers(0, 4, n),
        "goal_to_go": rng.integers(0, 2, n),
        "yards_gained": rng.integers(-10, 31, n),
        "complete_pass": complete,
        "incomplete_pass": incomplete,
        "interception": rng.integers(0, 2, n),
        "fumble": rng.integers(0, 2, n),
        "fumble_lost": rng.integers(0, 2, n),
        "play_duration": rng.uniform(1.0, 15.0, n),
    })
