import lightgbm as lgb
import pandas as pd

from ...dataset import (
    _FOUR_MINUTE_SECONDS,
    _GOAL_LINE_YARDLINE,
    _LATE_GAME_SECONDS,
    _NEUTRAL_SCORE_MARGIN,
    _RED_ZONE_YARDLINE,
    _TWO_MINUTE_SECONDS,
)
from .features import PLAY_TYPE_CATEGORICAL_COLS, PLAY_TYPE_FEATURE_COLS

_SCORE_MARGIN_BINS = [-100, -14, -7, 0, 7, 14, 100]
_SCORE_MARGIN_LABELS = [
    "trailing_big",
    "trailing_small",
    "tied_or_close",
    "leading_small",
    "leading_big",
    "blowout",
]

_YARDAGE_BINS = [0, 2, 6, 10, 100]
_YARDAGE_LABELS = ["short", "medium", "long", "very_long"]

# feature name -> True if keyed by offense_coach in the coach snapshot, False
# if keyed by defense_coach. Matches _COACH_TENDENCY_SPECS in modeling/plays.py.
_COACH_TENDENCY_IS_OFFENSE = {
    "off_pass_rate_hist": True,
    "off_early_down_pass_rate_hist": True,
    "off_red_zone_pass_rate_hist": True,
    "off_goal_line_pass_rate_hist": True,
    "off_neutral_pass_rate_hist": True,
    "off_fourth_down_go_for_it_rate_hist": True,
    "off_shotgun_rate_hist": True,
    "off_no_huddle_rate_hist": True,
    "def_pass_rate_allowed_hist": False,
    "def_blitz_rate_hist": False,
    "def_pressure_rate_hist": False,
    "def_man_rate_hist": False,
}


def build_live_feature_row(
    *,
    down: int,
    ydstogo: int,
    yardline_100: int,
    goal_to_go: bool,
    qtr: int,
    game_seconds_remaining: float,
    half_seconds_remaining: float,
    score_differential: int,
    posteam_timeouts_remaining: int,
    defteam_timeouts_remaining: int,
    previous_play_type: str | None,
    previous_yards_gained: float | None,
    previous_first_down: float | None,
    previous_turnover: float | None,
    offense_coach: str,
    defense_coach: str,
    coach_snapshot: dict,
    posteam_spread_line: float | None,
    total_line: float | None,
    div_game: bool,
    posteam_rest: int,
    defteam_rest: int,
    temp: float | None,
    wind: float | None,
    roof: str,
) -> pd.DataFrame:
    """Build a single-row, correctly-typed feature DataFrame for
    predict_play_type_probs from raw simulation game-state values.

    Reuses the same derived-column thresholds and bucket bins as
    dataset.py's _add_derived_columns_pbp (imported directly, not
    redefined) so a live prediction can't silently drift from what the
    model was actually trained on.

    coach_snapshot is the dict returned by
    fb_models.modeling.plays.build_coach_snapshot, keyed by coach name.
    A coach's tendency features are looked up from offense_coach's entry
    for the off_* features and defense_coach's entry for the def_*
    features -- missing entries (e.g. a coach with no tracked history as
    of the as_of cutoff) come through as NaN, which LightGBM handles
    natively.
    """
    row: dict = {
        "down": down,
        "ydstogo": ydstogo,
        "yardline_100": yardline_100,
        "goal_to_go": int(goal_to_go),
        "qtr": qtr,
        "game_seconds_remaining": game_seconds_remaining,
        "half_seconds_remaining": half_seconds_remaining,
        "score_differential": score_differential,
        "posteam_timeouts_remaining": posteam_timeouts_remaining,
        "defteam_timeouts_remaining": defteam_timeouts_remaining,
        "red_zone": yardline_100 <= _RED_ZONE_YARDLINE,
        "goal_line": yardline_100 <= _GOAL_LINE_YARDLINE,
        "two_minute": half_seconds_remaining <= _TWO_MINUTE_SECONDS,
        "four_minute": (
            qtr == 4
            and game_seconds_remaining <= _FOUR_MINUTE_SECONDS
            and score_differential > 0
        ),
        "passing_down": (down == 3 and ydstogo >= 7) or (down == 4 and ydstogo >= 5),
        "short_yardage": ydstogo <= 2,
        "neutral_script": (
            abs(score_differential) <= _NEUTRAL_SCORE_MARGIN
            and not (qtr == 4 and game_seconds_remaining <= _LATE_GAME_SECONDS)
        ),
        "previous_yards_gained": previous_yards_gained,
        "previous_first_down": previous_first_down,
        "previous_turnover": previous_turnover,
        "posteam_spread_line": posteam_spread_line,
        "total_line": total_line,
        "div_game": int(div_game),
        "posteam_rest": posteam_rest,
        "defteam_rest": defteam_rest,
        "temp": temp,
        "wind": wind,
        "score_margin_bucket": pd.cut(
            [score_differential], bins=_SCORE_MARGIN_BINS, labels=_SCORE_MARGIN_LABELS
        )[0],
        "yardage_bucket": pd.cut(
            [ydstogo], bins=_YARDAGE_BINS, labels=_YARDAGE_LABELS, include_lowest=True
        )[0],
        "previous_play_type": previous_play_type,
        "roof": roof,
    }

    for name, is_offense in _COACH_TENDENCY_IS_OFFENSE.items():
        coach = offense_coach if is_offense else defense_coach
        row[name] = coach_snapshot.get(coach, {}).get(name)

    df = pd.DataFrame([row])
    for col in PLAY_TYPE_CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    return df[PLAY_TYPE_FEATURE_COLS]


def predict_play_type_probs(
    clf: lgb.LGBMClassifier, game_state: pd.DataFrame
) -> dict[str, float]:
    probs = clf.predict_proba(game_state[PLAY_TYPE_FEATURE_COLS])[0]  # type: ignore
    return dict(zip(clf.classes_, probs))  # type: ignore
