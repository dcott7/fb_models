import pandas as pd

_VALID_PLAY_TYPES = {"run", "pass", "punt", "field_goal"}
_PLAY_TYPE_REMAP = {"qb_kneel": "run", "qb_spike": "pass"}

_BLITZ_PASS_RUSHERS = 5

_SORT_COLS = ["season", "week", "order_sequence"]

_PREVIOUS_PLAY_SOURCE_COLS = {
    "previous_play_type": "play_type",
    "previous_yards_gained": "yards_gained",
    "previous_first_down": "first_down",
}

_PARTICIPATION_COLS = [
    "nflverse_game_id",
    "play_id",
    "defense_man_zone_type",
    "was_pressure",
    "number_of_pass_rushers",
]

_GAMES_COLS = [
    "game_id",
    "spread_line",
    "total_line",
    "div_game",
    "away_rest",
    "home_rest",
    "roof",
    "temp",
    "wind",
]


def _filter_and_label(pbp_df: pd.DataFrame) -> pd.DataFrame:
    df = pbp_df[(pbp_df["down"].notna()) & (pbp_df["season_type"] == "REG")].copy()

    df["play_type"] = df["play_type"].replace(_PLAY_TYPE_REMAP)
    df = df[df["play_type"].isin(_VALID_PLAY_TYPES)]

    return df


def _add_previous_play_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(_SORT_COLS)
    grouped = df.groupby(["game_id", "posteam"])

    for new_col, source_col in _PREVIOUS_PLAY_SOURCE_COLS.items():
        df[new_col] = grouped[source_col].shift(1)

    turnover = ((df["interception"] == 1) | (df["fumble_lost"] == 1)).astype(float)
    df["previous_turnover"] = turnover.groupby([df["game_id"], df["posteam"]]).shift(1)

    return df


def _expanding_rate(
    df: pd.DataFrame,
    group_col: str,
    hit_mask: pd.Series,
    eligible_mask: pd.Series,
) -> pd.Series:
    hits = (hit_mask & eligible_mask).astype(float)
    eligible = eligible_mask.astype(float)

    cum_hits = hits.groupby(df[group_col]).cumsum() - hits
    cum_eligible = eligible.groupby(df[group_col]).cumsum() - eligible

    return cum_hits / cum_eligible


def _add_coach_tendency_features(
    df: pd.DataFrame, participation_df: pd.DataFrame
) -> pd.DataFrame:
    part = participation_df[_PARTICIPATION_COLS]

    df = df.merge(
        part,
        left_on=["game_id", "play_id"],
        right_on=["nflverse_game_id", "play_id"],
        how="left",
    )
    df = df.sort_values(_SORT_COLS)

    # offense history
    df["off_pass_rate_hist"] = _expanding_rate(
        df, "offense_coach", df["is_pass"], df["is_go_for_it"]
    )
    df["off_early_down_pass_rate_hist"] = _expanding_rate(
        df, "offense_coach", df["is_pass"], df["is_go_for_it"] & df["early_down"]
    )
    df["off_red_zone_pass_rate_hist"] = _expanding_rate(
        df, "offense_coach", df["is_pass"], df["is_go_for_it"] & df["red_zone"]
    )
    df["off_goal_line_pass_rate_hist"] = _expanding_rate(
        df, "offense_coach", df["is_pass"], df["is_go_for_it"] & df["goal_line"]
    )
    df["off_neutral_pass_rate_hist"] = _expanding_rate(
        df, "offense_coach", df["is_pass"], df["is_go_for_it"] & df["neutral_script"]
    )
    df["off_fourth_down_go_for_it_rate_hist"] = _expanding_rate(
        df, "offense_coach", df["is_go_for_it"], df["down"] == 4
    )
    df["off_shotgun_rate_hist"] = _expanding_rate(
        df, "offense_coach", df["shotgun"] == 1, df["is_go_for_it"]
    )
    df["off_no_huddle_rate_hist"] = _expanding_rate(
        df, "offense_coach", df["no_huddle"] == 1, df["is_go_for_it"]
    )

    # defense history
    df["def_pass_rate_allowed_hist"] = _expanding_rate(
        df, "defense_coach", df["is_pass"], df["is_go_for_it"]
    )
    df["def_blitz_rate_hist"] = _expanding_rate(
        df,
        "defense_coach",
        df["number_of_pass_rushers"] >= _BLITZ_PASS_RUSHERS,
        df["number_of_pass_rushers"].notna(),
    )
    df["def_pressure_rate_hist"] = _expanding_rate(
        df,
        "defense_coach",
        df["was_pressure"] == True,
        df["was_pressure"].notna(),  # noqa: E712
    )
    df["def_man_rate_hist"] = _expanding_rate(
        df,
        "defense_coach",
        df["defense_man_zone_type"] == "MAN_COVERAGE",
        df["defense_man_zone_type"].notna(),
    )

    return df


def _add_game_context_features(
    df: pd.DataFrame, games_df: pd.DataFrame
) -> pd.DataFrame:
    df = df.merge(games_df[_GAMES_COLS], on="game_id", how="left")

    is_home = df["posteam"] == df["home_team"]

    # spread_line is the home team's expected margin (positive = home favored);
    # reframe relative to the offense on this play, same sign convention as
    # score_differential.
    df["posteam_spread_line"] = df["spread_line"].where(is_home, -df["spread_line"])
    df["posteam_rest"] = df["home_rest"].where(is_home, df["away_rest"])
    df["defteam_rest"] = df["away_rest"].where(is_home, df["home_rest"])

    return df.drop(columns=["spread_line", "home_rest", "away_rest"])


def build_plays(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """Enriched, one-row-per-play dataset shared by every play-level model.

    Filters to legitimate down plays with a clean play_type label (run/pass/
    punt/field_goal, qb_kneel/qb_spike remapped), and adds leakage safe
    previous play, expanding coach tendency, and game context (weather,
    betting lines, rest days) columns. Retains season/week/game identifiers
    so callers can do season based train/test splits; each model's own
    module is responsible for selecting its feature subset, label, and
    casting categorical dtypes before training.
    """
    df = _filter_and_label(pbp_df)
    df = _add_previous_play_features(df)
    df = _add_coach_tendency_features(df, participation_df)
    df = _add_game_context_features(df, games_df)

    return df.reset_index(drop=True)
