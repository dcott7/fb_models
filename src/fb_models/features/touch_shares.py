from collections import defaultdict

import pandas as pd

from ..formations import normalize_offense_formation
from .player_usage import (
    _SKILL_GROUPS,
    _latest_player_group_and_rank,
    _player_group_and_rank,
    _prepare_plays,
)

# QB is a normal rush-touch candidate (scrambles/designed QB runs are folded
# into play_type=="run" -- confirmed ~7.4% of 2023 run plays), unlike
# player_usage.py's rank-prior scope which only needs RB/TE/WR since QB is
# resolved deterministically in player_selection.
_TOUCH_RANK_GROUPS = _SKILL_GROUPS | {"QB"}

# "no receiver credited" -- sacks (no target is possible) plus genuinely
# uncredited incompletions/throwaways, ~11% of real pass plays. A sentinel
# string lets it flow through the same groupby/count machinery as a player
# id; swapped back to None (the public-facing value) once counting is done.
_NONE_SENTINEL = "__NO_TARGET__"


def _prepare_touch_plays(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame,
    as_of_season: int | None = None,
    as_of_week: int | None = None,
) -> pd.DataFrame:
    df = _prepare_plays(
        pbp_df,
        participation_df,
        extra_pbp_cols=[
            "play_type",
            "field_position",
            "rusher_player_id",
            "receiver_player_id",
        ],
        extra_participation_cols=["offense_formation"],
    )
    df = normalize_offense_formation(df)

    if as_of_season is not None:
        before_cutoff = (df["season"] < as_of_season) | (
            (df["season"] == as_of_season) & (df["week"] < as_of_week)
        )
        df = df[before_cutoff]

    return df[df["offense_personnel_package"].notna() & df["offense_formation"].notna()]


def _nested_counts(df: pd.DataFrame, group_cols: list[str], outcome_col: str) -> dict:
    """Raw counts nested len(group_cols) levels deep, with the innermost
    dict keyed by outcome_col value. E.g. group_cols=["team","package"]
    gives {team: {package: {outcome: n}}}.
    """
    if not group_cols:
        return df[outcome_col].value_counts().to_dict()

    counts = df.groupby(group_cols + [outcome_col]).size()

    nested: dict = {}
    for key, n in counts.items():
        *path, outcome = key
        node = nested
        for part in path:
            node = node.setdefault(part, {})
        node[outcome] = int(n)

    return nested


def _replace_sentinel_keys(node: dict) -> dict:
    sample = next(iter(node.values()), None)
    if isinstance(sample, dict):
        for child in node.values():
            _replace_sentinel_keys(child)
    elif _NONE_SENTINEL in node:
        node[None] = node.pop(_NONE_SENTINEL)

    return node


_CONTEXT_COLS = [
    "posteam",
    "offense_personnel_package",
    "offense_formation",
    "field_position",
]
_PACKAGE_FORMATION_COLS = ["posteam", "offense_personnel_package", "offense_formation"]
_PACKAGE_COLS = ["posteam", "offense_personnel_package"]
_TEAM_COLS = ["posteam"]


def _build_levels(df: pd.DataFrame, outcome_col: str) -> dict:
    return {
        "context": _nested_counts(df, _CONTEXT_COLS, outcome_col),
        "package_formation": _nested_counts(df, _PACKAGE_FORMATION_COLS, outcome_col),
        "package": _nested_counts(df, _PACKAGE_COLS, outcome_col),
        "team": _nested_counts(df, _TEAM_COLS, outcome_col),
        "league": _nested_counts(df, [], outcome_col),
    }


def build_touch_shares(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame,
    as_of_season: int | None = None,
    as_of_week: int | None = None,
) -> dict:
    """Raw (unsmoothed) touch counts for who gets the ball, the input to
    modeling/usage/predict.py's sampling functions.

    Returns {"rush": <levels>, "target": <levels>}, where <levels> is
    {"context": team -> package -> formation -> field_position -> {gsis_id: n},
     "package_formation": team -> package -> formation -> {gsis_id: n},
     "package": team -> package -> {gsis_id: n},
     "team": team -> {gsis_id: n},
     "league": {gsis_id: n}}.

    "target"'s dicts additionally carry a None key for the "no receiver
    credited" outcome (sacks + uncredited incompletions, ~11% of real pass
    plays). "rush" has no such key -- rusher_player_id is ~100% populated
    on run plays.

    These are intentionally raw counts, not shares: modeling/usage/predict.py
    renormalizes over exactly the on-field candidates player_selection chose
    for a given simulated play (which is always a small subset of everyone
    who has ever touched the ball in a bucket), with an escalating fallback
    across the levels above when the finest bucket is too thin -- confirmed
    on real data that (team, package, formation, field_position) buckets are
    thin (978 buckets in a single season, median 5 plays, 25th pct 2).

    Mirrors build_player_package_shares's as_of_season/as_of_week cutoff:
    pass the season/week of a simulated historical game so the snapshot
    only reflects plays strictly before it.
    """
    df = _prepare_touch_plays(
        pbp_df, participation_df, as_of_season=as_of_season, as_of_week=as_of_week
    )

    rush_df = df[(df["play_type"] == "run") & df["rusher_player_id"].notna()]
    rush_levels = _build_levels(rush_df, "rusher_player_id")

    pass_df = df[df["play_type"] == "pass"].copy()
    pass_df["receiver_player_id"] = pass_df["receiver_player_id"].fillna(_NONE_SENTINEL)
    target_levels = _build_levels(pass_df, "receiver_player_id")
    for level in target_levels.values():
        _replace_sentinel_keys(level)

    return {"rush": rush_levels, "target": target_levels}


def player_rank_priors(
    depth_chart_df: pd.DataFrame,
    touch_shares: dict,
    as_of_season: int | None = None,
    as_of_week: int | None = None,
) -> dict:
    """(team, gsis_id) -> (position_group, usage_rank), including QB (unlike
    features.player_usage's default skill-groups-only scope) since the QB
    is a normal rush-touch candidate here.

    usage_rank is derived by ranking each team's depth-chart-listed players
    within their position group by total historical touches (rush attempts
    + targets from `touch_shares`) -- NOT nflverse's own `depth_team`
    field. Confirmed on real data that `depth_team` is numbered per
    *slot* (e.g. separate X/Z/slot receiver spots), not a single ordered
    depth chart: PHI's 2024 depth chart tags A.J. Brown, DeVonta Smith,
    AND Jahan Dotson all `depth_team==1` simultaneously, despite wildly
    different real target volumes (632 vs 398 vs 158 in the backtest
    window) -- a rank-based prior keyed off the raw field can't tell them
    apart and (confirmed via backtest) badly miscalibrates touch
    attribution as a result. Ranking by actual usage instead fixes this.

    Same as_of cutoff as build_touch_shares -- pass the matching
    season/week so a historical snapshot can't see a player's later-season
    role, and build `touch_shares` with the same cutoff so the usage
    ranking doesn't leak future data either.
    """
    if as_of_season is not None:
        before_cutoff = (depth_chart_df["season"] < as_of_season) | (
            (depth_chart_df["season"] == as_of_season)
            & (depth_chart_df["week"] < as_of_week)
        )
        depth_chart_df = depth_chart_df[before_cutoff]

    player_group_rank = _player_group_and_rank(
        depth_chart_df, groups=_TOUCH_RANK_GROUPS
    )
    latest = _latest_player_group_and_rank(player_group_rank)

    total_touches: dict = defaultdict(int)
    for outcome_type in ("rush", "target"):
        for gsis_id, n in touch_shares[outcome_type]["league"].items():
            if gsis_id is not None:
                total_touches[gsis_id] += n

    groups: dict = defaultdict(list)
    for row in latest.itertuples(index=False):
        groups[(row.team, row.position_group)].append(row.gsis_id)

    priors = {}
    for (team, group), gsis_ids in groups.items():
        ranked = sorted(gsis_ids, key=lambda pid: -total_touches.get(pid, 0))
        for usage_rank, gsis_id in enumerate(ranked, start=1):
            priors[(team, gsis_id)] = (group, usage_rank)

    return priors
