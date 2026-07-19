import re
from collections import Counter, defaultdict

import pandas as pd

from ..personnel_packages import derive_personnel_package
from .teams import build_offense_depth_chart
from .utils import convert_default_dict

_SKILL_GROUPS = {"RB", "TE", "WR"}

# Depth-chart-rank prior used to smooth a player's package-conditional snap
# share: dominates when a player has near-zero real history (rookies, new
# starters, in-season promotions) and fades out as real participation
# accumulates. alpha is in units of "pseudo-plays" of prior weight -- tune
# against the backtest in the personnel/player-selection notebook.
_RANK_PRIOR = {1: 0.55, 2: 0.30, 3: 0.10}
_RANK_PRIOR_DEFAULT = 0.05
_SMOOTHING_ALPHA = 7.0

_AVAILABLE_STATUSES = {"ACT"}

_HEADCOUNT_COL_PATTERNS = {
    "rb": r"(\d+)\s*RB",
    "te": r"(\d+)\s*TE",
    "wr": r"(\d+)\s*WR",
}


_VALID_PLAY_TYPES = {"run", "pass"}


def _prepare_plays(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame,
    extra_pbp_cols: list[str] | None = None,
    extra_participation_cols: list[str] | None = None,
) -> pd.DataFrame:
    # offense_personnel is also tracked for punt/field_goal/special-teams
    # plays (confirmed on real data: e.g. "1 C, 1 DE, 3 G, 1 K, 1 LS, 1 P,
    # 3 T" parses to a spurious (0,0,0) RB/TE/WR headcount) -- restricted to
    # run/pass to match the personnel package model's own scope, since this
    # feature only ever needs to inform which skill players are on the field
    # for a run/pass snap.
    pbp = pbp_df[
        (pbp_df["down"].notna())
        & (pbp_df["season_type"] == "REG")
        & (pbp_df["play_type"].isin(_VALID_PLAY_TYPES))
    ]

    pbp_cols = ["game_id", "play_id", "season", "week", "posteam"] + (
        extra_pbp_cols or []
    )
    part_cols = [
        "nflverse_game_id",
        "play_id",
        "offense_personnel",
        "offense_players",
    ] + (extra_participation_cols or [])

    part = participation_df[part_cols].copy()
    part = derive_personnel_package(part)

    df = pbp[pbp_cols].merge(
        part,
        left_on=["game_id", "play_id"],
        right_on=["nflverse_game_id", "play_id"],
        how="inner",
    )

    return df


def _player_group_and_rank(
    depth_chart_df: pd.DataFrame, groups: set[str] = _SKILL_GROUPS
) -> pd.DataFrame:
    """(team, season, gsis_id) -> the player's most common position_group and
    depth-chart rank across weeks that season, restricted to `groups`
    (default: the skill groups RB/TE/WR that need usage-share sampling in
    player_selection -- QB/OL are resolved deterministically by rank there
    and don't need a share lookup). Callers needing a rank prior for QB too
    (e.g. rush-touch shares, where the QB is a normal scramble/designed-run
    candidate) can pass `groups=_SKILL_GROUPS | {"QB"}`.
    """
    offense_depth = build_offense_depth_chart(depth_chart_df)

    rows = []
    for team, seasons in offense_depth.items():
        for season, weeks in seasons.items():
            for _week, position_groups in weeks.items():
                for group, ranked in position_groups.items():
                    if group not in groups:
                        continue
                    for rank, gsis_id in ranked.items():
                        rows.append((team, season, gsis_id, group, rank))

    if not rows:
        return pd.DataFrame(
            columns=["team", "season", "gsis_id", "position_group", "rank"]
        )

    df = pd.DataFrame(rows, columns=["team", "season", "gsis_id", "group", "rank"])
    mode = df.groupby(["team", "season", "gsis_id"])[["group", "rank"]].agg(
        lambda s: s.mode().iat[0]
    )
    return mode.reset_index().rename(columns={"group": "position_group"})


def _latest_player_group_and_rank(player_group_rank: pd.DataFrame) -> pd.DataFrame:
    """Collapse (team, season, gsis_id) rows to one row per (team, gsis_id),
    keeping the most recent season's group/rank.

    A player's depth-chart rank can drift across seasons (call-up, injury,
    reshuffle) -- using a single "current" rank per player avoids the same
    gsis_id appearing twice in the same team/package/group share dict (which
    would silently collide when building the final dict, undercounting that
    player's true combined share) and better reflects their present role
    than blending stale past standings.
    """
    if player_group_rank.empty:
        return player_group_rank.drop(columns=["season"])

    idx = player_group_rank.groupby(["team", "gsis_id"])["season"].idxmax()
    return player_group_rank.loc[idx, ["team", "gsis_id", "position_group", "rank"]]


def build_player_package_shares(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame,
    depth_chart_df: pd.DataFrame,
    as_of_season: int | None = None,
    as_of_week: int | None = None,
) -> dict:
    """Per (team, package, position group), each rostered player's smoothed
    share of that team's skill-position snaps in that package -- the
    sampling-weight input to modeling/player_selection/predict.py.

    Mirrors modeling.plays.build_coach_snapshot's as_of_season/as_of_week
    cutoff: pass the season/week of a simulated *historical* game so the
    snapshot only reflects plays strictly before it -- otherwise a player's
    usage share would be computed using data from after the simulated game.
    Omit both only when simulating a hypothetical game beyond all available
    data.

    Returns shares[team][package][group] = {gsis_id: share, ...}, where
    shares sum to 1 across the rostered players at that team/group.
    """
    df = _prepare_plays(pbp_df, participation_df)

    if as_of_season is not None:
        before_cutoff = (df["season"] < as_of_season) | (
            (df["season"] == as_of_season) & (df["week"] < as_of_week)
        )
        df = df[before_cutoff]

        # Depth-chart rank/group needs the same cutoff -- otherwise a
        # simulated historical game could pick up a player's role from a
        # *later* season (e.g. a rookie's post-cutoff promotion to starter),
        # leaking future roster composition into the snapshot.
        dc_before_cutoff = (depth_chart_df["season"] < as_of_season) | (
            (depth_chart_df["season"] == as_of_season)
            & (depth_chart_df["week"] < as_of_week)
        )
        depth_chart_df = depth_chart_df[dc_before_cutoff]

    df = df[df["offense_personnel_package"].notna()]
    df = df.explode("offense_players").rename(columns={"offense_players": "gsis_id"})
    df = df.dropna(subset=["gsis_id"])

    player_group_rank = _player_group_and_rank(depth_chart_df)
    latest = _latest_player_group_and_rank(player_group_rank)

    df = df.merge(
        latest,
        left_on=["posteam", "gsis_id"],
        right_on=["team", "gsis_id"],
        how="inner",
    )

    counts = (
        df.groupby(
            ["posteam", "offense_personnel_package", "position_group", "gsis_id"]
        )
        .agg(n=("rank", "size"), rank=("rank", "first"))
        .reset_index()
    )
    counts["prior"] = counts["rank"].map(
        lambda r: _RANK_PRIOR.get(int(r), _RANK_PRIOR_DEFAULT)
    )
    counts["weight"] = counts["n"] + _SMOOTHING_ALPHA * counts["prior"]

    shares: defaultdict = defaultdict(lambda: defaultdict(dict))
    group_cols = ["posteam", "offense_personnel_package", "position_group"]
    for (team, package, group), rows in counts.groupby(group_cols):
        total = rows["weight"].sum()
        shares[team][package][group] = dict(
            zip(rows["gsis_id"], rows["weight"] / total)
        )

    return convert_default_dict(shares)


def build_roster_status(weekly_rosters_df: pd.DataFrame) -> dict:
    """(gsis_id, season, week) -> status string, for availability filtering
    in select_on_field_players. A missing key means no roster record exists
    yet for that week -- treat as available (can't know future inactive
    lists in advance), mirroring build_coach_snapshot's handling of
    simulating beyond available data.
    """
    return {
        (row.gsis_id, row.season, row.week): row.status
        for row in weekly_rosters_df.itertuples(index=False)
    }


def is_player_available(
    roster_status: dict, gsis_id: str, season: int, week: int
) -> bool:
    status = roster_status.get((gsis_id, season, week))
    return status is None or status in _AVAILABLE_STATUSES


def _parse_headcounts(personnel: str) -> tuple[int, int, int]:
    counts = {}
    for name, pattern in _HEADCOUNT_COL_PATTERNS.items():
        match = re.search(pattern, personnel)
        counts[name] = int(match.group(1)) if match else 0

    return counts["rb"], counts["te"], counts["wr"]


def build_other_package_headcounts(
    pbp_df: pd.DataFrame,
    participation_df: pd.DataFrame,
    as_of_season: int | None = None,
    as_of_week: int | None = None,
) -> dict:
    """Per team, the empirical distribution of real (rb, te, wr) headcounts
    among plays whose derived package is "OTHER" (no fixed headcount to
    fall back on, unlike the 8 kept packages). Falls back to a league-wide
    distribution for teams with no OTHER-package history.

    Returns {team: {(rb, te, wr): probability, ...}, ...}, plus a special
    "_league" key holding the league-wide fallback distribution.
    """
    df = _prepare_plays(pbp_df, participation_df)

    if as_of_season is not None:
        before_cutoff = (df["season"] < as_of_season) | (
            (df["season"] == as_of_season) & (df["week"] < as_of_week)
        )
        df = df[before_cutoff]

    other = df[df["offense_personnel_package"] == "OTHER"]

    league_counter: Counter = Counter()
    team_counters: defaultdict = defaultdict(Counter)

    for team, personnel in zip(other["posteam"], other["offense_personnel"]):
        headcount = _parse_headcounts(personnel)
        team_counters[team][headcount] += 1
        league_counter[headcount] += 1

    def _normalize(counter: Counter) -> dict:
        total = sum(counter.values())
        return {headcount: n / total for headcount, n in counter.items()}

    headcounts = {team: _normalize(counter) for team, counter in team_counters.items()}
    headcounts["_league"] = _normalize(league_counter)

    return headcounts
