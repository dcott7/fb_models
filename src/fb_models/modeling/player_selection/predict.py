import numpy as np

from ...features.player_usage import is_player_available
from ...personnel_packages import package_headcounts

_OL_GROUPS = ["LT", "LG", "C", "RG", "RT"]

# last-resort default if a team has no OTHER-package history at all AND no
# league-wide fallback is available (e.g. a brand-new/expansion team with no
# history whatsoever) -- standard 11-personnel shape.
_DEFAULT_HEADCOUNT = (1, 1, 3)


def _resolve_by_rank(
    *,
    offense_depth_chart: dict,
    roster_status: dict,
    team: str,
    season: int,
    week: int,
    group: str,
) -> str | None:
    """Depth-chart rank1 -> rank2 -> rank3... fallback, skipping any player
    whose roster status marks them unavailable that week. Used for QB and
    the 5 O-line slots, which are resolved deterministically rather than
    sampled: their count never varies by personnel package, and in-season
    starter changes are captured automatically since rank is looked up per
    simulated week rather than once per season.

    Falls back to the most recent known depth-chart week for the team if
    none exists yet for the exact target week (future/hypothetical
    simulation), then to the most recent prior season, mirroring
    build_coach_snapshot's "nothing to exclude" handling of simulating
    beyond available data.
    """
    team_depth = offense_depth_chart.get(team, {})
    season_depth = team_depth.get(season, {})

    week_ranks = season_depth.get(week)
    if week_ranks is None:
        candidate_weeks = sorted(w for w in season_depth if w <= week)
        if candidate_weeks:
            week_ranks = season_depth[candidate_weeks[-1]]
        else:
            prior_seasons = sorted(s for s in team_depth if s < season)
            if not prior_seasons:
                return None
            latest_season = team_depth[prior_seasons[-1]]
            week_ranks = latest_season[max(latest_season)]

    ranks = week_ranks.get(group, {})
    for rank in sorted(ranks):
        gsis_id = ranks[rank]
        if is_player_available(roster_status, gsis_id, season, week):
            return gsis_id

    return None


def _sample_other_headcount(
    other_package_headcounts: dict, *, team: str, rng: np.random.Generator
) -> tuple[int, int, int]:
    dist = other_package_headcounts.get(team) or other_package_headcounts.get(
        "_league", {}
    )
    if not dist:
        return _DEFAULT_HEADCOUNT

    headcounts = list(dist.keys())
    probs = np.array(list(dist.values()), dtype=float)
    idx = rng.choice(len(headcounts), p=probs / probs.sum())
    return headcounts[idx]


def _sample_group(
    *,
    player_package_shares: dict,
    roster_status: dict,
    team: str,
    season: int,
    week: int,
    package: str,
    group: str,
    count: int,
    rng: np.random.Generator,
) -> list[str]:
    if count <= 0:
        return []

    shares = player_package_shares.get(team, {}).get(package, {}).get(group, {})
    available = {
        gsis_id: weight
        for gsis_id, weight in shares.items()
        if is_player_available(roster_status, gsis_id, season, week)
    }
    if not available:
        # No eligible player with tracked history for this package -- fall
        # back to the full (untracked-availability) pool rather than
        # leaving the slot empty.
        available = shares

    if not available:
        return []

    candidates = list(available.keys())
    weights = np.array(list(available.values()), dtype=float)
    weights = weights / weights.sum()

    n = min(count, len(candidates))
    chosen = rng.choice(candidates, size=n, replace=False, p=weights)
    return list(chosen)


def select_on_field_players(
    *,
    team: str,
    season: int,
    week: int,
    package: str,
    player_package_shares: dict,
    offense_depth_chart: dict,
    roster_status: dict,
    other_package_headcounts: dict,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """Sample the 11 offensive players on the field for one simulated play.

    QB and the 5 O-line slots are resolved deterministically by depth-chart
    rank (see _resolve_by_rank) since their count never varies by personnel
    package. RB/TE/WR counts come from the already-decided personnel
    package (play_type -> formation -> personnel -> this step); their
    *identities* are sampled without replacement, weighted by each
    depth-chart-listed player's smoothed historical share of that team's
    snaps in that package (see features.player_usage.build_player_package_shares).

    For "OTHER" packages (no fixed headcount), draws a real historical
    (rb, te, wr) headcount tuple from the team's own OTHER-labeled plays
    (falling back to the league-wide distribution, then a hardcoded default
    for teams with no history at all) before sampling players for it.

    Returns {"QB": [...1...], "OL": [...5...], "RB": [...], "TE": [...],
    "WR": [...]}.
    """
    on_field: dict[str, list[str]] = {}

    qb = _resolve_by_rank(
        offense_depth_chart=offense_depth_chart,
        roster_status=roster_status,
        team=team,
        season=season,
        week=week,
        group="QB",
    )
    on_field["QB"] = [qb] if qb is not None else []

    on_field["OL"] = [
        gsis_id
        for group in _OL_GROUPS
        if (
            gsis_id := _resolve_by_rank(
                offense_depth_chart=offense_depth_chart,
                roster_status=roster_status,
                team=team,
                season=season,
                week=week,
                group=group,
            )
        )
        is not None
    ]

    headcounts = package_headcounts(package)
    if headcounts is None:
        headcounts = _sample_other_headcount(
            other_package_headcounts, team=team, rng=rng
        )
    rb, te, wr = headcounts

    for group, count in [("RB", rb), ("TE", te), ("WR", wr)]:
        on_field[group] = _sample_group(
            player_package_shares=player_package_shares,
            roster_status=roster_status,
            team=team,
            season=season,
            week=week,
            package=package,
            group=group,
            count=count,
            rng=rng,
        )

    return on_field
