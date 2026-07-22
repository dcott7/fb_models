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


def _other_headcount_distribution(
    other_package_headcounts: dict, *, team: str
) -> dict[tuple[int, int, int], float]:
    """P(rb, te, wr) for an "OTHER" personnel package (no fixed headcount).
    Falls back to the league-wide distribution, then a hardcoded default
    (as a degenerate one-outcome distribution) for a team with no OTHER-
    package history at all.
    """
    dist = other_package_headcounts.get(team) or other_package_headcounts.get(
        "_league", {}
    )
    if not dist:
        return {_DEFAULT_HEADCOUNT: 1.0}

    return dist


def _group_candidate_weights(
    player_package_shares: dict,
    roster_status: dict,
    *,
    team: str,
    season: int,
    week: int,
    package: str,
    group: str,
) -> dict[str, float]:
    """P(candidate fills this slot) for every depth-chart-listed player at
    `group`, restricted to players available this week and renormalized.
    Falls back to the full (untracked-availability) pool if no eligible
    player has any tracked history for this package, rather than returning
    an empty distribution.
    """
    shares = player_package_shares.get(team, {}).get(package, {}).get(group, {})
    available = {
        gsis_id: weight
        for gsis_id, weight in shares.items()
        if is_player_available(roster_status, gsis_id, season, week)
    }
    if not available:
        available = shares

    total = sum(available.values())
    if total <= 0:
        return {}

    return {gsis_id: weight / total for gsis_id, weight in available.items()}


def compute_on_field_candidates(
    *,
    team: str,
    season: int,
    week: int,
    package: str,
    player_package_shares: dict,
    offense_depth_chart: dict,
    roster_status: dict,
    other_package_headcounts: dict,
) -> dict:
    """The inputs needed to determine the 11 offensive players on the field
    for one simulated play. Pure and RNG-free -- the simulation is
    responsible for every random draw, so a single RNG stream can drive a
    whole simulated game and a backtest can score these distributions
    directly.

    QB and the 5 O-line slots are resolved deterministically by depth-chart
    rank (see _resolve_by_rank) since their count never varies by personnel
    package -- these come back as final gsis_ids, not distributions.

    RB/TE/WR counts come from the already-decided personnel package
    (play_type -> formation -> personnel -> this step) via
    personnel_packages.package_headcounts, which the simulation can call
    directly (also pure/deterministic) -- EXCEPT for the "OTHER" package,
    which has no fixed headcount; for that case this returns
    other_headcount_distribution for the simulation to draw a real
    historical (rb, te, wr) tuple from first.

    Whatever headcount is used, the *identities* to fill those RB/TE/WR
    slots should be drawn without replacement from candidates[group],
    weighted by each depth-chart-listed player's smoothed historical share
    of that team's snaps in that package (see
    features.player_usage.build_player_package_shares).

    Returns:
        {
            "QB": gsis_id | None,
            "OL": [...up to 5 gsis_ids...],
            "other_headcount_distribution": {(rb, te, wr): probability, ...}
                if package_headcounts(package) is None, else None,
            "candidates": {"RB": {gsis_id: probability, ...}, "TE": {...}, "WR": {...}},
        }
    """
    qb = _resolve_by_rank(
        offense_depth_chart=offense_depth_chart,
        roster_status=roster_status,
        team=team,
        season=season,
        week=week,
        group="QB",
    )

    ol = [
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

    other_headcount_distribution = None
    if package_headcounts(package) is None:
        other_headcount_distribution = _other_headcount_distribution(
            other_package_headcounts, team=team
        )

    candidates = {
        group: _group_candidate_weights(
            player_package_shares,
            roster_status,
            team=team,
            season=season,
            week=week,
            package=package,
            group=group,
        )
        for group in ("RB", "TE", "WR")
    }

    return {
        "QB": qb,
        "OL": ol,
        "other_headcount_distribution": other_headcount_distribution,
        "candidates": candidates,
    }
