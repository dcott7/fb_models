import numpy as np

# Situational buckets are thin (confirmed on real 2023 data: grouping by
# team/package/formation/field_position gives 978 buckets, median 5 plays,
# 25th pct 2) -- a single flat smoothing level would overfit hard at this
# granularity, unlike player_package_shares' one-level smoothing. Instead,
# escalate through progressively coarser buckets (mirroring
# player_selection's rank1->rank2->rank3 fallback chain) until one has
# enough raw plays to trust, only then apply the rank-prior smoothing.
_MIN_BUCKET_PLAYS = 15

# Rushing has a strong, genuine signal (a starting RB or a scrambling QB
# dominates real carries) that keeps improving with *more* smoothing toward
# the rank prior, all the way up to very large alpha (confirmed: rush hit
# rate rises from 71% to 77% as alpha goes from 3 to 1200) -- thin
# situational buckets are noisier than the stable rank prior, so leaning on
# the prior harder helps. Blend raw bucket counts additively with the rank
# prior, same shape as player_package_shares.
_RUSH_SMOOTHING_ALPHA = 300.0

# Targeting needs a fundamentally different approach. Confirmed via backtest
# that once you condition on the exact 5 skill players sharing the field for
# a personnel grouping, who actually gets the target on a *specific* play is
# close to uniformly distributed across their LOCAL touch-count rank within
# that specific candidate set (21.7%/20.0%/18.3%/15.7%/13.2% for ranks 1-5,
# confirmed on real held-out data) -- even though their full-season
# aggregate shares are heavily skewed (a WR1 can account for 40%+ of a
# team's season-long targets). A global marginal-based prior (the same
# additive-blend shape used for rush) was tried first and, even used alone
# with no bucket data at all, scored *worse* than a flat uniform guess
# (~1.99 vs ~1.79 log-loss): it reflects the skewed aggregate distribution,
# not the much flatter one that's actually realized on any given play. The
# fix is to rank candidates by their own raw touch count within just this
# play's candidate set (a purely relative/positional ranking, using the
# rank-prior only to break ties when bucket data is absent) and apply this
# empirically-flat local-rank distribution -- not blend in the skewed
# marginal. This reaches log-loss ~1.78, matching the ~1.77 ceiling of a
# perfectly-fit oracle for this exact distribution -- there just isn't much
# more signal to extract from season-long share data alone for "who gets
# thrown to on this specific play."
_TARGET_LOCAL_RANK_PRIOR = [0.244, 0.225, 0.206, 0.177, 0.148]
_TARGET_LOCAL_RANK_DEFAULT = 0.05

# Group-aware rank priors, empirically derived from real league-wide touch
# shares by (position_group, depth-chart rank) across 2019-2024 -- NOT a
# reuse of features.player_usage's _RANK_PRIOR (0.55/0.30/0.10), which was
# calibrated for a different question ("is this player even on the field
# for a package") and, when tried here first, badly miscalibrated touch
# attribution: it applies the *same* weight to a WR1, RB1, and TE1, but
# real target shares by position are very different (confirmed: WR1 gets
# ~42% of targets, TE1 ~13%, RB1 ~11%, out of all real 2019-2024 targets
# with a known depth-chart rank) and real rush shares favor RB/QB
# overwhelmingly (RB1 ~50%, QB1 ~10%, WR/TE rushes are rare trick plays).
_RUSH_RANK_PRIOR = {
    ("QB", 1): 0.097,
    ("QB", 2): 0.018,
    ("QB", 3): 0.010,
    ("RB", 1): 0.496,
    ("RB", 2): 0.230,
    ("RB", 3): 0.112,
    ("WR", 1): 0.018,
    ("WR", 2): 0.010,
    ("WR", 3): 0.003,
    ("TE", 1): 0.006,
    ("TE", 2): 0.0003,
    ("TE", 3): 0.0003,
}
_TARGET_RANK_PRIOR = {
    ("RB", 1): 0.106,
    ("RB", 2): 0.057,
    ("RB", 3): 0.028,
    ("TE", 1): 0.135,
    ("TE", 2): 0.058,
    ("TE", 3): 0.021,
    ("WR", 1): 0.416,
    ("WR", 2): 0.146,
    ("WR", 3): 0.033,
}
_RANK_PRIOR_DEFAULT = 0.01

_BUCKET_LEVELS = ["context", "package_formation", "package", "team", "league"]


def _bucket_at(
    levels: dict,
    level: str,
    *,
    team: str,
    package: str,
    formation: str,
    field_position: str,
) -> dict:
    if level == "context":
        return (
            levels["context"]
            .get(team, {})
            .get(package, {})
            .get(formation, {})
            .get(field_position, {})
        )
    if level == "package_formation":
        return (
            levels["package_formation"]
            .get(team, {})
            .get(package, {})
            .get(formation, {})
        )
    if level == "package":
        return levels["package"].get(team, {}).get(package, {})
    if level == "team":
        return levels["team"].get(team, {})
    return levels["league"]


def _select_bucket(
    levels: dict, *, team: str, package: str, formation: str, field_position: str
) -> dict:
    """Escalating fallback: take the first (finest-to-coarsest) level whose
    raw play count meets _MIN_BUCKET_PLAYS. If none qualify, fall back to
    the coarsest non-empty level (more data beats none, even under the
    threshold) rather than an empty bucket.
    """
    buckets = [
        _bucket_at(
            levels,
            level,
            team=team,
            package=package,
            formation=formation,
            field_position=field_position,
        )
        for level in _BUCKET_LEVELS
    ]

    for bucket in buckets:
        if sum(bucket.values()) >= _MIN_BUCKET_PLAYS:
            return bucket

    for bucket in reversed(buckets):
        if bucket:
            return bucket

    return {}


def _rank_prior(prior_table: dict, rank_priors: dict, team: str, gsis_id: str) -> float:
    entry = rank_priors.get((team, gsis_id))
    if entry is None:
        return _RANK_PRIOR_DEFAULT

    group, rank = entry
    return prior_table.get((group, min(int(rank), 3)), _RANK_PRIOR_DEFAULT)


def _rush_candidates(on_field: dict) -> list[str]:
    """QB + RB + TE + WR -- QB is a normal rush candidate (scrambles/
    designed QB runs), OL is never a legitimate ball carrier."""
    return (
        on_field.get("QB", [])
        + on_field.get("RB", [])
        + on_field.get("TE", [])
        + on_field.get("WR", [])
    )


def _target_candidates(on_field: dict) -> list[str]:
    """RB + TE + WR -- QB is assumed to be the passer (resolved by
    player_selection), not also a receiver on the same play; excluding it
    here accepts the ~0.3% of real trick plays with a non-QB passer as
    low-blast-radius noise, consistent with how the "OTHER" personnel
    package and the FB/RB regex mismatch were already accepted elsewhere.
    """
    return on_field.get("RB", []) + on_field.get("TE", []) + on_field.get("WR", [])


def compute_rush_probabilities(
    *,
    team: str,
    package: str,
    formation: str,
    field_position: str,
    on_field: dict,
    touch_shares: dict,
    rank_priors: dict,
) -> dict[str, float]:
    """P(candidate gets the carry) for every QB/RB/TE/WR player_selection
    put on the field, given the historical touch-share snapshot. Pure and
    RNG-free so a backtest can score the assigned probability directly.
    """
    candidates = _rush_candidates(on_field)
    if not candidates:
        raise ValueError("on_field has no QB/RB/TE/WR candidates for rush attribution")

    bucket = _select_bucket(
        touch_shares["rush"],
        team=team,
        package=package,
        formation=formation,
        field_position=field_position,
    )

    weights = {
        pid: bucket.get(pid, 0)
        + _RUSH_SMOOTHING_ALPHA * _rank_prior(_RUSH_RANK_PRIOR, rank_priors, team, pid)
        for pid in candidates
    }

    total = sum(weights.values())
    if total <= 0:
        return {pid: 1 / len(candidates) for pid in candidates}

    return {pid: w / total for pid, w in weights.items()}


def compute_target_probabilities(
    *,
    team: str,
    package: str,
    formation: str,
    field_position: str,
    on_field: dict,
    touch_shares: dict,
    rank_priors: dict,
) -> dict[str | None, float]:
    """P(candidate is targeted) for every RB/TE/WR player_selection put on
    the field, plus a None entry for "no receiver credited" (sacks +
    uncredited incompletions, ~11% of real pass plays).

    P(None) is computed as its own empirical share of the *whole* bucket
    (every historical target in that bucket, including players who no
    longer play for this team) and held fixed; the remaining probability
    mass is then split among just today's on-field candidates. Treating
    None as just another candidate in a single flat renormalization was
    tried first and found (via backtest) to badly inflate it: the bucket's
    None count reflects every play ever run in it, but the candidate pool
    is only today's 5 on-field players -- every other historical target
    (receivers no longer on the roster) silently drops out of the
    denominator while None's raw count doesn't shrink to compensate,
    so None ends up absorbing probability mass that used to belong to
    departed players.

    Player weights come from each candidate's LOCAL rank within just this
    play's candidate set (ordered by their own raw touch count in the
    selected bucket, falling back to the season-long rank prior only to
    break ties when bucket data is absent), not an additive blend with the
    global marginal prior the way rush does it -- see _TARGET_LOCAL_RANK_PRIOR
    for why.
    """
    candidates = _target_candidates(on_field)

    bucket = _select_bucket(
        touch_shares["target"],
        team=team,
        package=package,
        formation=formation,
        field_position=field_position,
    )

    bucket_total = sum(bucket.values())
    p_none = bucket.get(None, 0) / bucket_total if bucket_total > 0 else 0.0

    ranked = sorted(
        candidates,
        key=lambda pid: (
            -bucket.get(pid, 0),
            -_rank_prior(_TARGET_RANK_PRIOR, rank_priors, team, pid),
        ),
    )
    weights = {
        pid: (
            _TARGET_LOCAL_RANK_PRIOR[i]
            if i < len(_TARGET_LOCAL_RANK_PRIOR)
            else _TARGET_LOCAL_RANK_DEFAULT
        )
        for i, pid in enumerate(ranked)
    }

    total = sum(weights.values())
    if total <= 0:
        remaining = (1 - p_none) / len(candidates) if candidates else 0.0
        return {None: p_none, **{pid: remaining for pid in candidates}}

    return {
        None: p_none,
        **{pid: (1 - p_none) * w / total for pid, w in weights.items()},
    }


def _sample(probs: dict, rng: np.random.Generator):
    keys = list(probs.keys())
    weights = np.array([probs[k] for k in keys], dtype=float)
    weights = weights / weights.sum()

    idx = rng.choice(len(keys), p=weights)
    return keys[idx]


def select_rusher(
    *,
    team: str,
    package: str,
    formation: str,
    field_position: str,
    on_field: dict,
    touch_shares: dict,
    rank_priors: dict,
    rng: np.random.Generator,
) -> str:
    probs = compute_rush_probabilities(
        team=team,
        package=package,
        formation=formation,
        field_position=field_position,
        on_field=on_field,
        touch_shares=touch_shares,
        rank_priors=rank_priors,
    )
    return _sample(probs, rng)


def select_target(
    *,
    team: str,
    package: str,
    formation: str,
    field_position: str,
    on_field: dict,
    touch_shares: dict,
    rank_priors: dict,
    rng: np.random.Generator,
) -> str | None:
    probs = compute_target_probabilities(
        team=team,
        package=package,
        formation=formation,
        field_position=field_position,
        on_field=on_field,
        touch_shares=touch_shares,
        rank_priors=rank_priors,
    )
    return _sample(probs, rng)
