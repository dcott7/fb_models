from collections import defaultdict

import pandas as pd

from .utils import convert_default_dict


def build_teams(depth_chart: pd.DataFrame) -> dict:
    teams: defaultdict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    grouped = depth_chart.groupby(
        ["club_code", "season", "week", "formation", "depth_position"]
    )

    for (team, season, week, formation, depth_position), pos_data in grouped:
        teams[team][season][week][formation][depth_position] = dict(
            zip(pos_data["depth_team"], pos_data["gsis_id"])
        )

    return convert_default_dict(teams)


# depth_chart's own "position" column is the clean field (WR/RB/FB/TE/QB/T/G/C
# plus <0.5% defensive/special-teams noise on offense rows) -- "depth_position"
# is dirtier (whitespace garbage, WR1/WR2/LWR variants) but still useful for
# telling apart the 5 distinct O-line slots, since "position" only says T/G/C.
# FB is folded into RB: the personnel-package regex in personnel_packages.py
# only counts literal "RB" tags, so a package's "RB slot(s)" can be filled by
# either an RB or FB body.
_OFFENSE_POSITION_GROUPS = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "TE": "TE",
    "WR": "WR",
    "T": "OL",
    "G": "OL",
    "C": "OL",
}

_OL_SLOT_MAP = {
    "LT": "LT",
    "LOT": "LT",
    "RT": "RT",
    "ROT": "RT",
    "LG": "LG",
    "RG": "RG",
    "C": "C",
    "OC": "C",
}


def build_offense_depth_chart(depth_chart: pd.DataFrame) -> dict:
    """Depth chart restricted to offense, bucketed to {QB, RB, TE, WR, LT,
    LG, C, RG, RT}, keyed for rank-based lookup at simulation time.

    Returns depth[club_code][season][week][group][depth_team_rank] = gsis_id.
    """
    df = depth_chart[depth_chart["formation"] == "Offense"].copy()

    df["position_group"] = df["position"].map(_OFFENSE_POSITION_GROUPS)
    df = df[df["position_group"].notna()]

    is_ol = df["position_group"] == "OL"
    df.loc[is_ol, "position_group"] = df.loc[is_ol, "depth_position"].map(_OL_SLOT_MAP)
    df = df[df["position_group"].notna()]

    df["depth_team"] = df["depth_team"].astype(int)

    depth: defaultdict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    grouped = df.groupby(["club_code", "season", "week", "position_group"])
    for (team, season, week, group), pos_data in grouped:
        depth[team][season][week][group] = dict(
            zip(pos_data["depth_team"], pos_data["gsis_id"])
        )

    return convert_default_dict(depth)
