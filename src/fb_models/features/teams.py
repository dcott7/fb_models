from collections import defaultdict

import pandas as pd

from .utils import convert_default_dict


def build_teams(depth_chart: pd.DataFrame) -> dict:
    # {team: {season: {week: {formation: {depth_position: {depth_team: gsis_id}}}}}}
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
