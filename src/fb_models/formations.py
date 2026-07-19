import pandas as pd

# nflverse changed its offense_formation tracking methodology starting in
# 2023: 2019-2022 break "under center" into SINGLEBACK/I_FORM/JUMBO/WILDCAT
# and separately track EMPTY (empty backfield -- confirmed 98.5% shotgun
# snaps on real data, so it's a shotgun-family look, not under center),
# while 2023+ collapses all the true under-center variants into a single
# UNDER CENTER category and no longer breaks out EMPTY at all. Remapping
# the old labels keeps the label space consistent with the current (and
# presumably future) tracking convention across all seasons, rather than
# training on two incompatible taxonomies that happen to share a column
# name.
FORMATION_REMAP = {
    "SINGLEBACK": "UNDER CENTER",
    "I_FORM": "UNDER CENTER",
    "JUMBO": "UNDER CENTER",
    "WILDCAT": "UNDER CENTER",
    "EMPTY": "SHOTGUN",
}


def normalize_offense_formation(df: pd.DataFrame) -> pd.DataFrame:
    df["offense_formation"] = df["offense_formation"].replace(FORMATION_REMAP)
    return df
