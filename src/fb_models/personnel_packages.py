import pandas as pd

# offense_personnel has a season-boundary taxonomy problem: 2016-2022 use
# skill-position shorthand ("1 RB, 1 TE, 3 WR", ~100 unique strings/season),
# while 2023+ spells out the full 11-man lineup including O-line ("1 C, 2 G,
# 1 QB, 1 RB, 2 T, 1 TE, 3 WR", ~1,458 unique strings in 2024 alone). Parsing
# out just the RB/TE counts (present in both formats) and deriving the
# standard football-shorthand package code ("11", "12", "21", ...) gives a
# season-stable label space -- verified on real 2016-2025 data: the top 8
# packages below cover ~99% of all plays every season. Anything else buckets
# to OTHER.
PERSONNEL_KEEP_PACKAGES = {"11", "12", "21", "13", "22", "01", "10", "02"}


def derive_personnel_package(df: pd.DataFrame) -> pd.DataFrame:
    has_personnel = df["offense_personnel"].notna()

    rb = df["offense_personnel"].str.extract(r"(\d+)\s*RB")[0].fillna("0")
    te = df["offense_personnel"].str.extract(r"(\d+)\s*TE")[0].fillna("0")
    package = rb + te

    df["offense_personnel_package"] = package.where(
        package.isin(PERSONNEL_KEEP_PACKAGES), "OTHER"
    )
    df.loc[~has_personnel, "offense_personnel_package"] = None

    return df


def package_headcounts(package: str) -> tuple[int, int, int] | None:
    """(rb, te, wr) implied by a package code, or None for "OTHER"/unknown.

    wr is derived as 5 - rb - te since a standard personnel package has 5
    non-O-line, non-QB skill players split among RB/TE/WR.
    """
    if package not in PERSONNEL_KEEP_PACKAGES:
        return None

    rb, te = int(package[0]), int(package[1])
    return rb, te, 5 - rb - te
