from pathlib import Path

import pandas as pd
import requests

from .config import RAW_DATA_DIR

__all__ = [
    "load_pbp_dataset",
    "load_participation_dataset",
    "load_players_dataset",
    "load_weekly_rosters_dataset",
    "load_games_dataset",
]

DATA_DIR = RAW_DATA_DIR / "nflverse"


_PLAYERS_RETAIN_COLS: list[str] = [
    "gsis_id",
    "display_name",
    "position_group",
    "position",
    "status",
]

_WEEKLY_ROSTERS_RETAIN_COLS: list[str] = [
    "gsis_id",
    "season",
    "week",
    "team",
    "status",
]

_PBP_RETAIN_COLS: list[str] = [
    "season_type",
    "season",
    "game_id",
    "play_id",
    "down",
    "yardline_100",
    "rush",
    "pass",
    "rush_attempt",
    "pass_attempt",
    "rusher_player_id",
    "receiver_player_id",
    "play_type",
    "ydstogo",
    "score_differential",
    "qtr",
    "game_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "goal_to_go",
    "home_coach",
    "away_coach",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "half_seconds_remaining",
    "shotgun",
    "no_huddle",
    "fourth_down_failed",
    "drive",
    "drive_play_count",
    "drive_first_downs",
    "week",
    "order_sequence",
    "yards_gained",
    "first_down",
    "interception",
    "fumble_lost",
]

_PARTICIPATION_RETAIN_COLS: list[str] = [
    "nflverse_game_id",
    "play_id",
    "possession_team",
    "n_offense",
    "n_defense",
    "offense_formation",
    "offense_personnel",
    "defense_personnel",
    "offense_players",
    "defense_players",
    "defense_man_zone_type",
    "was_pressure",
    "number_of_pass_rushers",
]

_GAMES_RETAIN_COL: list[str] = [
    "game_id",
    "season",
    "game_type",
    "week",
    "away_team",
    "home_team",
    "location",
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
    "away_coach",
    "home_coach",
]


_RED_ZONE_YARDLINE = 20
_GOAL_LINE_YARDLINE = 5
_NEUTRAL_SCORE_MARGIN = 8
_LATE_GAME_SECONDS = 300
_TWO_MINUTE_SECONDS = 120
_FOUR_MINUTE_SECONDS = 240


class NflVerseRepo:
    BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_or_download(self, filename: str, url: str) -> Path:
        file_path = self.data_dir / filename

        if file_path.exists():
            return file_path

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            print(f"Downloaded {filename}")

            return file_path

        except requests.RequestException as e:
            raise RuntimeError(f"Unable to retrieve {filename}: {e}")

    def get_pbp(self, season: int) -> Path:
        filename = f"pbp_{season}.parquet"
        url = f"{self.BASE_URL}/pbp/play_by_play_{season}.parquet"
        return self._get_or_download(filename, url)

    def get_participation(self, season: int) -> Path:
        filename = f"participation_{season}.parquet"
        url = f"{self.BASE_URL}/pbp_participation/pbp_participation_{season}.parquet"
        return self._get_or_download(filename, url)

    def get_players(self) -> Path:
        filename = "players.parquet"
        url = f"{self.BASE_URL}/players/players.parquet"
        return self._get_or_download(filename, url)

    def get_weekly_rosters(self, season: int) -> Path:
        filename = f"roster_weekly_{season}.parquet"
        url = f"{self.BASE_URL}/weekly_rosters/roster_weekly_{season}.parquet"
        return self._get_or_download(filename, url)

    def get_games(self) -> Path:
        filename = "games.csv"
        url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
        return self._get_or_download(filename, url)


def _load_pbp(seasons: list[int]) -> pd.DataFrame:
    repo = NflVerseRepo()
    dfs = [pd.read_parquet(repo.get_pbp(season)) for season in seasons]
    return pd.concat(dfs, ignore_index=True)


def _load_participation(seasons: list[int]) -> pd.DataFrame:
    repo = NflVerseRepo()
    dfs = [pd.read_parquet(repo.get_participation(season)) for season in seasons]
    return pd.concat(dfs, ignore_index=True)


def _load_weekly_rosters(seasons: list[int]) -> pd.DataFrame:
    repo = NflVerseRepo()
    dfs = [pd.read_parquet(repo.get_weekly_rosters(season)) for season in seasons]
    return pd.concat(dfs, ignore_index=True)


def _load_players() -> pd.DataFrame:
    repo = NflVerseRepo()
    return pd.read_parquet(repo.get_players())


def _load_games() -> pd.DataFrame:
    repo = NflVerseRepo()
    return pd.read_csv(repo.get_games())


def _add_derived_columns_pbp(df: pd.DataFrame) -> pd.DataFrame:
    df["red_zone"] = df["yardline_100"] <= _RED_ZONE_YARDLINE
    df["goal_line"] = df["yardline_100"] <= _GOAL_LINE_YARDLINE
    df["field_position"] = "normal"
    df.loc[df["red_zone"], "field_position"] = "red_zone"
    df.loc[df["goal_line"], "field_position"] = "goal_line"

    df["leading"] = df["score_differential"] > 0
    df["trailing"] = df["score_differential"] < 0
    df["one_score_game"] = df["score_differential"].abs() <= _NEUTRAL_SCORE_MARGIN
    df["neutral_script"] = df["one_score_game"] & ~(
        (df["qtr"] == 4) & (df["game_seconds_remaining"] <= _LATE_GAME_SECONDS)
    )
    df["score_margin_bucket"] = pd.cut(
        df["score_differential"],
        bins=[-100, -14, -7, 0, 7, 14, 100],
        labels=[
            "trailing_big",
            "trailing_small",
            "tied_or_close",
            "leading_small",
            "leading_big",
            "blowout",
        ],
    )

    df["two_minute"] = df["half_seconds_remaining"] <= _TWO_MINUTE_SECONDS
    df["four_minute"] = (
        (df["qtr"] == 4)
        & (df["game_seconds_remaining"] <= _FOUR_MINUTE_SECONDS)
        & (df["score_differential"] > 0)
    )

    df["early_down"] = df["down"].isin([1, 2])
    df["passing_down"] = ((df["down"] == 3) & (df["ydstogo"] >= 7)) | (
        (df["down"] == 4) & (df["ydstogo"] >= 5)
    )
    df["short_yardage"] = df["ydstogo"] <= 2
    df["yardage_bucket"] = pd.cut(
        df["ydstogo"],
        bins=[0, 2, 6, 10, 100],
        labels=["short", "medium", "long", "very_long"],
        include_lowest=True,
    )

    df["is_pass"] = df["play_type"] == "pass"
    df["is_rush"] = df["play_type"] == "run"
    df["is_go_for_it"] = df["play_type"].isin(["run", "pass"])

    df["offense_coach"] = df["home_coach"].where(
        df["posteam"] == df["home_team"], df["away_coach"]
    )
    df["defense_coach"] = df["home_coach"].where(
        df["defteam"] == df["home_team"], df["away_coach"]
    )

    return df


def _add_derived_columns_participation(df: pd.DataFrame) -> pd.DataFrame:
    df["season"] = df["nflverse_game_id"].str.extract(r"(\d{4})").astype(int)

    df[["team_1", "team_2"]] = df["nflverse_game_id"].str.extract(
        r"\d{4}_\d+_(\w+)_(\w+)"
    )
    df["defending_team"] = df["team_2"].where(
        df["possession_team"] == df["team_1"], df["team_1"]
    )

    df["offense_players"] = df["offense_players"].str.split(";")
    df["defense_players"] = df["defense_players"].str.split(";")

    return df


def load_participation_dataset(seasons: list[int]) -> pd.DataFrame:
    df = _load_participation(seasons=seasons)[_PARTICIPATION_RETAIN_COLS].copy()
    return _add_derived_columns_participation(df)


def load_pbp_dataset(seasons: list[int]) -> pd.DataFrame:
    df = _load_pbp(seasons=seasons)[_PBP_RETAIN_COLS].copy()
    return _add_derived_columns_pbp(df)


def load_players_dataset() -> pd.DataFrame:
    return _load_players()[_PLAYERS_RETAIN_COLS]


def load_weekly_rosters_dataset(seasons: list[int]) -> pd.DataFrame:
    return _load_weekly_rosters(seasons=seasons)[_WEEKLY_ROSTERS_RETAIN_COLS]


def load_games_dataset() -> pd.DataFrame:
    return _load_games()[_GAMES_RETAIN_COL]


if __name__ == "__main__":
    _DEFAULT_SEASONS = list(range(1999, 2026))

    print(f"Fetching pbp for seasons {_DEFAULT_SEASONS[0]}-{_DEFAULT_SEASONS[-1]}...")
    load_pbp_dataset(seasons=_DEFAULT_SEASONS)

    print("Fetching participation...")
    load_participation_dataset(seasons=_DEFAULT_SEASONS)

    print("Fetching weekly rosters...")
    load_weekly_rosters_dataset(seasons=_DEFAULT_SEASONS)

    print("Fetching players...")
    load_players_dataset()

    print("Fetching games...")
    load_games_dataset()

    print("Data cached.")
