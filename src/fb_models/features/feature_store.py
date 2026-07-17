import pandas as pd

from .players import build_players
from .participation import build_participation
from .usage import build_usage
from .situations import build_situation
from .teams import build_teams
from .coaches import build_coaches
from .games import build_games


class FeatureStore:

    def __init__(
        self,
        plays_df: pd.DataFrame,
        participation_df: pd.DataFrame,
        players_df: pd.DataFrame,
        weekly_rosters_df: pd.DataFrame,
        games_df: pd.DataFrame
    ):
        self.plays_df = plays_df
        self.participation_df = participation_df
        self.players_df = players_df
        self.weekly_rosters_df = weekly_rosters_df
        self.games_df = games_df

        self.players: dict = {}
        self.participation: dict = {}
        self.usage: dict = {}

        self.situations: dict = {}
        self.teams: dict = {}
        self.coaches: dict = {}
        self.games: dict = {}

    def build(self) -> None:
        self.players = build_players(self.players_df, self.weekly_rosters_df)
        self.participation = build_participation(self.participation_df)
        self.usage = build_usage(self.plays_df, self.participation_df)

        self.situations = build_situation(self.plays_df)
        # self.teams = build_teams()
        self.coaches = build_coaches(self.plays_df, self.participation_df)
        self.games = build_games(self.games_df)