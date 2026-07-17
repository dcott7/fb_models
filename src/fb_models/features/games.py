import pandas as pd


def build_games(games_df: pd.DataFrame) -> dict:
    df = games_df.copy()
    df["div_game"] = df["div_game"].astype(bool)

    games: dict = {}

    for _, game in df.iterrows():
        games[game["game_id"]] = {
            "season": game["season"],
            "week": game["week"],
            "game_type": game["game_type"],
            "away_team": game["away_team"],
            "home_team": game["home_team"],
            "away_coach": game["away_coach"],
            "home_coach": game["home_coach"],
            "context": {
                "location": game["location"],
                "div_game": game["div_game"],
                "away_rest": game["away_rest"],
                "home_rest": game["home_rest"],
                "roof": game["roof"],
                "surface": game["surface"],
                "temp": game["temp"],
                "wind": game["wind"],
            },
            "betting": {
                "spread_line": game["spread_line"],
                "total_line": game["total_line"],
                "away_moneyline": game["away_moneyline"],
                "home_moneyline": game["home_moneyline"],
                "away_spread_odds": game["away_spread_odds"],
                "home_spread_odds": game["home_spread_odds"],
                "under_odds": game["under_odds"],
                "over_odds": game["over_odds"],
            },
        }

    return games
