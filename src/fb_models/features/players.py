import pandas as pd


def build_players(players_df: pd.DataFrame, weekly_rosters_df: pd.DataFrame) -> dict:
    players: dict = {}

    for player_id, player_info in players_df.groupby("gsis_id"):
        players[player_id] = {
            "name": player_info["display_name"].iloc[0],
            "position_group": player_info["position_group"].iloc[0],
            "position": player_info["position"].iloc[0],
            "status": player_info["status"].iloc[0],
            "roster": {}
        }

    for player_id, roster_data in weekly_rosters_df.groupby("gsis_id"):
        roster = {
            season: {
                week: {
                    "team": week_data["team"].iloc[0],
                    "status": week_data["status"].iloc[0]
                }
                for week, week_data in season_data.groupby("week")
            }
            for season, season_data in roster_data.groupby("season")
        }

        if player_id in players:
            players[player_id]["roster"] = roster
        else:
            print(f"Warning: Adding player ({player_id}) roster data who did not exist in players_df")
            # optional: handle players that exist only in weekly_rosters_df
            players[player_id] = {
                "roster": roster
            }

    return players