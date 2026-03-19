from pathlib import Path
import requests


DATA_DIR = Path(__file__).parent.parent / "data"
BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"


def pull_game(season: int):
    url = f"{BASE_URL}/pbp/play_by_play_{season}.parquet"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(DATA_DIR / f"pbp_{season}.parquet", "wb") as f:
                f.write(response.content)
            print(f"Successfully downloaded play-by-play data for season {season}.")
        else:
            print(
                f"Failed to download play-by-play data for season {season}. Status code: {response.status_code}"
            )
    except requests.RequestException as e:
        print(f"Error downloading play-by-play data for season {season}: {e}")
        print(
            f"Failed to download play-by-play data for season {season}. Status code: {response.status_code}"
        )


def main() -> None:
    seasons = range(1999, 2026)  # Adjust the range as needed
    for season in seasons:
        pull_game(season)


if __name__ == "__main__":
    main()
