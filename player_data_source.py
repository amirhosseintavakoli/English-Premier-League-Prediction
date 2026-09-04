"""Fetch EPL player stats from understat.com.

fbref.com's Big-5-leagues player stats page is blocked the same way its match
schedule pages are (see football_data_source.py): plain requests get a 403
Cloudflare "Just a moment..." challenge page instead of the stats table.

understat.com exposes an internal AJAX endpoint (main/getPlayersStats) used by
its own league pages to fill in the players table client-side. It has no bot
protection and returns per-player season totals (goals, assists, xG, xA,
minutes, cards) as JSON, so this module fetches player data from there
instead. understat only covers the EPL (and a few other top leagues), not
fbref's Big-5, but that's all this app uses.
"""
import pandas as pd
import requests

# understat identifies a season by its start year, e.g. the 2023-2024 season is '2023'
SEASON_YEARS = {
    '2022-2023': '2022',
    '2023-2024': '2023',
    '2024-2025': '2024',
    '2025-2026': '2025',
    '2026-2027': '2026',
}

ENDPOINT = "https://understat.com/main/getPlayersStats"

# understat position tokens -> single primary position, mirroring fbref's Pos column
POSITION_MAP = {'GK': 'GK', 'D': 'DF', 'M': 'MF', 'F': 'FW'}

# understat spells some clubs differently than football-data.co.uk (used for match
# data, see football_data_source.py); normalize so Squad joins Team on a merge.
SQUAD_NAME_MAP = {
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Newcastle United': 'Newcastle',
    'Nottingham Forest': "Nott'm Forest",
    'Wolverhampton Wanderers': 'Wolves',
}

NUMERIC_COLUMNS = [
    'games', 'time', 'goals', 'xG', 'assists', 'xA', 'shots', 'key_passes',
    'yellow_cards', 'red_cards', 'npg', 'npxG', 'xGChain', 'xGBuildup',
]


def _primary_position(position):
    """understat's position field is a space-separated list of role tokens,
    ordered by prominence (e.g. 'D F M S'); pick the first recognized one."""
    for token in position.split():
        if token in POSITION_MAP:
            return POSITION_MAP[token]
    return None


def fetch_season(season, year, timeout=15):
    """Fetch and normalize a single season's player stats from understat."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/league/EPL/{year}",
    }
    resp = requests.post(ENDPOINT, headers=headers, data={"league": "EPL", "season": year}, timeout=timeout)
    resp.raise_for_status()
    players = resp.json().get('players', [])
    if not players:
        return pd.DataFrame()

    df = pd.DataFrame(players)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Season'] = season
    df['Player'] = df['player_name']
    # a player transferred mid-season keeps a comma-joined "TeamA,TeamB" squad
    df['Squad'] = df['team_title'].apply(
        lambda s: ','.join(SQUAD_NAME_MAP.get(t, t) for t in s.split(','))
    )
    df['Pos'] = df['position'].map(_primary_position)

    # G+A-PK per 90: non-penalty goals (npg) + assists, over 90-minute units played
    minutes_per90 = df['time'] / 90
    df['Per90_G+A-PK'] = (df['npg'] + df['assists']) / minutes_per90.replace(0, pd.NA)

    df = df.rename(columns={
        'games': 'Games', 'time': 'Minutes', 'goals': 'Goals', 'assists': 'Assists',
        'npg': 'NonPenaltyGoals', 'yellow_cards': 'YellowCards', 'red_cards': 'RedCards',
        'key_passes': 'KeyPasses', 'shots': 'Shots',
    })

    keep = ['Player', 'Squad', 'Pos', 'Season', 'Games', 'Minutes', 'Goals', 'Assists',
            'xG', 'xA', 'NonPenaltyGoals', 'Per90_G+A-PK', 'Shots', 'KeyPasses',
            'YellowCards', 'RedCards']
    return df[keep]


def fetch_player_data(season_year_map=None, timeout=15):
    """Fetch and combine EPL player stats for multiple seasons.

    season_year_map: Optional mapping season->understat season-start-year
    (e.g. {'2022-2023': '2022'}). If not provided, SEASON_YEARS is used.
    A season with no data yet available on understat (e.g. one that hasn't
    started) is skipped rather than raising.
    """
    if season_year_map is None:
        season_year_map = SEASON_YEARS

    dfs = []
    for season, year in season_year_map.items():
        try:
            print(f"Fetching player data for season {season}...")
            tmp = fetch_season(season, year, timeout=timeout)
        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e)
            raise
        if tmp.empty:
            print(f"No player data available yet for season {season}.")
            continue
        dfs.append(tmp)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
