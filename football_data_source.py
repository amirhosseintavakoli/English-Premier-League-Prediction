"""Fetch EPL match data from football-data.co.uk.

fbref.com blocks automated requests behind a Cloudflare challenge that cannot be
bypassed with header spoofing, TLS impersonation, or real browser automation
(verified: plain requests, curl_cffi Chrome impersonation, and headless/headed
Playwright Chromium all receive the "Just a moment..." challenge page).

football-data.co.uk publishes free per-season CSV archives of EPL results with no
such protection, so this module fetches from there instead. The output columns
match what the fbref schedule table used to provide (Day, Date, Home, Away, Score,
Season, plus per-side match stats), so callers that used to consume the raw fbref
table can consume this instead.
"""
from io import StringIO

import numpy as np
import pandas as pd
import requests

# football-data.co.uk season codes, e.g. 2022-2023 -> '2223'
SEASON_CODES = {
    '2022-2023': '2223',
    '2023-2024': '2324',
    '2024-2025': '2425',
    '2025-2026': '2526',
    '2026-2027': '2627',
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# football-data.co.uk column -> output column (goals handled separately via Score)
STAT_COLUMN_MAP = {
    'HTR': 'HalfTimeResult',
    'HS': 'HomeShots',
    'AS': 'AwayShots',
    'HST': 'HomeShotsOnTarget',
    'AST': 'AwayShotsOnTarget',
    'HC': 'HomeCorners',
    'AC': 'AwayCorners',
    'HF': 'HomeFouls',
    'AF': 'AwayFouls',
    'HY': 'HomeYellowCards',
    'AY': 'AwayYellowCards',
    'HR': 'HomeRedCards',
    'AR': 'AwayRedCards',
}


def fetch_season(season, code, timeout=15):
    """Fetch and normalize a single season's CSV from football-data.co.uk."""
    url = BASE_URL.format(code=code)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    raw = pd.read_csv(StringIO(resp.text))
    raw = raw.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])

    date = pd.to_datetime(raw['Date'], dayfirst=True, errors='coerce')

    home_goals = raw.get('FTHG')
    away_goals = raw.get('FTAG')
    score = [
        f"{int(h)}–{int(a)}" if pd.notna(h) and pd.notna(a) else np.nan
        for h, a in zip(home_goals, away_goals)
    ]

    out = pd.DataFrame({
        'Day': date.dt.strftime('%a'),
        'Date': date.dt.strftime('%Y-%m-%d'),
        'Home': raw['HomeTeam'],
        'Away': raw['AwayTeam'],
        'Score': score,
        'Season': season,
    })

    for src_col, dst_col in STAT_COLUMN_MAP.items():
        if src_col in raw.columns:
            out[dst_col] = raw[src_col].values

    return out.dropna(subset=['Date'])


def fetch_match_data(season_code_map=None):
    """Fetch and combine EPL match data for the requested seasons.

    season_code_map: optional mapping season -> football-data.co.uk season code
    (e.g. {'2022-2023': '2223'}). Defaults to SEASON_CODES. A season whose CSV
    isn't available yet (e.g. a season that hasn't started) is skipped rather
    than raising.
    """
    seasons = season_code_map or SEASON_CODES
    dfs = []
    for season, code in seasons.items():
        try:
            print(f"Fetching match data for season {season} from football-data.co.uk...")
            dfs.append(fetch_season(season, code))
        except requests.exceptions.HTTPError as e:
            print(f"Skipping season {season}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
