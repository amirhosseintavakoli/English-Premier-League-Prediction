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

# combined-leagues feed of the next round(s) of not-yet-played fixtures (results-only
# archives above never include these, since a match isn't in them until it's played)
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURES_DIV = "E0"  # football-data.co.uk's code for the EPL

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


def _season_for_date(date, season_codes):
    """Map a fixture date to one of season_codes' season labels using its EPL
    season-start-year convention (the season starting in year Y runs July Y
    through June Y+1)."""
    if pd.isna(date):
        return None
    start_year = date.year if date.month >= 7 else date.year - 1
    return f"{start_year}-{start_year + 1}" if f"{start_year}-{start_year + 1}" in season_codes else None


def fetch_fixtures(season_codes, timeout=15):
    """Fetch not-yet-played EPL fixtures from football-data.co.uk's combined
    fixtures feed (the only place it publishes fixtures ahead of kickoff; the
    per-season archives in fetch_season only ever contain played matches).

    Only covers the next round or two of matches across all leagues, so this
    is necessarily a short lookahead, not the full remaining season schedule.
    Rows are tagged with a season label from season_codes based on their date;
    a fixture whose date doesn't map to a season in season_codes is dropped.
    """
    resp = requests.get(FIXTURES_URL, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = 'utf-8-sig'

    raw = pd.read_csv(StringIO(resp.text))
    raw = raw[raw['Div'] == FIXTURES_DIV].dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
    if raw.empty:
        return pd.DataFrame()

    date = pd.to_datetime(raw['Date'], dayfirst=True, errors='coerce')
    season = date.map(lambda d: _season_for_date(d, season_codes))

    out = pd.DataFrame({
        'Day': date.dt.strftime('%a'),
        'Date': date.dt.strftime('%Y-%m-%d'),
        'Home': raw['HomeTeam'].values,
        'Away': raw['AwayTeam'].values,
        'Score': np.nan,
        'Season': season.values,
    })
    return out.dropna(subset=['Date', 'Season'])


def fetch_match_data(season_code_map=None, include_fixtures=True):
    """Fetch and combine EPL match data for the requested seasons.

    season_code_map: optional mapping season -> football-data.co.uk season code
    (e.g. {'2022-2023': '2223'}). Defaults to SEASON_CODES. A season whose CSV
    isn't available yet (e.g. a season that hasn't started) is skipped rather
    than raising.

    include_fixtures: also append upcoming, not-yet-played fixtures (see
    fetch_fixtures) so callers can generate predictions for them.
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

    if include_fixtures:
        try:
            print("Fetching upcoming fixtures from football-data.co.uk...")
            fixtures = fetch_fixtures(seasons)
            if not fixtures.empty:
                played = set()
                for d in dfs:
                    played.update(zip(d['Season'], d['Date'], d['Home'], d['Away']))
                fixtures = fixtures[~fixtures.apply(
                    lambda r: (r['Season'], r['Date'], r['Home'], r['Away']) in played, axis=1)]
                if not fixtures.empty:
                    dfs.append(fixtures)
        except requests.exceptions.HTTPError as e:
            print(f"Skipping fixtures: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
