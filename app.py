import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance

# Function to load Premier League Match data from multiple seasons
@st.cache_data
def load_match_data(season_url_map=None):
    """
    check if the data is already loaded on Github. If yes, 
    use the data.    

    If no,
    Load and combine match schedules from fbref for multiple seasons.

    season_url_map: Optional mapping season->url. If not provided, a default
    mapping for a handful of recent EPL seasons is used.
    """

    # check if the data is loaded
    try:
        df = pd.read_csv("match_data.csv")
        print("Loaded the cleaned match data.")
        return df
    except Exception as e:
        print("Loading the raw data ...")

    # standard headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko)"
                    "Chrome/115.0"
                    "Safari/537.36",
    
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }

    # sensible defaults to keep existing behaviour
    print("Loading match data...")
    if season_url_map is None:
        season_url_map = {
            '2022-2023': 'https://fbref.com/en/comps/9/2022-2023/schedule/2022-2023-Premier-League-Scores-and-Fixtures',
            '2023-2024': 'https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures',
            '2024-2025': 'https://fbref.com/en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures',
            '2025-2026': 'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures',
        }

    dfs = []
    for season, url in season_url_map.items():
        try:
            print(f"Fetching match data for season {season}...")
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            tmp = pd.read_html(StringIO(resp.text))[0]
            tmp['Season'] = season
            tmp = tmp.dropna(subset=['Date'])
            dfs.append(tmp)
        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e, resp.status_code)
            print("Response headers:", resp.headers)
            raise
        except Exception as e:
            print("Other error:", e)
            raise
    if not dfs:
        print("No match data loaded.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    # drop columns only if they exist (avoid KeyError)
    drop_cols = [c for c in ['Venue', 'Match Report', 'Notes'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    print(f"Loaded match data with {len(df)} rows from {len(dfs)} seasons.")

    ## split the Date into a year, month and date
    df['year']  = [int(d.split("-")[0]) for d in df.Date]
    df['month'] = [int(d.split("-")[1]) for d in df.Date]
    df['day']   = [int(d.split("-")[2]) for d in df.Date]
    print("Extracted year, month, day from Date.")

    df['FullTimeHomeGoals'] = [np.nan if pd.isna(x) else int(x.split("–")[0].strip()) for x in df['Score']]
    df['FullTimeAwayGoals'] = [np.nan if pd.isna(x) else int(x.split("–")[1].strip()) for x in df['Score']]
    print("Extracted FullTimeHomeGoals and FullTimeAwayGoals from Score.")

    ## create a stable unique ID for every team (covers Home and Away)
    teams = sorted(
        set(df['Home'].dropna().unique()).union(set(df['Away'].dropna().unique()))
    )

    team_to_id = {name: i for i, name in enumerate(teams, start=1)}
    print(f"Assigned unique IDs to {len(teams)} teams.")

    # add ID columns to the match dataframe
    df['HomeTeamID'] = df['Home'].map(team_to_id)
    df['AwayTeamID'] = df['Away'].map(team_to_id)

    # ensure a stable match id
    df = df.reset_index(drop=True)
    df['MatchID'] = df.index + 1
    print("Assigned stable MatchID to each match.")

    df.to_csv("match_data.csv", index = False, encoding="utf-8")
    print("Match Data Exported.")

    return df

# Function to load Big 5 European Leagues Player data from multiple seasons
@st.cache_data
def load_player_data(season_url_map=None):
    """Load and combine player stats pages from fbref for multiple seasons.

    season_url_map: Optional mapping season->url. If not provided a default mapping
    for several seasons is used.
    """

    # load if the player data already exists
    try:
        df = pd.read_csv("player_data.csv", header=[0,1])
        print("Loaded the player data.")
        return df
    except Exception as e:
        print("Loading the raw player data ...")


    # standard headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }
    
    print("Loading player data...")
    if season_url_map is None:
        season_url_map = {
            '2022-2023': 'https://fbref.com/en/comps/Big5/2021-2022/stats/players/2021-2022-Big-5-European-Leagues-Stats',
            '2023-2024': 'https://fbref.com/en/comps/Big5/2022-2023/stats/players/2022-2023-Big-5-European-Leagues-Stats',
            '2024-2025': 'https://fbref.com/en/comps/Big5/2023-2024/stats/players/2023-2024-Big-5-European-Leagues-Stats',
            '2025-2026': 'https://fbref.com/en/comps/Big5/2024-2025/stats/players/2023-2024-Big-5-European-Leagues-Stats',
        }

    dfs = []
    for season, url in season_url_map.items():
        try:
            print(f"Fetching player data for season {season}...")
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()  # raises HTTPError for 4xx/5xx
            tmp = pd.read_html(StringIO(resp.text))[0]
            tmp['Season'] = season
            dfs.append(tmp)
        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e, resp.status_code)
            print("Response headers:", resp.headers)
            raise
        except Exception as e:
            print("Other error:", e)
            raise
    if not dfs:
        print("No player data loaded.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded player data with {len(df)} rows from {len(dfs)} seasons.")

    df.to_csv("player_data.csv", index = False, encoding="utf-8")
    print("Player Data Exported")

    return df

# Function to clean and preprocess the player data
def clean_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean player-level fbref data and keep only EPL players.

    This function handles tables where columns are read as MultiIndex from
    pandas.read_html; it simplifies the column names and keeps only the
    relevant EPL players while converting useful fields to numeric types.
    """
    if df.empty:
        return df

    def _clean_col(col):
        # support both simple string columns and multi-index (tuples)
        if isinstance(col, tuple):
            top = str(col[0]).lower() if col[0] is not None else ''
            bot = col[1]
            if 'unnamed' in top:
                return bot
            if 'per 90' in top:
                return f'Per90_{bot}'
            if 'playing time' in top:
                return f'PT_{bot}'
            if any(k in top for k in ('progression', 'performance', 'expected')):
                return bot
            if 'season' in top:
                return col[0]
            return bot
        # if it's a plain column name
        return col

    # Build new columns list using helper
    df = df.copy()
    df.columns = [_clean_col(c) for c in df.columns]


    # Normalise competition names
    if 'Comp' in df.columns:
        df['Comp'] = df['Comp'].replace({
            'eng Premier League': 'EPL',
            'fr Ligue 1': 'Ligue 1',
            'it Serie A': 'Serie A',
            'de Bundesliga': 'Bundesliga',
            'es La Liga': 'La Liga'
        })
        df = df[df['Comp'] == 'EPL']

    # tidy up some columns if present
    drop_columns = [c for c in ['Rk', 'Born', 'Comp', 'Matches'] if c in df.columns]
    if drop_columns:
        df = df.drop(columns=drop_columns)

    if 'Nation' in df.columns:
        df['Nation'] = df['Nation'].str.extract(r'([A-Z]{3})', expand=False)

    # simplify position column
    if 'Pos' in df.columns:
        df['Pos'] = df['Pos'].str.split(',').str[0]

    # ensure Per90_G+A-PK is numeric and use it for ranking if present
    if 'Per90_G+A-PK' in df.columns:
        df['Per90_G+A-PK'] = pd.to_numeric(df['Per90_G+A-PK'], errors='coerce')
        df['Rank'] = df.groupby(['Season', 'Pos'])['Per90_G+A-PK'].rank(ascending=False, method='min').astype('Int64')
        df = df.sort_values(['Season', 'Pos', 'Rank'])

    print(f"Cleaned player data now has {len(df)} rows.")
    return df

# Function to rank players by G+A-PK and return quartile counts baed on analysis_match_player_level.ipynb
def rank_players_by_ga(df: pd.DataFrame, position='FW', metric='Per90_G+A-PK'):
    """Rank players by metric and position and return quartile-based counts.   
    Parameters
    - df: DataFrame with cleaned player data (expects columns: Season, Squad)
    - position: position to filter by ('FW', 'MF', 'DF')
    - metric: metric to rank players by (default: Per90_G+A-PK)
    Returns a dictionary of DataFrames keyed by position ('FW', 'MF', 'DF') with pivoted
    quartile counts (columns 0-3).
    """
    tmp = df[df['Pos'] == position].copy()
    tmp['Quartile'] = tmp.groupby(['Season'])[metric].transform(
        lambda x: pd.qcut(x, 4, labels=False, duplicates='drop'))

    grouped = tmp.groupby(['Season', 'Pos', 'Squad', 'Quartile'])['Player'].nunique().reset_index()
    grouped = grouped.rename(columns={'Player': 'num_players'})

    # Create pivoted tables for positions of interest
    pivot = grouped.pivot_table(index=['Season', 'Squad'], columns='Quartile', values='num_players', fill_value=0)
    for q in range(4):
        if q not in pivot.columns:
            pivot[q] = 0
    pivot = pivot.sort_index(axis=1).reset_index()

    # rename quartile columns to indicate position
    pivot = pivot.rename(columns={q: f'{metric}_{position}_Q{q}' for q in range(4)})

    print("Ranked players by G+A-PK quartiles.")
    # print(pivot.columns)
    return pivot

# Transform match data to team-centric view
def team_view(df):

    """Transform match-level data to team-level perspective.
        Add outcomes and points from the full-time scores.

    Example:
    - df = team_view(match_data)
    """

    # add outcome from full-time scores (W/L/D from each team's perspective)
    def wl_from_scores(row):
        gf = row['GoalsFor']
        ga = row['GoalsAgainst']
        if pd.isna(gf) or pd.isna(ga):
            return None
        if gf > ga:
            return 'W'
        if gf < ga:
            return 'L'
        return 'D'

    # add points from full-time scores (3/0/1 from each team's perspective)
    def points_from_scores(row):
        gf = row['GoalsFor']
        ga = row['GoalsAgainst']
        if pd.isna(gf) or pd.isna(ga):
            return None
        if gf > ga:
            return 3
        if gf < ga:
            return 0
        return 1

    # home-team perspective
    home = pd.DataFrame({
        'Season': df['Season'],
        'MatchID': df['MatchID'],
        'Date': df['Date'],
        'year': df['year'],
        'month': df['month'],
        'day': df['day'],
        'DayofWeek': df['Day'],
        'Team': df['Home'],
        'TeamID': df.get('HomeTeamID'),
        'OpponentID': df.get('AwayTeamID'),
        'IsHome': True,
        'GoalsFor': df.get('FullTimeHomeGoals'),
        'GoalsAgainst': df.get('FullTimeAwayGoals'),
        'Shots': df.get('HomeShots'),
        'ShotsOnTarget': df.get('HomeShotsOnTarget'),
        'Corners': df.get('HomeCorners'),
        'Fouls': df.get('HomeFouls'),
        'YellowCards': df.get('HomeYellowCards'),
        'RedCards': df.get('HomeRedCards'),
        'ShotsAgainst': df.get('AwayShots'),
        'ShotsOnTargetAgainst': df.get('AwayShotsOnTarget'),
        'CornersAgainst': df.get('AwayCorners'),
        'FoulsAgainst': df.get('AwayFouls'),
        'YellowCardsAgainst': df.get('AwayYellowCards'),
        'RedCardsAgainst': df.get('AwayRedCards'),
        'HalfTimeResult': df.get('HalfTimeResult')
    })
    # away perspective
    away = pd.DataFrame({
        'Season': df['Season'],
        'MatchID': df['MatchID'],
        'Date': df['Date'],
        'year': df['year'],
        'month': df['month'],
        'day': df['day'],
        'DayofWeek': df['Day'],
        'Team': df.get('Away'),
        'TeamID': df.get('AwayTeamID'),
        'OpponentID': df.get('HomeTeamID'),
        'IsHome': False,
        'GoalsFor': df.get('FullTimeAwayGoals'),
        'GoalsAgainst': df.get('FullTimeHomeGoals'),
        'Shots': df.get('AwayShots'),
        'ShotsOnTarget': df.get('AwayShotsOnTarget'),
        'Corners': df.get('AwayCorners'),
        'Fouls': df.get('AwayFouls'),
        'YellowCards': df.get('AwayYellowCards'),
        'RedCards': df.get('AwayRedCards'),
        'ShotsAgainst': df.get('HomeShots'),
        'ShotsOnTargetAgainst': df.get('HomeShotsOnTarget'),
        'CornersAgainst': df.get('HomeCorners'),
        'FoulsAgainst': df.get('HomeFouls'),
        'YellowCardsAgainst': df.get('HomeYellowCards'),
        'RedCardsAgainst': df.get('HomeRedCards'),
        'HalfTimeResult': df.get('HalfTimeResult')
    })

    df = pd.concat([home, away], ignore_index=True).sort_values(['Season', 'TeamID', 'MatchID']).reset_index(drop=True)

    # sort the data by team ID and match ID and create a new variable indicating the teams' game number since beginning of each season
    df['Week'] = df.groupby(['Season', 'TeamID']).cumcount() + 1
    df['Week'] = df['Week'].astype(int)

    df['Outcome']  = df.apply(wl_from_scores, axis=1)
    df['Points']   = df.apply(points_from_scores, axis=1)

    # reorder columns
    cols = ['Season', 'MatchID','Date', 'Week', 'DayofWeek','Team','TeamID','IsHome',
            'GoalsFor','GoalsAgainst','Outcome', 'Points']
    df = df[[c for c in cols if c in df.columns]]

    return df

def rolling_average_stats(df, stat_cols, window=5):
    """Compute rolling average statistics for specified columns over a given window size.

    Parameters:
    - df: DataFrame with team-level match data (expects 'Season', 'TeamID', and stat_cols)
    - stat_cols: list of column names to compute rolling averages for
    - window: integer size of the rolling window

    Returns:
    - DataFrame with additional columns for rolling averages (named as 'RollingAvg_<stat_col>')

    Example:
    - df = rolling_average_stats(df, ['GoalsFor', 'Shots'], window=3)
    """
    df = df.copy()
    for col in stat_cols:
        rolling_col_name = f'RollingAvg{window}_{col}'
        df[rolling_col_name] = df.groupby(['Season', 'TeamID'])[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    return df

def lagged_stats(df, stat_cols, lag=1):
    """Compute lagged statistics for specified columns.

    Parameters:
    - df: DataFrame with team-level match data (expects 'Season', 'TeamID', and stat_cols)
    - stat_cols: list of column names to compute lagged values for
    - lag: integer number of matches to lag by

    Returns:
    - DataFrame with additional columns for lagged stats (named as 'Lagged_<stat_col>')

    Example:
    - df = lagged_stats(df, ['GoalsFor', 'Shots'], lag=1)
    """
    df = df.copy()
    for col in stat_cols:
        lagged_col_name = f'Lagged{lag}_{col}'
        df[lagged_col_name] = df.groupby(['Season', 'TeamID'])[col].shift(lag)
    return df

def max_past_stats(df, stat_cols, window=1):
    """Compute best past statistics for specified columns over a given window size.

    Parameters:
    - df: DataFrame with team-level match data (expects 'Season', 'TeamID', and stat_cols)
    - stat_cols: list of column names to compute best past values for
    - window: integer size of the look-back window

    Returns:
    - DataFrame with additional columns for best past stats (named as 'MaxPast_<stat_col>')

    Example:
    - df = best_past_stats(df, ['GoalsFor', 'Shots'], window=3)
    """
    df = df.copy()
    for col in stat_cols:
        best_past_col_name = f'MaxPast{window}_{col}'
        df[best_past_col_name] = df.groupby(['Season', 'TeamID'])[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )
    return df

def min_past_stats(df, stat_cols, window=1):
    """Compute worst past statistics for specified columns over a given window size.

    Parameters:
    - df: DataFrame with team-level match data (expects 'Season', 'TeamID', and stat_cols)
    - stat_cols: list of column names to compute worst past values for
    - window: integer size of the look-back window

    Returns:
    - DataFrame with additional columns for worst past stats (named as 'MinPast_<stat_col>')

    Example:
    - df = worst_past_stats(df, ['GoalsFor', 'Shots'], window=3)
    """
    df = df.copy()
    for col in stat_cols:
        worst_past_col_name = f'MinPast{window}_{col}'
        df[worst_past_col_name] = df.groupby(['Season', 'TeamID'])[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).min()
        )
    return df

def merge_team_player(df, player_data, stat_cols, rolling_window=5, lag=1, past_window=3):
    """Build estimation dataset with rolling averages, lagged stats, and best/worst past stats.

    Parameters:
    - df: DataFrame with team-level match data (expects 'Season', 'TeamID', and stat_cols)
    - stat_cols: list of column names to compute features for
    - rolling_window: integer size of the rolling average window
    - lag: integer number of matches to lag by
    - past_window: integer size of the look-back window for best/worst stats

    Returns:
    - DataFrame with additional feature columns.
    """
    df = rolling_average_stats(df, stat_cols, window=rolling_window)
    df = lagged_stats(df, stat_cols, lag=lag)
    df = max_past_stats(df, stat_cols, window=past_window)
    df = min_past_stats(df, stat_cols, window=past_window)

    for pos in ['FW', 'MF', 'DF']:
        df_player_ranking = rank_players_by_ga(player_data, position=pos, metric='Per90_G+A-PK')
 
        df = df.merge(df_player_ranking, how='left',
                        left_on=['Season', 'Team'],
                        right_on=['Season', 'Squad'])
    
    return df

def build_dataset(df, features=None, categorical=['Week', 'IsHome', 'TeamID', 'DayofWeek']):
    """Return X (with dummies applied), y, label encoder, label mapping, used raw features, and model columns.

    - 'features' is a list of raw dataframe columns you want to use (e.g. ['Week','IsHome','GoalsFor_roll3']).
    - categorical: list of columns that should be one-hot encoded.
    The function drops rows with NA in Outcome or any chosen raw feature before encoding.
    """
    # Default feature set (you can edit the runner cell to change which sets to try)
    DEFAULT_FEATURES = ['Week', 'IsHome', 'TeamID', 'DayofWeek']

    if features is None:
        features = DEFAULT_FEATURES
    
    #ensure features exist in df
    present = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Warning: some requested features are missing from the dataframe: {missing}")
    
    cols = ['Outcome'] + present
    data = df[cols].dropna().copy()

    # Separate X_raw and Y
    X_raw = data[present].copy()
    Y = data['Outcome']

    # One-hot encode categorical variables (only those present)
    cat_present = [c for c in categorical if c in X_raw.columns]
    if cat_present:
        X = pd.get_dummies(X_raw, columns=cat_present, dummy_na=False, drop_first=False)
    else:
        X = X_raw.astype(float)

    # Keep track of final model columns to ensure consistent prediction later
    model_columns = X.columns.tolist()
    print(f"Model will use {len(model_columns)} features after encoding.")

    # Ensure numeric dtype for downstream scalers/models
    X = X.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(Y)
    label_mapping = {cls: int(i) for i, cls in enumerate(le.classes_)}
    return X, y, le, label_mapping, model_columns

def safe_train_test_split(X, y, test_size=0.2, random_state=173):
    """Attempt stratified train-test split; fall back to regular split if stratification fails."""
    try:
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    except Exception as e:
        print("Stratified split failed (reason):", e)
        print("Falling back to a regular random split without stratification.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_models(X_train, y_train, use_xgb=True, use_rf=True):
    """Train KNN (scaled), Logistic (scaled), RandomForest (unscaled), and XGBoost (unscaled).
    Returns a dict with trained models and the scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # Logistic regression (try to use penalty='none' when available)
    log_reg = LogisticRegression(max_iter=2000)
    log_reg.fit(X_train_scaled, y_train)

    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=173)
    rf_clf.fit(X_train, y_train)  # RF can handle unscaled inputs

    # XGBoost
    xgb_clf = XGBClassifier(use_label_encoder=True, eval_metric='mlogloss')
    xgb_clf.fit(X_train, y_train)  # XGBoost can handle unscaled inputs

    nnet_clf = MLPClassifier(hidden_layer_sizes=(200,20,), 
                             early_stopping=True,
                             random_state=173)
    nnet_clf.fit(X_train_scaled, y_train)
    
    return {'scaler': scaler, 'knn': knn, 'log_reg': log_reg, 'rf': rf_clf, 'xgb': xgb_clf, 'nnet': nnet_clf}

def evaluate(models, X_test, y_test):
    """Evaluate trained models on test set and return accuracy scores."""
    scaler = models['scaler']
    X_test_scaled = scaler.transform(X_test)
    results = {}
    results['knn_acc'] = accuracy_score(y_test, models['knn'].predict(X_test_scaled))
    results['logreg_acc'] = accuracy_score(y_test, models['log_reg'].predict(X_test_scaled))
    results['rf_acc'] = accuracy_score(y_test, models['rf'].predict(X_test))
    results['xgb_acc'] = accuracy_score(y_test, models['xgb'].predict(X_test))
    results['nnet_acc'] = accuracy_score(y_test, models['nnet'].predict(X_test_scaled))
    return results

def predict(df, features=None, model_columns=None, models=None, le=None, categorical=['Week', 'IsHome', 'TeamID', 'DayofWeek']):
    """ Create dummies for categorical vars on the full df using the raw_features list,
    then reindex to model_columns (adding missing columns with zeros) before scaling and predicting.
    """

    DEFAULT_FEATURES = ['Week', 'IsHome', 'TeamID', 'DayofWeek']
    if features is None:
        features = DEFAULT_FEATURES
    
    #ensure features exist in df
    present = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Warning: some requested features are missing from the dataframe: {missing}")
    
    X_raw = df[present].copy()

    # One-hot encode categorical variables present in the raw features
    cat_present = [c for c in categorical if c in X_raw.columns]
    if cat_present:
        X_all = pd.get_dummies(X_raw, columns=cat_present, dummy_na=False, drop_first=False)
    else:
        X_all = X_raw.copy()

    # Reindex to match the model training columns, fill missing with 0
    # X_all_wo_na = X_all.reindex(columns=model_columns).astype(float).dropna()
    X_all = X_all.reindex(columns=model_columns).astype(float)
    
    # Keep only rows without missing values
    valid_rows = X_all.dropna().index
    X_all = X_all.loc[valid_rows]
    df_subset = df.loc[valid_rows].copy()

    scaler = models['scaler']
    X_scaled = scaler.transform(X_all)

    df_subset['Predicted_knn']     = le.inverse_transform(models['knn'].predict(X_scaled))
    df_subset['Predicted_logreg']  = le.inverse_transform(models['log_reg'].predict(X_scaled))
    df_subset['Predicted_rf']      = le.inverse_transform(models['rf'].predict(X_all))
    df_subset['Predicted_xgb']     = le.inverse_transform(models['xgb'].predict(X_all))
    df_subset['Predicted_nnet']    = le.inverse_transform(models['nnet'].predict(X_scaled))
    
    return df_subset
    # return df

# Set page config
st.set_page_config(page_title="Premier League Prediction", layout="wide")
st.title("Premier League Match Prediction")
if __name__ == "__main__":
    match_data = load_match_data()
    player_data = clean_player_data(load_player_data())
    df = team_view(match_data)
    df = merge_team_player(df, player_data, stat_cols=['GoalsFor', 'GoalsAgainst'], rolling_window=5, lag=1, past_window=3)

    # intro to the website
    st.write("This application allows you to train and evaluate various machine learning models to predict Premier League match outcomes based on historical match and player data. You can select different features to include in the model, train the models, and view their performance metrics.")
    st.write("Here's the GitHub repo: https://github.com/amirhosseintavakoli/English-Premier-League-Prediction")
    # create a feature list to select from
    st.header("Feature Selection")
    st.write("Feature Dictionary:" \
    "\n- Week: Match week number for the team in the season (categorical)" \
    "\n- IsHome: Whether the team is playing at home (True/False)" \
    "\n- TeamID: Unique identifier for the team (categorical)" \
    "\n- DayofWeek: Day of the week the match is played (1=Monday,...7=Sunday)" \
    "\n- RollingAvgX_Stat: Rolling average of 'Stat' over the past X matches" \
    "\n- LaggedX_Stat: Value of 'Stat' from X matches ago" \
    "\n- MaxPastX_Stat: Maximum value of 'Stat' over the past X matches" \
    "\n- MinPastX_Stat: Minimum value of 'Stat' over the past X matches" \
    "\n- Per90_G+A-PK_POS_QX: Number of players in position POS (FW/MF/DF) in quartile X (0=best,3=worst) based on (Goals+Assits-PenatlyKicks) per 90 mins in the past season" 
    )


    # default and options for feature selection
    # in future versions, this could be dynamic based on user choice
    # 1. choice of variable, choice of windown, choice of transformations
    # e.g. rolling avg 3/5/7, lagged 1/2/3, past best/worst 3/5
    # 2. selection of categorical vars to one-hot encode
    default = ['Week', 'IsHome', 'TeamID', 'DayofWeek']
    options = ['Week', 'IsHome', 'TeamID', 'DayofWeek', 
               'RollingAvg5_GoalsFor', 'RollingAvg5_GoalsAgainst',
               'RollingAvg3_GoalsFor', 'RollingAvg3_GoalsAgainst',
                'Lagged1_GoalsFor', 'Lagged1_GoalsAgainst',
                'Lagged2_GoalsFor', 'Lagged2_GoalsAgainst',
                'Lagged3_GoalsFor', 'Lagged3_GoalsAgainst',
                'MaxPast3_GoalsFor', 'MaxPast3_GoalsAgainst',
                'MinPast3_GoalsFor', 'MinPast3_GoalsAgainst',
                'Per90_G+A-PK_FW_Q0', 'Per90_G+A-PK_FW_Q1',
                'Per90_G+A-PK_FW_Q2', 'Per90_G+A-PK_FW_Q3',
                'Per90_G+A-PK_MF_Q0', 'Per90_G+A-PK_MF_Q1',
                'Per90_G+A-PK_MF_Q2', 'Per90_G+A-PK_MF_Q3',
                'Per90_G+A-PK_DF_Q0', 'Per90_G+A-PK_DF_Q1',
                'Per90_G+A-PK_DF_Q2', 'Per90_G+A-PK_DF_Q3',
               ]
    features = st.multiselect("Select Features", default=default, options=options, key="selected_features")

    # Train Models button — when clicked we train and store models/results in session_state
    if st.button("Train Models"):
        X, y, le, label_mapping, model_columns = build_dataset(df, features=features,
                                                            categorical=['Week', 'IsHome', 'TeamID', 'DayofWeek'])
        print("Feature matrix X shape:", X.shape)

        X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=0.2, random_state=173)
        models = train_models(X_train, y_train)
        results = evaluate(models, X_test, y_test)

        # Example prediction on the full dataset
        df_predictions = predict(df, features=features, model_columns=model_columns, models=models, le=le)

        # Persist trained artifacts so they survive Streamlit reruns caused by other widgets
        st.session_state['models'] = models
        st.session_state['model_columns'] = model_columns
        st.session_state['le'] = le
        st.session_state['results'] = results
        st.session_state['df_predictions'] = df_predictions
        print("Trained models and predictions stored in session_state.")

    # If we've trained models (or they exist in session_state), show evaluation and optional displays
    if 'models' in st.session_state:
        st.header("Model Evaluation Results")
        st.write("K-Neighbors Accuracy: ", st.session_state['results'].get('knn_acc'))
        st.write("Logistic Regression Accuracy: ", st.session_state['results'].get('logreg_acc'))
        st.write("Random Forest Accuracy: ", st.session_state['results'].get('rf_acc'))
        st.write("XGBoost Accuracy: ", st.session_state['results'].get('xgb_acc'))
        st.write("Neural Network Accuracy: ", st.session_state['results'].get('nnet_acc'))


        # st.write(st.session_state.get('results', {}))

        # Prediction table (only when user requests it)
        # choose the season and week to display
        selected_season = st.select_slider("Select Season", options=['2022-2023', '2023-2024', '2024-2025', '2025-2026'], key="selected_season", value='2025-2026')
        selected_week = st.select_slider("Select Week", options=list(range(1, 39)), key="selected_week", value=13)


        if st.button("Show Prediction data"):
            preds = st.session_state.get('df_predictions')
            if preds is not None and 'Season' in preds.columns and 'Week' in preds.columns:
                st.dataframe(preds[(preds['Season'] == selected_season) & (preds['Week'] == selected_week)][['Team', 'Date', 'Week', 'Outcome', 'Predicted_knn', 'Predicted_logreg', 'Predicted_rf', 'Predicted_xgb', 'Predicted_nnet']].reset_index(drop=True))
            else:
                st.write("No prediction dataframe available.")

        # XGBoost Feature Importance (render on demand)
        if st.button("Show XGBoost Feature Importance"):
            try:
                ax = xgb.plot_importance(st.session_state['models']['xgb'], max_num_features=12, grid=False)
                # Get the figure associated with the axes and display it
                fig = ax.figure if hasattr(ax, 'figure') else plt.gcf()
                st.pyplot(fig)
                plt.clf()
            except Exception as e:
                st.write("Unable to plot feature importance:", e)
