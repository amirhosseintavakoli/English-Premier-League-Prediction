import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
from xgboost import to_graphviz

class EPLAnalysis:
    def __init__(self):
        self.model = None
    
    def get_data(self):
        print(f"Data player data from FBref...")
        player_urls = {
            '2022-2023': 'https://fbref.com/en/comps/Big5/2021-2022/stats/players/2021-2022-Big-5-European-Leagues-Stats',
            '2023-2024': 'https://fbref.com/en/comps/Big5/2022-2023/stats/players/2022-2023-Big-5-European-Leagues-Stats',
            '2024-2025': 'https://fbref.com/en/comps/Big5/2023-2024/stats/players/2023-2024-Big-5-European-Leagues-Stats',
            '2025-2026': 'https://fbref.com/en/comps/Big5/2024-2025/stats/players/2023-2024-Big-5-European-Leagues-Stats'
        }
        all_players = []

        for season, url in player_urls.items():
            try:
                print(f"Processing season: {season}")
                season_players = pd.read_html(url)[0]
                season_players['Season'] = season
                all_players.append(season_players)
            except Exception as e:
                print(f"Failed to fetch data for season {season}: {e}")
        
        if all_players:
            players_df = pd.concat(all_players, ignore_index=True)
            print(f"Total players fetched: {len(players_df)}")
        else:
            print("No player data fetched.")
            return False
            
        print(f"Fetching data from FBref...")

        match_urls = {
            '2022-2023': 'https://fbref.com/en/comps/9/2022-2023/schedule/2022-2023-Premier-League-Scores-and-Fixtures',
            '2023-2024': 'https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures',
            '2024-2025': 'https://fbref.com/en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures',
            '2025-2026': 'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures'
        }

        all_matches = []

        for season, url in match_urls.items():
            try:
                print(f"Processing season: {season}")
                season_matches = pd.read_html(url)[0]
                season_matches['Season'] = season
                all_matches.append(season_matches)
            except Exception as e:
                print(f"Failed to fetch data for season {season}: {e}")
        
        if all_matches:
            matches_df = pd.concat(all_matches, ignore_index=True)
            print(f"Total matches fetched: {len(matches_df)}")
        else:
            print("No match data fetched.")
            return False
        
        # players_df  = self.clean_player_data(players_df)
        long_df     = self.clean_match_data(matches_df)
        

    # match-level data cleaning steps
    def clean_match_data(self, df):
        print("Cleaning match data...")
        df.rename(columns={'Home': 'HomeTeam', 
                                    'Away': 'AwayTeam', 
                                    'xG': 'HomexG', 
                                    'xG.1': 'AwayxG'}, inplace=True)
        
        ## split the Date into a year, month and date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['year']  = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day']   = df['Date'].dt.day

        ## split the Score into FullTimeHomeGoals and FullTimeAwayGoals
        df['FullTimeHomeGoals'] = [np.nan if pd.isna(x) else int(x.split("–")[0].strip()) for x in df['Score']]
        df['FullTimeAwayGoals'] = [np.nan if pd.isna(x) else int(x.split("–")[1].strip()) for x in df['Score']]

        ## create a stable unique ID for every team (covers HomeTeam and AwayTeam)
        teams = sorted(set(df['HomeTeam'].dropna().unique()).union(set(df['AwayTeam'].dropna().unique())))

        team_to_id = {name: i for i, name in enumerate(teams, start=1)}
        # add ID columns to the match dataframe
        df['HomeTeamID'] = df['HomeTeam'].map(team_to_id)
        df['AwayTeamID'] = df['AwayTeam'].map(team_to_id)

        df = df.reset_index(drop=True)
        df['MatchID'] = df.index + 1

        long_df = self.team_view(df)
        long_df['Outcome'] = long_df.apply(self.wl_from_scores, axis=1)
        long_df['Points'] = long_df.apply(self.points_from_scores, axis=1)

        print(long_df.columns)


        # sort the data by team ID and match ID and create a new variable indicating the teams' game number since beginning of each season
        long_df = long_df.sort_values(['Season', 'TeamID', 'MatchID']).reset_index(drop=True)
        long_df['Week'] = long_df.groupby(['Season', 'TeamID']).cumcount() + 1
        print(long_df['Week'].value_counts())
        long_df['Week'] = long_df['Week'].astype('int64')

        # reorder columns
        cols = ['Season', 'MatchID','Date', 'Week','year','month','day','Team','TeamID','IsHome',
                'GoalsFor','GoalsAgainst','Outcome', 'Points']
        long_df = long_df[[c for c in cols if c in long_df.columns]]

        long_df = self.calculate_rolling_averages(long_df)

        # compute the quality of players based on their G+A-PK per 90 minutes        
        long_df = long_df.merge(self.rank_players_by_ga(self.clean_player_data(players_df)), how='left',
                        left_on=['Season', 'Team'], right_on=['Season', 'Squad'])
        long_df = long_df.rename(columns={'num_players': 'NumTopQuartilePlayers'})
        long_df = long_df.drop(columns=['Squad']).fillna(0)

        return long_df

    # player-level data cleaning steps
    def clean_player_data(self, df):
        print("Cleaning player data...")    
        # rename columns to avoid multi-index label
        # remove the first part of the multi-index if it contains 'unnamed'
        df.columns = [col[1] if 'unnamed' in col[0].lower() else col for col in df.columns]
        df.columns = ['Per90_' + col[1] if 'per 90' in col[0].lower() else col for col in df.columns]
        df.columns = ['PT_' + col[1] if 'playing time' in col[0].lower() else col for col in df.columns]
        df.columns = [col[1] if 'progression' in col[0].lower() else col for col in df.columns]
        df.columns = [col[1] if 'performance' in col[0].lower() else col for col in df.columns]
        df.columns = [col[1] if 'expected' in col[0].lower() else col for col in df.columns]
        df.columns = [col[0] if 'season' in col[0].lower() else col for col in df.columns]

        df['Comp'] = df['Comp'].replace({'eng Premier League': 'EPL',
                                        'fr Ligue 1': 'Ligue 1',
                                        'it Serie A': 'Serie A',
                                        'de Bundesliga': 'Bundesliga',
                                        'es La Liga': 'La Liga'})
        # remove the nations's abbreviation from the player's nation. All the characters before the first capital letter
        df['Nation'] = df['Nation'].str.extract(r'([A-Z]{3})', expand=False)

        # keep only EPL players
        df = df[df['Comp'] == 'EPL']

        # consider the first position only
        df['Pos'] = df['Pos'].str.split(',').str[0]

        # drop unnecessary columns
        df = df.drop(columns=['Rk', 'Born', 'Comp', 'Matches'])

        return df

    # create long format of the match data
    def team_view(self, df):
        # home perspective
        home = pd.DataFrame({
            'Season': df['Season'],
            'MatchID': df['MatchID'],
            'Date': df['Date'],
            'year': df['year'],
            'month': df['month'],
            'day': df['day'],
            'Team': df['HomeTeam'],
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
            'Team': df.get('AwayTeam'),
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
        return pd.concat([home, away], ignore_index=True)

    # add outcome from full-time scores (W/L/D from each team's perspective)
    def wl_from_scores(self, row):
        gf = row['GoalsFor']
        ga = row['GoalsAgainst']
        if pd.isna(gf) or pd.isna(ga):
            return None
        if gf > ga:
            return 'W'
        if gf < ga:
            return 'L'
        return 'D'

    def points_from_scores(self, row):
        gf = row['GoalsFor']
        ga = row['GoalsAgainst']
        if pd.isna(gf) or pd.isna(ga):
            return None
        if gf > ga:
            return 3
        if gf < ga:
            return 0
        return 1
    
    def calculate_rolling_averages(self, df):
        # roll_cols = ['GoalsFor', 'GoalsAgainst', 'Shots', 'ShotsOnTarget', 'Corners', 'Fouls',
        #              'YellowCards', 'RedCards', 'ShotsAgainst', 'ShotsOnTargetAgainst',
        #              'CornersAgainst', 'FoulsAgainst', 'YellowCardsAgainst', 'RedCardsAgainst']
        roll_cols = ['IsHome','GoalsFor','GoalsAgainst','Outcome']
        
        for col in roll_cols:
            df[f'{col}_roll3'] = df.groupby(['Season', 'TeamID'])[col].transform(lambda x: x.shift().rolling(window=3, min_periods=1).mean())
            df[f'{col}_roll5'] = df.groupby(['Season', 'TeamID'])[col].transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
            df[f'{col}_roll10'] = df.groupby(['Season', 'TeamID'])[col].transform(lambda x: x.shift().rolling(window=10, min_periods=1).mean())
        
        return df
    
    # rank players by position and season based on their Per90_G+A-PK
    def rank_players_by_ga(self, df):
        df['Per90_G+A-PK'] = df['Per90_G+A-PK'].astype(float)

        # partition players into quartiles based on their Per90_G+A-PK by position and season
        df['Quartile'] = df.groupby(['Season', 'Pos'])['Per90_G+A-PK'].transform(lambda x: pd.qcut(x, 4, labels=False, duplicates='drop'))

        df = df.groupby(['Season', 'Pos', 'Squad', 'Quartile']).agg( num_players = ('Player', 'nunique') ).reset_index()

        # filter to show only the top quartile players (Quartile 3)
        num_players_in_q4 = df[df['Quartile'] == 3].reset_index()
        num_players_in_q3 = df[df['Quartile'] == 2].reset_index()
        num_players_in_q2 = df[df['Quartile'] == 1].reset_index()
        num_players_in_q1 = df[df['Quartile'] == 0].reset_index()

        num_players_in_q4 = num_players_in_q4.pivot(index=['Season', 'Squad'], columns='Pos', values='num_players').fillna(0).reset_index()
        
        return num_players_in_q4


    def feature_engineering(self):
        # Add your feature engineering steps here
        pass

    def train_model(self, X, y):
        # Add your model training steps here
        pass

    def evaluate_model(self, model, X_test, y_test):
        # Add your model evaluation steps here
        pass

# Example usage
if __name__ == "__main__":
    
    analysis = EPLAnalysis()    
    if analysis.get_data():
        print("Data fetched and cleaned successfully.")
    else:
        print("Failed to fetch or clean data.")