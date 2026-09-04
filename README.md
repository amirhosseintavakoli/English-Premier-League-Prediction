# English-Premier-League-Prediction
See the website for the outcome
- https://english-premier-league-prediction.streamlit.app/

Please note that:
- It may take some time for the application to load
- Use a desktop browser to the best experience

This application allows you to train and evaluate various machine learning models to predict Premier League match outcomes based on historical match and player data. You can select different features to include in the model, train the models, and view their performance metrics. This demo is presented on Streamlit Community Cloud.

# Data Sources

* **Match data**: [football-data.co.uk](https://www.football-data.co.uk/englandm.php) per-season CSV archives.
* **Player data**: [understat.com](https://understat.com/)'s player stats (goals, assists, xG, xA, minutes, cards).

Both were originally scraped from fbref.com, which now blocks automated requests behind a Cloudflare challenge. See `football_data_source.py` and `player_data_source.py` for details.

Note: football-data.co.uk's per-season archives only publish **completed match results**, not the full season's fixture list. To still get predictions on upcoming matches, the app also pulls football-data.co.uk's combined fixtures feed and appends the next round or two of not-yet-played EPL fixtures (with no score yet). These are excluded from model training but are exactly the "test sample" the app predicts on — this is the intended way to use the app: train on past results, then read off predictions for the upcoming week. Only weeks that far ahead of the current date will show fixtures; anything further out isn't published yet.

# Features Selection

This application allows you to train and evaluate various machine learning models to predict Premier League match outcomes based on historical match and player data. You can select different features to include in the model, train the models, and view their performance metrics.

**Feature Dictionary:**

* Week: Match week number for the team in the season (categorical)
* IsHome: Whether the team is playing at home (True/False)
* TeamID: Unique identifier for the team (categorical)
* DayofWeek: Day of the week the match is played (1=Monday,...7=Sunday)
* RollingAvgX_Stat: Rolling average of 'Stat' over the past X matches
* LaggedX_Stat: Value of 'Stat' from X matches ago
* MaxPastX_Stat: Maximum value of 'Stat' over the past X matches
* MinPastX_Stat: Minimum value of 'Stat' over the past X matches
* Per90_G+A-PK_POS_QX: Number of players in position POS (FW/MF/DF) in quartile X (0=best,3=worst) based on (Goals+Assits-PenatlyKicks) per 90 mins in the past season

# Prediction
Using the sliders, you can see the match predictions for specific weeks based on each machine learning model, including the upcoming, not-yet-played week (see Data Sources above).

# XGBoost Feature Importance
This button allows you to observe the importance of selected features in the xgboost model.

# Future Path
This is a work in progress and I'd appreciate any comments or feedbacks. Feel free to play around with this application and send your comments and request new features.
