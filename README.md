# English-Premier-League-Prediction
See the website for the outcome (it may take some time for the application to load) \n
https://english-premier-league-prediction.streamlit.app/

This application allows you to train and evaluate various machine learning models to predict Premier League match outcomes based on historical match and player data. You can select different features to include in the model, train the models, and view their performance metrics.

# Features Selection

This application allows you to train and evaluate various machine learning models to predict Premier League match outcomes based on historical match and player data. You can select different features to include in the model, train the models, and view their performance metrics.

Feature Dictionary: \n
Week: Match week number for the team in the season (categorical) \n
IsHome: Whether the team is playing at home (True/False)
TeamID: Unique identifier for the team (categorical)
DayofWeek: Day of the week the match is played (1=Monday,...7=Sunday)
RollingAvgX_Stat: Rolling average of 'Stat' over the past X matches
LaggedX_Stat: Value of 'Stat' from X matches ago
MaxPastX_Stat: Maximum value of 'Stat' over the past X matches
MinPastX_Stat: Minimum value of 'Stat' over the past X matches
Per90_G+A-PK_POS_QX: Number of players in position POS (FW/MF/DF) in quartile X (0=best,3=worst) based on (Goals+Assits-PenatlyKicks) per 90 mins in the past season

# Prediction
Using the sliders, you can see the match predictions for specific weeks based on each machine learning model.

# XGBoost Feature Importance
This button allows you to observe the importance of selected features in the xgboost model.

# Future Path
This is a work in progress and I'd appreciate any comments or feedbacks. Feel free to play around with this application and send your comments and request new features.