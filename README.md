# English-Premier-League-Prediction
This project provide prediction on 2025/2026 EPL games based on EPL historic data

## Phase 1: Project Setup & Planning (Day 1–2)
- Define project goals and scope (match outcome prediction first).
- Create a GitHub repo with a clear README and project structure.
- Set up a Python environment (e.g., venv or conda) and install key libraries:
  pip install pandas numpy scikit-learn matplotlib seaborn xgboost

## Phase 2: Data Acquisition (Day 3–5)
- Use existing historical match data.
- Search for free APIs or datasets:
  • Football-Data.org
  • FBref
  • Kaggle Datasets
- Collect features like:
  • Team stats (goals, possession, shots)
  • Match metadata (home/away, date)
  • Recent form (last 5 matches)
  • Head-to-head history

## Phase 3: Data Cleaning & Feature Engineering (Day 6–9)
- Handle missing values, normalize formats.
- Encode categorical variables (e.g., team names).
- Create derived features:
  • Rolling averages (e.g., goals scored in last 5 games)
  • Home/away performance
  • Elo ratings or team strength scores

## Phase 4: Model Development (Day 10–15)
- Split data into train/test sets.
- Try baseline models:
  • Logistic Regression
  • Random Forest
  • XGBoost
- Evaluate using metrics:
  • Accuracy, F1-score, Confusion Matrix
- Use cross-validation for robustness.

## Phase 5: Model Tuning & Analysis (Day 16–18)
- Hyperparameter tuning (GridSearchCV or RandomizedSearchCV).
- Feature importance analysis.
- Save models using joblib or pickle.

## Phase 6: Season Simulation (Optional, Day 19–21)
- Use match predictions to simulate a full season.
- Aggregate points to predict league winner.
- Run multiple simulations to estimate probabilities.

## Phase 7: Visualization & Reporting (Day 22–25)
- Create plots:
  • Match prediction accuracy
  • Feature importance
  • League table simulation
- Use matplotlib, seaborn, or Plotly.

## Phase 8: Deployment & GitHub Showcase (Day 26–30)
- Clean up code and notebooks.
- Write a detailed README:
  • Project overview
  • Data sources
  • Modeling approach
  • Results and insights
- Optional: Build a dashboard using Streamlit or Dash.
- Push everything to GitHub with clear commits and structure.

## Tools & Libraries
- Python: Core language
- scikit-learn / XGBoost: ML models
- Pandas / NumPy: Data manipulation
- Matplotlib / Seaborn / Plotly: Visualization
- Streamlit / Dash: Optional dashboard
- GitHub: Project hosting

## How to create an evnironment on Github
- python -m venv pl_env
- pip install pandas numpy scikit-learn matplotlib seaborn xgboost
- pip freeze > requirements.txt


# Progress
2025-09-16: 

The scraping data from FBRef is way easier than finding an API
Greate guide: https://medium.com/@ricardoandreom/how-to-scrape-and-personalize-data-from-fbref-with-python-a-guide-to-unlocking-football-insights-7e623607afca

2025-09-23:
To Do: I can identify the top player within each league and match them with their current team. In general, the current version does not consider the current squad performance.
In the upcoming versions, considering transfers i.e. inflow and outflow of player, within/across league transfers, and the quality of transfer can help with the prediction. Not surprisingly, this analysis is quite similar to firm-leve and worker-level analysis.
