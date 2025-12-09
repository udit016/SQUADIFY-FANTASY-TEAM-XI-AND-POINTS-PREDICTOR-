# Squadify: End-to-End IPL Fantasy Score Prediction Pipeline
Squadify is an advanced Sports Analytics platform developed to predict IPL fantasy cricket scores with high precision. By leveraging a massive historical dataset spanning from 2008 to the present, 
the system forecasts fantasy points for upcoming matches and generates the best possible 11 players to maximize winning potential.

The pipeline automates the entire lifecycle of fantasy prediction—from web scraping decades of raw match data to using variance-focused Machine Learning to construct the "Perfect XI."

# Features
1. Developed an end-to-end data engineering and machine learning pipeline for IPL fantasy score prediction.
2. Designed and deployed ETL workflows with Apache Airflow and Docker, enabling automated Web Scraping, Data Transformation, and ingestion into PostgreSQL.
3. Performed advanced feature engineering, deriving 30+ context-aware statistics from ball-by-ball, player, and venue data to enable variance-sensitive predictions.
4. Implemented Dream11 rule-based optimization to generate valid fantasy teams and maximize user score potential.

# Tech Stack
1. Core Language:	Python
2. Machine Learning:	XGBoost, scikit-learn
3. Data Analysis:	Pandas, NumPy
4. Orchestration & Containerization:	Apache Airflow, Docker
5. Database:	PostgreSQL

# Enhanced Feature Engineering
The features are engineered into six strategic categories:
1. 10 Recent Performance: Raw values (runs, wickets, fantasy points) from the last 3 matches to capture immediate form volatility.

2. 5 Player Archetype: Current-year ceiling/floor analysis (e.g., max_fantasy_score, scores_over_60_count).

3. 5 Ball-by-Ball Context: Aggression metrics including powerplay boundaries and death-over run rates.

4. 4 Opposition Context: Specific matchups against the opponent's bowling attack structure.

5. 4 Venue Context: Historical ground scoring patterns, boundary rates, and pitch difficulty.

6. 2 Match Context: Situational flags for high-scoring venues and win probability.

# Model Training & Evaluation
The Squadify engine is powered by an XGBoost Variance Model trained on a massive historical dataset. 
Unlike standard regression models that regress to the mean, this model is specifically tuned to maximize prediction variance, 
ensuring it captures the high-ceiling potential of top-tier players.

1. Training Performance
Algorithm: XGBoost Regressor (Variance-Tuned)

Training Samples: 14,503 historical player innings.

Feature Set: 30 Context-Aware Features.

Training R² Score: 0.838 (Indicates strong fit to historical patterns).

Training RMSE: 16.45

Prediction Standard Deviation: 28.9 (Threshold: 10.0) — This high deviation confirms the model successfully differentiates between average performers and match-winners.

2. Variance & "Average Trap" Analysis
The primary success metric for Squadify is avoiding the "Averaging Trap" (where models predict safe scores of 30-40 points for everyone).

Variance Score: 0.666

Avoiding Average Trap: YES

Prediction Range: The model successfully predicts scores ranging from -1.1 to 108.0, capturing the full volatility of T20 cricket.

3. Test Evaluation (Unseen Data)
The model was evaluated on a dedicated test set covering the period April 01, 2024 – May 01, 2024 (838 test matches).

Test Samples: 838

Test RMSE: 49.94

Test Prediction Std: 15.7

# Results & Prediction Strategy
The Squadify system successfully executed its end-to-end pipeline, moving from raw feature extraction to the generation of a valid, high-potential fantasy team.

1. XGBoost Variance Predictions
The core model generated predictions for all 37 squad members, successfully avoiding conservative averages to identify high-variance "match winners".

2. Dream11 Optimization (The "Perfect XI")
To convert raw predictions into a playable team, the system employed a  optimizer. It selected the best 11 players while strictly adhering to official fantasy sports constraints:
Constraints Applied:

1. Team Size: Exactly 11 players (No duplicates).

2. Role Constraints: Enforced valid counts for Wicketkeepers (1-4), Batters (3-6), All-Rounders (1-4), and Bowlers (3-6).

3. Squad Balance: Max 7 players from a single team.

4. Starting Probability: Filtered for players with high start confidence (Avg: 93.9%).
