# Predicting Football Player Market Values in Europe’s Top 5 Leagues

## 1. Introduction – what I am doing

The goal of this project is to build machine learning models that predict
football player market values in Europe’s “Big Five” leagues
(English Premier League, La Liga, Serie A, Bundesliga and Ligue 1).
I approximate market value using Transfermarkt’s estimated market values
and try to predict them from season-level performance statistics
collected from SofaScore using an API, together with age and league information
from Transfermarkt using a scraper.

Concretely, I formulate this as a supervised regression problem on the
logarithm of a player’s market value. Concretely, I formulate the task
as a supervised regression problem on the logarithm of a player’s market value.
Instead of a single global model, I train fully position-specific gradient
boosting models (one per role: GK/DF/MF/FW) and evaluate them on a held-out
test season (2024/25). I then use SHAP to interpret which performance metrics
drive player value by position.

The models are evaluated on a hold-out test season (2024/25), which
simulates the realistic task of predicting values for an upcoming
season from past seasons’ data.

---

## 2. Data – what data I am using

The project is based on data that I have collected my self scraping Transfermarkt
and using an API to get relevant performance metrics from Sofascore:

- `big5_players_clean.parquet`

This file contains one row per **player–season** in the Big Five leagues,
for multiple seasons leading up to and including **2024/25**.
The main elements are:

- **Target**
  - `market_value_eur`: Transfermarkt-estimated market value (EUR)
  - `log_market_value`: log(1 + market_value_eur), used as the regression target

- **Contextual variables**
  - `season`: season identifier (e.g. `"2021/22"`, `"2024/25"`)
  - `league`: league name
  - `club`: club name
  - `position_group`: one of `GK`, `DF`, `MF`, `FW`
  - `age`: age in years during the season

- **Performance statistics (aggregated over the season)**
  - Shooting and scoring: goals, assists, shots, shots on target, expected goals
  - Passing: key passes, accurate passes, passes into final third, pass accuracy
  - Possession and dribbling: dribbles, successful dribbles, carries
  - Defensive actions: tackles, interceptions, clearances, duels won, aerial duels
  - Minutes played and appearances

To improve comparability across players with different playing time,
volume stats are mostly converted into per-90 values during preprocessing
(e.g. `sofa_goals_per90`, `sofa_tackles_per90`).

I restrict the dataset to players with **a minimum number of minutes**
(≥ 450, which corresponds to around 5 full games) and non-missing market
values to avoid extreme noise from fringe players. The 2024/25 season is 
**completely held out** as testdata; all earlier seasons are used for model 
training and tuning.

---

## 3. Features and target

### Target

The supervised learning target is:

- **`log_market_value`** = log(1 + `market_value_eur`)

Using the logarithm stabilises variance between very cheap and very
expensive players and reduces the impact of a few superstars on the
loss function. Final predictions are transformed back to euros using
the inverse transformation `exp(log_value) - 1`.

### Features

From `big5_players_clean.parquet` I construct the following feature set:

- **Demographics and context**
  - `age` and `age_sq` (age squared to capture non-linear age curves)
  - `league` (categorical)
  - `season` (used only for splitting, not as a direct predictor)
  - `position_group` (used for splitting by position, not inside the models)

- **Playing time**
  - `minutes` and derived per-90 scaling factors

- **Per-90 performance stats**
  - Attacking: goals_per90, assists_per90, shots_per90,
    shots_on_target_per90, expected_goals_per90
  - Creativity: key_passes_per90, passes_into_final_third_per90
  - Defensive: tackles_per90, interceptions_per90, clearances_per90,
    total_duels_won_per90, aerial_duels_won_per90, ground_duels_won_per90
  - Possession/dribbling: successful_dribbles_per90, etc.

- **Percentage/efficiency stats**
  - duel win percentages, aerial duel %, dribble success %,
    pass accuracy %, etc. where available

During feature preparation I explicitly **drop potential leakage and ID
information**, such as:

- `market_value_eur`, `market_value`, `log_market_value` (target variants),
- direct identifiers (`player_id`, `sofa_player_id`, `sofa_team_id`, names, date of birth).

This is important to prevent the models from implicitly “looking up”
the target via IDs or perfectly correlated proxies.

The final feature space is split into numeric and categorical variables
for use in scikit-learn pipelines with imputation and one-hot encoding.

---

## 4. Methods – which ML models and why

**Position-specific LightGBM regressors**

   - Gradient-boosted decision trees (LightGBM) trained **separately for
     each position group**: goalkeepers (GK), defenders (DF),
     midfielders (MF) and forwards (FW).
   - Motivations:
     - The relationship between features and value is highly
       position-dependent: saves and goals conceded matter for GKs,
       duels and clearances for defenders, creativity and progression
       for midfielders, scoring and chance quality for forwards.
     - Tree-based models naturally handle non-linearities and complex
       interactions, such as age curves or thresholds in playing time.
   - Each position-specific model is trained on log market values and
     evaluated on the corresponding subset of 2024/25 players.

For interpretability, I use **SHAP (SHapley Additive Explanations)** to:

- compute global feature importance for the LightGBM models, and
- inspect which features contribute positively or negatively to the
  predicted log market value for different positions.

---

## 5. Hyper-parameter Tuning – Summary

Hyper-parameter tuning is fully automated inside the pipeline and runs separately for each position group (GK/DF/MF/FW):

- I use Optuna (≈60 trials per position) to tune a LightGBM model on the preprocessed training data. The search  
  optimises learning rate, depth, leaves, sampling, and regularisation, using 3-fold CV on
  log_market_value.

    -   I used Optuna because it provides an efficient and automated way to tune complex gradient-boosting 
        models. Unlike manual grid search, Optuna uses Bayesian optimisation to explore hyper-parameters intelligently, focusing on promising regions rather than testing every combination. This makes it significantly faster and more effective for models like LightGBM, where parameters interact non-linearly.

- After tuning, I train the best LightGBM model and use SHAP to select the top 25 most important features for    
  that position. This reduces noise and keeps only the strongest predictors.

- Final models are then fitted on the reduced feature set:

    - GK → tuned LightGBM

    - DF/MF/FW → ensemble (tuned LightGBM + XGBoost + CatBoost) stacked with an ElasticNet meta-model.

This approach ensures tuning is reproducible, position-specific, and computationally efficient while maximising out-of-sample performance.

---

## 6. Evaluation – results on out-of-sample test data

All models are evaluated on the **2024/25 season**, which was completely
held out during training and tuning. Metrics are computed in the
original euro scale:

- **R²** (coefficient of determination),
- **MAE** (mean absolute error, in euros),
- **MAPE** (mean absolute percentage error).

For the final position-specific LightGBM models I obtain approximately:

- **Goalkeepers (GK)**  
  - R² ≈ **49.2%**  
  - MAE ≈ **€4.26m**  
  - MAPE ≈ **60.4%**

- **Defenders (DF)**  
  - R² ≈ **63.9%**  
  - MAE ≈ **€4.91m**  
  - MAPE ≈ **67.0%**

- **Midfielders (MF)**  
  - R² ≈ **60.4%**  
  - MAE ≈ **€6.10m**  
  - MAPE ≈ **56.8%**

- **Forwards (FW)**  
  - R² ≈ **64.0%**  
  - MAE ≈ **€6.65m**  
  - MAPE ≈ **74.6%**

- **All positions combined**  
  - R² ≈ **63.3%**  
  - MAE ≈ **€5.66m**  
  - MAPE ≈ **65.5%**

These numbers show that the models explain a substantial share of the
variance in market value, especially for defenders, midfielders and
forwards. Performance for goalkeepers is noticeably lower, which is
consistent with the more limited and noisier set of GK features
available.

Percentage errors (MAPE) are relatively high, especially for forwards.
This is partly because:

- market values span several orders of magnitude (from ~€0.1m to
  well over €100m),
- very cheap players make even modest absolute errors look large in
  relative terms, and
- superstar forwards are particularly hard to model because their
  valuations reflect hype, potential and commercial value that are not
  present in the input data.

SHAP analysis supports the quantitative results:

- **Age and age²** are dominant drivers across positions, reflecting
  typical peak age curves.
- **League effects** (especially playing in the English Premier League)
  have strong positive contributions.
- **Playing time and reliability** (minutes played, accurate passes,
  appearances) matter across all outfield positions.
- **Position-specific stats** (saves and goals conceded for GKs, duels
  and clearances for defenders, creativity for midfielders, goals and
  chance quality for forwards) show up as important features in the
  corresponding models.

---

## 7. Concluding remarks and learning

This project demonstrates that:

- A relatively standard machine-learning pipeline can capture a meaningful 
  fraction of the structure in football player market values.
- Position-specific models make sense: the determinants of value are
  clearly different for goalkeepers, defenders, midfielders and forwards.
- Even with fairly rich performance data, there remains substantial
  unexplained variance, especially for low-value squad players and
  high-profile forwards.

From a learning perspective, the main takeaways for me are:

- **Data preparation and leakage control**: I saw how easily IDs or
  near-target columns can artificially inflate performance if not
  removed. Designing a clean feature space is at least as important as
  the choice of model.
- **Target transformations**: Modelling `log_market_value` instead of
  raw euros produced more stable training and more meaningful errors
  across the value distribution.
- **Temporal validation**: Splitting by season rather than random rows
  is crucial when the real task is to predict future seasons.
- **Model comparison and interpretability**: The combination of position-specific 
  gradient-boosted models and SHAP explainability provided both strong
  predictive accuracy and interpretable insights, which is essential 
  for realistic scouting and valuation work.


I did use Generative AI (e.g. ChatGPT) during the project, mainly as a
coding assistant for structuring the pipeline, debugging sklearn /
LightGBM / SHAP integration, and refining the documentation. However, I
validated the code, inspected the outputs and interpreted the results
myself, and I am confident that I understand the modelling choices and
their implications.

If I had more time and data, the next steps would include:

- incorporating **temporal dynamics** (multiple seasons per player and
  value trajectories),
- including contract, wage and injury information where available,

Overall, the project gave me hands-on experience with end-to-end
machine-learning for a realistic football analytics problem, from raw
data to model evaluation and interpretation.


## Appendix - How to run this project in local execution

pip install lightgbm xgboost shap scikit-learn pandas numpy matplotlib

$env:PYTHONPATH = "src"

python -m scouting_ml.models.train_market_value_full `<br>  
--dataset "data/model/big5_players_clean.parquet" `<br>  
--test-season "2024/25" `<br>  
--output "data/model/big5_predictions_full_v2.csv" `<br>  
--trials 60<br>   

Outputs then will appear in: 

data/model/<br>  
logs/shap/

