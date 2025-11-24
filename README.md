# Predicting Football Player Market Values in Europe’s Top 5 Leagues

## 1. Introduction – what I am doing

The goal of this project is to build machine learning models that predict
football player market values in Europe’s “Big Five” leagues
(English Premier League, La Liga, Serie A, Bundesliga and Ligue 1).
I approximate market value using Transfermarkt’s estimated market values
and try to predict them from season-level performance statistics
collected from SofaScore, together with age and league information.

Concretely, I formulate this as a supervised regression problem on the
logarithm of a player’s market value. I compare a simple global baseline
model to a more sophisticated, position-specific gradient boosting
approach and then analyse feature importance with SHAP to better
understand which attributes drive value for goalkeepers, defenders,
midfielders and forwards.

The models are evaluated on a hold-out test season (2024/25), which
simulates the realistic task of predicting values for an upcoming
season from past seasons’ data.

---

## 2. Data – what data I am using

The project is based on a pre-processed dataset:

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
(≥ 450) and non-missing market values to avoid extreme noise from
fringe players. The 2024/25 season is **completely held out** as test
data; all earlier seasons are used for model training and tuning.

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

I apply two main types of models:

1. **Baseline model: global Ridge regression**

   - A linear model with L2 regularisation (Ridge) trained on all
     positions combined.
   - Categorical features are one-hot encoded, numeric features are
     standardised.
   - This baseline is simple, fast to train, and provides a transparent
     reference level of performance.

2. **Main model: position-specific LightGBM regressors**

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

This combination (linear baseline + tree-based models + SHAP) balances
technical sophistication with interpretability, which is important for
a realistic football scouting use case.

---

## 5. Hyper-parameter tuning – my approach

Hyper-parameter tuning is done in two stages:

1. **Exploratory tuning (outside the notebook)**

   - I first conducted exploratory tuning runs for LightGBM using
     simple grid search and, in some experiments, Optuna on the training
     seasons only.
   - The objective was to minimise the RMSE on the **log market value**
     using 3-fold cross-validation.
   - These runs suggested a reasonably stable region of good parameters
     (e.g. several hundred to ~1500 trees, moderate learning rate and
     subsampling).

2. **Fixed, documented configuration (inside the notebook)**

   - To keep the final notebook reproducible and within reasonable
     runtime, I fix a single LightGBM configuration per position
     (same structure, possibly slightly adjusted depth/regularisation).
   - I document the chosen hyper-parameters in the notebook and apply
     them consistently for training on all pre-2024/25 seasons and
     evaluation on 2024/25.
   - The Ridge baseline uses a small `GridSearchCV` over α to pick the
     regularisation strength that minimises validation MAE.

This approach shows that I understand hyper-parameter tuning in
principle, but keeps the final deliverable focused and efficient.

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

- A relatively standard machine-learning pipeline (feature engineering,
  leakage control, baseline linear model, tree-based models, and SHAP)
  can capture a meaningful fraction of the structure in football player
  market values.
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
- **Model comparison and interpretability**: Combining a simple
  baseline, a more flexible tree-based model and SHAP allowed me to
  not only improve accuracy but also explain which features were driving
  the predictions in football terms.

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
- experimenting with **ranking losses** or quantile regression that
  better reflect scouting and transfer decision-making.

Overall, the project gave me hands-on experience with end-to-end
machine-learning for a realistic football analytics problem, from raw
data to model evaluation and interpretation.


## Appendix - How to run this project in local execution

pip install lightgbm xgboost shap scikit-learn pandas numpy matplotlib

python -m scouting_ml.models.train_market_value_full `
>>   --dataset "data/model/big5_players_clean.parquet" `
>>   --test-season "2024/25" `
>>   --output "data/model/big5_predictions_full_v2.csv" `
>>   --trials 60
>> 

Outputs then will appear in: 

data/model/
logs/shap/

