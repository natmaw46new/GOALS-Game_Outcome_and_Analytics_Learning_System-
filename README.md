# GOALS — Game Outcome and Analytics Learning System

> *"Given recent player performance trends, what can we expect to happen in the next match?"*

**GOALS** is a multi-stage machine learning pipeline that predicts Premier League match outcomes (Win / Draw / Loss) by constructing position-specific composite performance scores from per-player match statistics, then aggregating those scores to the team level.

**Course:** EECE5644 — Introduction to Machine Learning & Pattern Recognition
**Team:** Amine Kebichi · Nathaniel Maw
**League:** Premier League — all 20 clubs, 4 seasons (2021/22 – 2024/25)
**Deadline:** April 18, 2026

---

## Pipeline Overview

```
FotMob per-match player stats (Premier League)
              │
              ▼
    02_preprocessing       ← cleaning, context features, null handling
              │
              ▼
 03_feature_engineering    ← ratio features, rolling windows, composite scores
              │
              ▼
          04_eda            ← distributions, correlations, PCA
              │
              ▼
      05_regression         ← predict composite scores (Ridge + RF)
              │
              ▼
  07_team_aggregation       ← aggregate player predictions to match level
              │
        ┌─────┴──────┐
        ▼             ▼
06_clustering   08_classification
(archetypes)    (Win / Draw / Loss)
```

| Stage | Notebook | Status | Owner |
|-------|----------|--------|-------|
| Preprocessing | `02_preprocessing.ipynb` | ✅ Complete | Both |
| Feature Engineering | `03_feature_engineering.ipynb` | ✅ Complete | Both |
| EDA | `04_eda.ipynb` | ✅ Complete | Both |
| Regression | `05_regression.ipynb` | ✅ Complete | Amine |
| Team Aggregation | `07_team_aggregation.ipynb` | ✅ Complete | Both |
| Clustering | `06_clustering.ipynb` | ⏳ In progress | Nathaniel |
| Classification | `08_classification.ipynb` | ⏳ In progress | Nathaniel |

---

## Research Questions

- How strongly does recent individual player form predict team-level match outcomes?
- Which statistical metrics (xG, progressive passes, defensive actions) are the most reliable indicators of future performance?
- Do data-driven player archetypes correspond to recognisable tactical roles?
- How effectively can aggregated player-level predictions forecast the final result?

---

## Dataset

### Sources

| Source | Role | Seasons | Status |
|--------|------|---------|--------|
| **FotMob** | Primary — per-match player stats: dribbles, tackles, saves, xG, xA, chances created, player ratings | 2021/22–2024/25 | ✅ Scraped (~72% of matches) |
| **FBref** | Supplementary — season-level player stats | 2021/22–2024/25 | ✅ Scraped (not yet merged) |

FotMob's proprietary per-match player rating serves as an **independent benchmark** for validating the composite scores built in this project.

### Scale

| Property | Value |
|----------|-------|
| Total player-match observations | 21,242 outfield + 2,138 GK |
| Post-feature-engineering (train) | 15,288 outfield + 1,549 GK |
| Post-feature-engineering (test) | 4,748 outfield + 483 GK |
| Match-level aggregation (train) | 794 matches |
| Match-level aggregation (test) | 250 matches |
| Features per match (team aggregation) | 37 columns |

~28% of matches are unavailable — FotMob serves those via authenticated endpoints only.

### Train / Test Split

Strictly **temporal** — random shuffling is never used, as it would allow future information to leak into training.

| Partition | Seasons | Matches |
|-----------|---------|---------|
| Training | 2021/22, 2022/23, 2023/24 | 794 |
| Testing | 2024/25 | 250 |

Within the three training seasons, hyperparameter tuning uses **time-series-aware cross-validation** (`TimeSeriesSplit`).

---

## Position-Aware Composite Performance Scores

Four position-specific scores that reflect the distinct tactical contribution of each role. All input metrics are **z-score normalised** on the training set before weights are applied. A **5-match rolling mean** (`.shift(1).rolling(5)`) is applied per player — the current match is never in its own window.

### Attacker Score
```
ATT = 0.25·(Goals+Assists) + 0.20·xG + 0.15·xA + 0.15·Dribbles
    + 0.10·ShotsOnTarget   + 0.10·ChancesCreated + 0.05·Recoveries
```

### Midfielder Score
```
MID = 0.20·AccuratePasses + 0.20·ChancesCreated + 0.15·xA + 0.15·(Goals+Assists)
    + 0.15·TacklesWon     + 0.10·Interceptions   + 0.05·Recoveries
```

### Defender Score
```
DEF = 0.25·TacklesWon + 0.20·AerialDuelsWon + 0.20·Clearances
    + 0.15·Interceptions + 0.10·Blocks + 0.10·AccuratePasses
```

### Goalkeeper Score
```
GK = 0.30·Saves + 0.25·xGOTFaced + 0.15·DivingSaves
   + 0.15·SavesInsideBox + 0.10·HighClaims + 0.05·SweeperActions
```

### Weight Scheme Validation

Three schemes compared by Pearson r against FotMob `rating_title` (independent benchmark):

| Scheme | Description | r (all outfield) | RMSE |
|--------|-------------|-------------------|------|
| **A** | **Proposal weights (canonical)** | **+0.191** | **6.731** |
| B | Equal weights | +0.179 | 6.747 |
| C | Data-driven (correlation with rating) | +0.199 | 6.743 |

Scheme A selected: domain-motivated weights, lowest RMSE, avoids overfitting to FotMob's subjective ratings.

---

## Results

### Regression (`05_regression.ipynb`)

Target: `composite_score` (continuous). Best model: **Random Forest, position-grouped, enhanced features**.

| Position | RMSE | R² | n (test) |
|----------|------|----|----------|
| Defender | 0.0923 | **0.966** | 1,747 |
| Midfielder | 0.1163 | **0.941** | 977 |
| Winger | 0.1372 | **0.942** | 1,061 |
| Forward | 0.1409 | **0.938** | 507 |
| **Overall** | — | **0.947** | 4,292 |

Top feature: `roll5_defensive_actions` (importance 0.200). Enhanced context features (`roll5_team_goals_scored`, `roll5_opp_goals_scored`) improved R² by ~2pp over baseline-only features.

### Team Aggregation (`07_team_aggregation.ipynb`)

Composite differential validates signal — home team out-composites away team when winning:

| Outcome | Mean composite diff (home − away) |
|---------|----------------------------------|
| Home win | **+0.172** |
| Draw | −0.024 |
| Away win | **−0.178** |

Top feature correlations with outcome: `h_mean_composite` r=+0.338, `a_mean_composite` r=−0.318.

---

## Repository Structure

```
GOALS/
├── fotmob_final.ipynb              # FotMob scraper (LEAGUE_ID=47, Premier League)
├── GOALS_notebook.ipynb            # FBref scraper (data already collected)
├── findings.md                     # Full pipeline metrics and design decisions
├── data/
│   ├── FBref/
│   │   └── premier_league/{season}/   # standard, shooting, misc, goalkeeping, playing_time
│   ├── 47/{season}/output/            # FotMob Premier League parquets
│   │   ├── outfield_players.parquet
│   │   ├── goalkeepers.parquet
│   │   └── fixtures.parquet
│   ├── processed/
│   │   ├── outfield_clean.parquet     # 21,242 × 53
│   │   ├── gk_clean.parquet           # 2,138 × 53
│   │   ├── datasets/
│   │   │   ├── outfield_train_scaled.parquet
│   │   │   ├── outfield_test_scaled.parquet
│   │   │   ├── gk_train_scaled.parquet
│   │   │   └── gk_test_scaled.parquet
│   │   ├── match_features_train.parquet  # 794 × 37
│   │   └── match_features_test.parquet   # 250 × 37
│   └── models/
│       ├── scalers_outfield.pkl       # 4 RobustScalers (per position group)
│       ├── scaler_gk.pkl
│       ├── position_map.pkl
│       ├── rf_regression_defender.pkl
│       ├── rf_regression_midfielder.pkl
│       ├── rf_regression_winger.pkl
│       └── rf_regression_forward.pkl
└── notebooks/
    ├── 02_preprocessing.ipynb
    ├── 03_feature_engineering.ipynb
    ├── 04_eda.ipynb
    ├── 05_regression.ipynb
    ├── 06_clustering.ipynb            # ⏳ Nathaniel
    ├── 07_team_aggregation.ipynb
    └── 08_classification.ipynb        # ⏳ Nathaniel
```

`data/` is git-ignored — all data files are local only.

---

## Algorithms

| Algorithm | Stage | Notes |
|-----------|-------|-------|
| **Ridge Regression** | Regression | L2 regularisation handles multicollinearity between football stats (`alpha=100`) |
| **Random Forest Regressor** | Regression | Position-grouped; `TimeSeriesSplit` CV for tuning |
| **K-Means** | Clustering | Elbow + Silhouette evaluation; optimal k per position group |
| **Logistic Regression** | Classification | `class_weight='balanced'`; GridSearchCV over C |
| **Random Forest Classifier** | Classification | `class_weight='balanced'`; macro F1 primary metric |

### Baselines

| Task | Baseline |
|------|----------|
| Regression | Mean composite score per position (R²=0.018) |
| Classification | Majority-class predictor ("Home Win"); last-match result |
| Clustering | Random cluster assignment Silhouette score |

---

## Key Design Constraints

| Rule | Reason |
|------|--------|
| Temporal train/test split only | Shuffling leaks future match results into training |
| Fit all scalers/imputers on train only | Prevents test contamination |
| `class_weight='balanced'` for all classifiers | Premier League has ~45% home wins, ~25% draws |
| 5-match rolling with `.shift(1)` | Current match never in its own feature window |
| FotMob rating as validation signal only | Independent benchmark — not a training target |

---

## Team

| Member | Responsibilities |
|--------|-----------------|
| **Amine Kebichi** | FotMob scraping, preprocessing, feature engineering, regression, team aggregation |
| **Nathaniel Maw** | Clustering, classification, visualisation |
| **Both** | EDA, evaluation, final report |

---

## References

1. Sports Reference LLC, *FBref advanced football statistics*, fbref.com, 2024.
2. FotMob, *Per-match player statistics and ratings*, fotmob.com, 2024.
3. Scikit-learn developers, *scikit-learn: Machine Learning in Python*, JMLR 12, 2011.
