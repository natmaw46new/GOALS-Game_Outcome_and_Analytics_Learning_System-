# GOALS — Game Outcome and Analytics Learning System

> *"Given recent player performance trends, what can we expect to happen in the next match?"*

**GOALS** is a multi-stage machine learning pipeline that predicts Premier League match outcomes
(Home Win / Draw / Away Win) by constructing position-specific composite performance scores from
per-player FotMob match statistics, then aggregating those scores to the team level for classification.

**Course:** EECE5644 — Introduction to Machine Learning & Pattern Recognition  
**Team:** Amine Kebichi · Nathaniel Maw  
**League:** Premier League — all 20 clubs, 4 seasons (2021/22 – 2024/25)  

---

## Pipeline Overview

```
FotMob per-match player stats (Premier League, 4 seasons)
                    │
                    ▼
        01_fotmob_scraper          ← two-pass HTML + API scraper, rate-limited
                    │
                    ▼
        02_preprocessing           ← cleaning, context features, null handling
                    │
                    ▼
     03_feature_engineering        ← ratio features, 5-match rolling windows,
                    │                 composite scores (ATT / MID / DEF / GK)
                    ▼
              04_eda                ← distributions, correlation matrix, PCA
                    │
                    ▼
          05_regression             ← predict composite scores (Ridge + RF)
                    │
                    ▼
      07_team_aggregation           ← aggregate player predictions to match level
                    │
             ┌──────┴──────┐
             ▼             ▼
     06_clustering   08_classification
     (archetypes)    (H / D / A outcome)
```

| Notebook | Stage | Status |
|----------|-------|--------|
| `01_fotmob_scraper` | FotMob data collection | ✅ Complete |
| `02_preprocessing` | Cleaning & context features | ✅ Complete |
| `03_feature_engineering` | Rolling windows & composite scores | ✅ Complete |
| `04_eda` | Exploratory data analysis | ✅ Complete |
| `05_regression` | Player performance regression | ✅ Complete |
| `06_clustering` | Player archetype discovery | ✅ Complete |
| `07_team_aggregation` | Match-level feature construction | ✅ Complete |
| `08_classification` | Match outcome prediction | ✅ Complete |

---

## Key Results

### Regression — predicting composite scores

| Position | RMSE | R² |
|----------|------|----|
| Defender | 0.0923 | **0.966** |
| Midfielder | 0.1163 | **0.941** |
| Winger | 0.1372 | **0.942** |
| Forward | 0.1409 | **0.938** |
| **Overall (mean)** | — | **0.947** |

Best model: Random Forest, position-grouped, enhanced features (2024/25 test season).

### Classification — predicting match outcomes

| Model | Features | Accuracy | Macro F1 |
|-------|----------|----------|----------|
| Baseline (majority class) | — | 0.336 | 0.207 |
| Logistic Regression | raw | 0.456 | 0.371 |
| Logistic Regression | diff | 0.444 | 0.362 |
| Random Forest | diff | 0.532 | 0.519 |
| **Random Forest** | **raw** | **0.548** | **0.542** |

Best model: Random Forest with raw home/away features (+0.335 macro F1 over baseline).

---

## Dataset

**Source:** [FotMob](https://www.fotmob.com) — per-match player statistics for all Premier League clubs.

FBref was evaluated as an initial source but superseded due to incomplete per-player passing
and physical coverage at the individual match level.

| Property | Value |
|----------|-------|
| Seasons | 2021/22, 2022/23, 2023/24 (train) · 2024/25 (test) |
| Total matches scraped | 1,089 / 1,520 (~72% coverage) |
| Total player-match obs. | 21,242 outfield + 2,138 GK (after quality filter) |
| Post-feature-engineering train | 15,288 outfield + 1,549 GK rows |
| Post-feature-engineering test | 4,748 outfield + 483 GK rows |
| Match-level train / test | 794 / 250 matches |

~28% of matches are served via FotMob's authenticated API only and are unavailable without
login credentials.

### Train / Test Split

Strictly **temporal** — no random shuffling is ever used.

| Partition | Seasons | Matches |
|-----------|---------|---------|
| Training | 2021/22, 2022/23, 2023/24 | 794 |
| Testing | 2024/25 | 250 |

---

## Position-Aware Composite Performance Scores

Four position-specific scores constructed from 5-match rolling means (`.shift(1).rolling(5)`),
z-score normalised on training data only.

| Position | Top weighted metrics |
|----------|---------------------|
| **Attacker / Winger** | Goals+Assists (0.25), xG (0.20), xA (0.15), Dribbles (0.15) |
| **Midfielder** | Progressive Passes (0.20), Chances Created (0.20), xA (0.15) |
| **Defender** | Tackles Won (0.25), Aerial Duels Won (0.20), Clearances (0.20) |
| **Goalkeeper** | Saves (0.30), xG on Target Faced (0.25), Diving Saves (0.15) |

Weight scheme validated against FotMob's independent `rating_title` (Pearson r = 0.191,
best among three candidate schemes).

---

## Repository Structure

```
GOALS/
│
├── 01_fotmob_scraper.ipynb          # Two-pass FotMob scraper (HTML + API fallback)
├── 02_preprocessing.ipynb           # Cleaning, context features, null handling
├── 03_feature_engineering.ipynb     # Ratios, rolling windows, composite scores
├── 04_eda.ipynb                     # Distributions, correlations, PCA
├── 05_regression.ipynb              # Ridge + RF regression for composite scores
├── 06_clustering.ipynb              # K-Means player archetype discovery
├── 07_team_aggregation.ipynb        # Aggregate player scores to match level
├── 08_classification.ipynb          # Logistic Regression + RF outcome prediction
│
├── reports/                         # Final report (PDF + LaTeX source)
│
├── tester_ipynb_files/              # Experimental / scratch notebooks
│
├── old_files/                       # Superseded versions
│
└── data/                            # Git-ignored — local only
    ├── 47/                          # FotMob raw data (league ID 47 = Premier League)
    │   └── {season}/
    │       ├── raw/                 # Cached raw JSON per match
    │       └── output/
    │           ├── outfield_players.parquet
    │           ├── goalkeepers.parquet
    │           └── fixtures.parquet
    │
    ├── processed/
    │   ├── outfield_clean.parquet        # 21,242 × 53
    │   ├── gk_clean.parquet              # 2,138 × 53
    │   ├── datasets/
    │   │   ├── outfield_train_scaled.parquet   # 15,288 × 87
    │   │   ├── outfield_test_scaled.parquet    # 4,748 × 87
    │   │   ├── gk_train_scaled.parquet         # 1,549 × 89
    │   │   └── gk_test_scaled.parquet          # 483 × 89
    │   ├── match_features_train.parquet   # 794 × 37
    │   └── match_features_test.parquet    # 250 × 37
    │
    ├── models/
    │   ├── regression/              # Saved RF + Ridge regression models
    │   ├── clustering/              # Saved KMeans + PCA models per position
    │   ├── classification/          # Saved classifier models
    │   ├── scalers_outfield.pkl     # 4 RobustScalers (one per position group)
    │   ├── scaler_gk.pkl
    │   ├── position_map.pkl
    │   ├── ratio_fill_means.pkl
    │   └── imputation_medians.pkl
    │
    └── visualizations/              # Saved figures (PNG) from all notebooks
```

> **Note:** The `data/` directory is git-ignored. All data files are stored locally only.

---

## Algorithms

| Algorithm | Stage | Key config |
|-----------|-------|------------|
| Ridge Regression | Regression | α = 100; `TimeSeriesSplit` CV (n=5) |
| Random Forest Regressor | Regression | Position-grouped; exhaustive grid search |
| K-Means | Clustering | k=3 outfield, k=2 GK; PCA-reduced (10 / 8 components) |
| Logistic Regression | Classification | `class_weight='balanced'`; C tuned via grid search |
| Random Forest Classifier | Classification | `class_weight='balanced'`; macro F1 primary metric |

### Baselines

| Task | Baseline | Score |
|------|----------|-------|
| Regression | Mean composite per position | R² = 0.018 |
| Classification | Majority class (Home Win) | Macro F1 = 0.207 |
| Clustering | Random assignment | Silhouette ≈ 0.00 |

---

## Key Design Decisions

| Rule | Reason |
|------|--------|
| Temporal train/test split only | Shuffling leaks future match results into training |
| Fit all scalers/imputers on train only | Prevents test-set contamination |
| `class_weight='balanced'` for classifiers | PL has ~45% home wins, ~25% draws |
| `.shift(1).rolling(5)` per player | Current match is never in its own feature window |
| FotMob rating as validation only | Independent benchmark — never used as a training target |
| k=3 for outfield clustering | Domain interpretability prioritised over silhouette-optimal k=2 |

---

## Installation

```bash
git clone https://github.com/natmaw46new/GOALS-Game_Outcome_and_Analytics_Learning_System-.git
cd GOALS-Game_Outcome_and_Analytics_Learning_System-
pip install -r requirements.txt
```

**Core dependencies:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`,
`pyarrow`, `aiohttp`, `requests`, `beautifulsoup4`, `tqdm`, `joblib`

> Data is not included in this repository. Run `01_fotmob_scraper.ipynb` first,
> then execute notebooks 02–08 in order.

---

## Team

| Member | Responsibilities |
|--------|-----------------|
| **Nathaniel Maw** | FotMob scraping, preprocessing, feature engineering, regression, team aggregation, report writing |
| **Amine Kebichi** | Clustering, classification, visualisation, report writing |
| **Both** | EDA, evaluation framework, final report |

---

## References

1. T. Decroos and J. Van Haaren, *Soccerdata: A Python package for scraping soccer data*, 2023.
2. FotMob, *Per-match player statistics and ratings*, fotmob.com, 2024.
3. M. J. Dixon and S. G. Coles, "Modelling association football scores," *J. R. Stat. Soc. C*, 1997.
4. A. C. Constantinou et al., "pi-football," *Knowledge-Based Systems*, 2012.
5. L. Pappalardo et al., "PlayeRank," *ACM TIST*, 2019.
6. Scikit-learn developers, *scikit-learn: Machine Learning in Python*, JMLR 12, 2011.