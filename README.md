# GOALS — Game Outcome and Analytics Learning System

> *"Given recent player performance trends, what can we expect to happen in the next match?"*

**GOALS** is a multi-stage machine learning pipeline that analyses historical La Liga player statistics to deliver interpretable forecasts of future match outcomes. Rather than recapping what has already happened on the park, this system steps forward and asks the question every manager, pundit, and supporter truly wants answered — then answers it with data.

**Course:** EECE5644 — Introduction to Machine Learning & Pattern Recognition
**Team:** Amine Kebichi · Nathaniel Maw
**League:** La Liga — all 20 clubs, 4 seasons (2021/22 – 2024/25)

---

## The Pipeline

Three complementary ML tasks working in concert, like a well-drilled midfield unit:

```
FBref + FotMob Data
        │
        ▼
 Feature Engineering
        │
        ▼
 Position-Aware          ──►  Regression
 Composite Score                  │
        │                         ▼
        │               K-Means Clustering
        │                         │
        ▼                         ▼
 Team Feature  ◄──────────────────┘
  Aggregation
        │
        ▼
 Match Outcome
 Classification
   (W / D / L)
```

| Stage | Task | Goal |
|-------|------|------|
| **Regression** | Predict future player composite scores | Quantify individual form heading into the next fixture |
| **Clustering** | Discover data-driven player archetypes | Map statistical playstyles to recognisable tactical roles |
| **Classification** | Predict match outcomes (Win / Draw / Loss) | Forecast results from aggregated team performance features |

---

## Research Questions

- How strongly does recent individual player form influence team-level match outcomes?
- Which statistical metrics — xG, progressive passes, defensive actions — are the most reliable indicators of what a player will do next?
- Do data-driven player archetypes correspond to recognisable tactical roles on the pitch?
- How effectively can aggregated player-level predictions be combined to forecast the final result?

---

## Dataset

### Sources

| Source | Role | Access |
|--------|------|--------|
| **FBref** | Primary — season-level player stats: xG, progressive passes, pressures, shooting, misc | Scraped via `GOALS_notebook.ipynb` |
| **FotMob** | Supplementary — per-match player stats: dribbles, tackles won, diving saves, chances created, player ratings | Scraped via `fotmob_final.ipynb` |
| **StatsBomb Open Data** | Validation + contextual match info | External reference |

FotMob's proprietary per-match player rating acts as an **independent cross-validation signal** against the composite performance scores constructed in this project.

### Scope and Scale

| Property | Value |
|----------|-------|
| League | La Liga (Spain's top division) |
| Clubs | All 20 La Liga sides across seasons in scope |
| Seasons | 4 (2021/22 – 2024/25) |
| Matches | ~1,520 total (~380 per season) |
| Player-match observations | ~20,000–30,000 |
| Features per observation | ~40 statistical metrics |
| Labels | Match outcome: Win / Draw / Loss |

### Feature Categories

| Category | Example Metrics |
|----------|----------------|
| **Attacking** | Goals, Shots on Target, xG, Dribbles Completed |
| **Playmaking** | Assists, Chances Created, xA, Progressive Passes |
| **Passing** | Pass Completion Rate, Through Balls, Crosses Completed |
| **Defensive** | Tackles Won, Interceptions, Clearances, Aerial Duels Won, Blocks |
| **Goalkeeping** | Saves, Diving Saves, Saves Inside Box, High Claims, Acted as Sweeper |
| **Physical** | Recoveries, Touches, Dispossessed, Distance Covered |
| **Contextual** | Opponent Strength, Home/Away Indicator, Match Importance |

### Train / Test Split

The split is strictly **temporal** — random shuffling is explicitly avoided, as it would allow future information to leak into training and produce inflated, misleading results. Football data flows in one direction.

| Partition | Seasons | Approx. Matches | Approx. Player-Match Obs. |
|-----------|---------|-----------------|--------------------------|
| **Training** | 2021/22, 2022/23, 2023/24 | ~1,140 | ~15,000–22,500 |
| **Testing** | 2024/25 | ~380 | ~5,000–7,500 |
| **Total** | 4 seasons | ~1,520 | ~20,000–30,000 |

Within the three training seasons, hyperparameter tuning uses **time-series-aware cross-validation** — the data is walked forward chronologically, with earlier matchdays training and later matchdays validating in each fold.

---

## Position-Aware Composite Performance Scores

A pivotal design decision: rather than applying a single universal formula to every player on the pitch (which would be like judging a goalkeeper on the same criteria as a striker), the system constructs **four position-specific scores** that faithfully reflect the distinct tactical contribution expected of each role.

All input metrics are first **z-score normalised** across the training set before weights are applied.

### Attacker Score

Attacker performance lives and dies by goal contributions and chance creation.

```
Score_ATT = 0.25·(G+A) + 0.20·xG + 0.15·xA + 0.15·Dribbles
          + 0.10·Shots + 0.10·ChancesCreated + 0.05·Recoveries
```

| Metric | Source | Weight |
|--------|--------|--------|
| Goals + Assists (per 90) | FBref / FotMob | 0.25 |
| Expected Goals (xG) | FBref / FotMob | 0.20 |
| Expected Assists (xA) | FBref / FotMob | 0.15 |
| Successful Dribbles | FotMob | 0.15 |
| Total Shots | FotMob | 0.10 |
| Chances Created | FotMob | 0.10 |
| Ball Recoveries | FotMob | 0.05 |

### Midfielder Score

The modern midfielder must do everything — link defence to attack, win the ball back, and still arrive late into the box.

```
Score_MID = 0.20·ProgPass + 0.20·ChancesCreated + 0.15·xA + 0.15·(G+A)
          + 0.15·TacklesWon + 0.10·Interceptions + 0.05·Recoveries
```

| Metric | Source | Weight |
|--------|--------|--------|
| Progressive Passes | FBref | 0.20 |
| Chances Created | FotMob | 0.20 |
| Expected Assists (xA) | FBref / FotMob | 0.15 |
| Goals + Assists (per 90) | FBref / FotMob | 0.15 |
| Tackles Won | FotMob | 0.15 |
| Interceptions | FBref / FotMob | 0.10 |
| Ball Recoveries | FotMob | 0.05 |

### Defender Score

A defender's first duty is to defend — and this score holds them to exactly that standard.

```
Score_DEF = 0.25·TacklesWon + 0.20·AerialDuelsWon + 0.20·Clearances
          + 0.15·Interceptions + 0.10·Blocks + 0.10·ProgPass
```

| Metric | Source | Weight |
|--------|--------|--------|
| Tackles Won | FotMob | 0.25 |
| Aerial Duels Won | FotMob | 0.20 |
| Clearances | FBref / FotMob | 0.20 |
| Interceptions | FBref / FotMob | 0.15 |
| Blocks | FotMob | 0.10 |
| Progressive Passes | FBref | 0.10 |

### Goalkeeper Score

Built exclusively from shot-stopping and sweeping metrics — a clean sheet starts here.

```
Score_GK = 0.30·Saves + 0.25·xGOTFaced + 0.15·DivingSaves
         + 0.15·SavesInsideBox + 0.10·HighClaims + 0.05·SweeperActions
```

| Metric | Source | Weight |
|--------|--------|--------|
| Saves | FotMob | 0.30 |
| xGoals on Target Faced (xGOT) | FotMob | 0.25 |
| Diving Saves | FotMob | 0.15 |
| Saves Inside Box | FotMob | 0.15 |
| High Claims | FotMob | 0.10 |
| Acted as Sweeper | FotMob | 0.05 |

Position labels are sourced from FotMob's per-match lineup data. If a player shifts position between fixtures, the scoring scheme shifts right along with them.

---

## Algorithms and Evaluation

### Algorithms

| Algorithm | Stage | Purpose |
|-----------|-------|---------|
| **Ridge Regression** | Regression | Predict composite scores; L2 regularisation manages multicollinearity between football statistics |
| **Random Forest** | Regression + Classification | Captures nonlinear relationships and feature interactions |
| **K-Means** | Clustering | Groups players into statistical archetypes |

### Baseline Models

| Task | Baseline |
|------|----------|
| Regression | Mean composite score predictor |
| Classification | Majority-class predictor; last-match result predictor |
| Clustering | Random cluster assignment |

### Evaluation Metrics

| Task | Metrics |
|------|---------|
| **Regression** | MSE, RMSE, R² |
| **Classification** | Accuracy, Precision, Recall, F1 Score (macro), Confusion Matrix |
| **Clustering** | Silhouette Score, Elbow Method (inertia vs. k), PCA Visualisation |

Class-weighted loss functions are used throughout classification — La Liga's home win bias (~45%) would otherwise cause a naive model to ignore draws and away wins entirely.

---

## Player-to-Team Aggregation

Individual player predictions are aggregated into team-level features before classification. For each match:

- Mean predicted performance of the starting XI
- Minutes-weighted performance averages
- Offensive vs. defensive contribution totals across the lineup
- Distribution of player archetypes within the selected squad

These aggregated features serve as the primary inputs to the match outcome classifier.

---

## Repository Structure

```
GOALS/
├── CLAUDE.md                        # Persistent AI session context
├── fotmob_final.ipynb               # FotMob scraper (LEAGUE_ID=87 for La Liga)
├── GOALS_notebook.ipynb             # FBref scraper (data already collected)
├── data/
│   ├── FBref/
│   │   └── la_liga/{season}/        # standard, shooting, misc, goalkeeping, playing_time CSVs
│   └── 87/{season}/                 # FotMob La Liga output
│       ├── raw/                     # Cached match JSON (one file per match_id)
│       └── output/
│           ├── outfield_players.parquet
│           ├── goalkeepers.parquet
│           ├── fixtures.parquet
│           └── player_stats.parquet
└── notebooks/
    ├── 01_data_merge.ipynb          # FBref + FotMob join (fuzzy player name matching)
    ├── 02_eda.ipynb                 # Distributions, correlations, PCA
    ├── 03_feature_engineering.ipynb # Z-score normalisation + composite score construction
    ├── 04_regression.ipynb          # Ridge + Random Forest; time-series CV
    ├── 05_clustering.ipynb          # K-Means archetypes; Silhouette + Elbow
    └── 06_classification.ipynb     # Win/Draw/Loss prediction; class-weighted
```

**FBref season folder names:** `2021-2022`, `2022-2023`, `2023-2024`, `2024-2025`
**FotMob season folder names:** `2021_2022`, `2022_2023`, `2023_2024`, `2024_2025`

---

## Data Collection

### FBref (complete)

FBref data for La Liga (plus Premier League and Bundesliga) across all 4 seasons has already been scraped and is stored under `data/FBref/la_liga/`.

### FotMob La Liga (run required)

`fotmob_final.ipynb` is production-ready with rate-limiting, HMAC auth, retry logic, and idempotent JSON caching. Run it **4 times** — once per season — with Cell 1 configured as follows:

```python
LEAGUE_ID = 87          # La Liga
SEASON    = '2021/2022' # then 2022/2023, 2023/2024, 2024/2025
```

Each run is fully resumable from cached JSON if interrupted.

---

## Expected Challenges

| Challenge | Mitigation |
|-----------|------------|
| **No ground-truth performance score** | Four position-specific composite scores constructed from domain-motivated weights; sensitivity analysis across three weighting schemes validates robustness |
| **Multicollinearity** (xG and goals correlate heavily) | Ridge regularisation (L2); correlation matrix analysis to flag redundant features |
| **Class imbalance** (~45% home wins, ~25% draws) | `class_weight='balanced'` throughout; macro F1 evaluation |
| **Sparse minutes for fringe players** | Minimum appearance threshold filter; 20-club × 4-season volume absorbs filtering without significant data loss |
| **FBref / FotMob name mismatches** | Fuzzy string matching with date-keyed match anchors; unmatched records retained from available source rather than discarded |
| **Promotion and relegation** | Clubs included only for seasons in which they were a La Liga side; insufficient coverage → excluded from analysis |

---

## Timeline

| Milestone | Description | Target |
|-----------|-------------|--------|
| 1 | Data collection and preprocessing (FBref + FotMob) | Feb 21 – Feb 28 |
| 2 | Exploratory data analysis and feature engineering | Feb 28 – Mar 14 |
| 3 | Regression and clustering models | Mar 14 – Mar 28 |
| 4 | Classification model and pipeline integration | Mar 21 – Apr 4 |
| 5 | Evaluation, refinement, and forward fixture forecasting | Apr 4 – Apr 11 |
| **6** | **Final report and presentation** | **Apr 11 – Apr 18** |

---

## Team

| Member | Responsibilities |
|--------|-----------------|
| **Amine Kebichi** | Regression modelling, evaluation framework, report writing |
| **Nathaniel Maw** | Clustering analysis, classification models, visualisation |
| **Both** | Data preprocessing, EDA, evaluation, presentation |

---

## References

1. T. Decroos and J. Van Haaren, *Soccerdata: A Python package for scraping soccer data*, 2023.
2. Sports Reference LLC, *FBref advanced football statistics*, fbref.com, 2024.
3. StatsBomb, *StatsBomb Open Data*, github.com/statsbomb/open-data, 2023.
