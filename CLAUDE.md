# GOALS — Game Outcome and Analytics Learning System

**Course:** EECE5644 — Introduction to Machine Learning and Pattern Recognition
**Team:** Amine Kebichi (regression + evaluation), Nathaniel Maw (clustering + classification)
**Deadline:** Final report + presentation — **April 18, 2026**
**Today (session initialized):** March 20, 2026

---

## Project Overview

GOALS is a multi-stage ML pipeline that predicts Premier League match outcomes (Win/Draw/Loss) by constructing position-specific composite performance scores from per-player match statistics, then aggregating those scores to the team level.

**Data sources:**
- **FBref** — season-level player stats (standard, shooting, misc, goalkeeping, playing_time) — already scraped for all 4 seasons
- **FotMob** — per-match player stats with richer granularity (xG, xA, progressive passes, dribbles, etc.) — scraped for Premier League 2024/25 only; **seasons 2021/22–2023/24 still need to be scraped**

---

## Repository Structure

```
GOALS/
├── CLAUDE.md                       # This file — persistent session context
├── fotmob_final.ipynb              # FotMob scraper — LEAGUE_ID=47 for Premier League
├── GOALS_notebook.ipynb            # FBref scraper (data already collected)
├── (OLD)fotmob.ipynb               # Deprecated — ignore
├── data/
│   ├── FBref/
│   │   ├── premier_league/{season}/  # ✅ 4 seasons scraped
│   │   │   ├── standard.csv
│   │   │   ├── shooting.csv
│   │   │   ├── misc.csv
│   │   │   ├── goalkeeping.csv
│   │   │   └── playing_time.csv
│   │   ├── la_liga/{season}/
│   │   └── bundesliga/{season}/
│   ├── 47/2024_2025/               # FotMob Premier League 2024/25 ✅ already scraped
│   │   ├── raw/                    # Raw JSON per match_id
│   │   └── output/                 # outfield_players.parquet, goalkeepers.parquet, fixtures.parquet
│   └── 47/{season}/                # FotMob Premier League 2021/22–2023/24 — TO BE SCRAPED
│       ├── raw/
│       └── output/
└── notebooks/                      # ML pipeline notebooks — TO BE CREATED
    ├── 01_data_merge.ipynb
    ├── 02_eda.ipynb
    ├── 03_feature_engineering.ipynb
    ├── 04_regression.ipynb
    ├── 05_clustering.ipynb
    └── 06_classification.ipynb
```

**FBref season folder names:** `2021-2022`, `2022-2023`, `2023-2024`, `2024-2025`
**FotMob season folder names:** `2021_2022`, `2022_2023`, `2023_2024`, `2024_2025` (slashes replaced with underscores)

---

## Phase 1 (Immediate): FotMob Premier League Data Collection

**Status: PARTIALLY DONE — 2024/25 complete, need 2021/22, 2022/23, 2023/24**

`fotmob_final.ipynb` is production-ready. To scrape the remaining 3 seasons, run it **3 times** — once per season — after changing Cell 1 config:

```python
LEAGUE_ID = 47          # Premier League (already correct)
SEASON    = '2021/2022' # then '2022/2023', '2023/2024'  (2024/2025 already done)
```

Each run produces under `data/47/{season_underscored}/output/`:
- `outfield_players.parquet` — wide-format per-player-per-match (main input)
- `goalkeepers.parquet` — same for GKs
- `fixtures.parquet` — match metadata with results
- `player_stats.parquet` — raw long-format archive

Do not change any other cells — the scraper is idempotent and caches raw JSON.

---

## ML Pipeline Specification

### Data Split (temporal — never random shuffle)

| Split | Seasons | Purpose |
|-------|---------|---------|
| Train | 2021/22, 2022/23, 2023/24 | Model fitting + cross-validation |
| Test  | 2024/25 | Final held-out evaluation |

**CV strategy:** Walk-forward chronological within training seasons (no future leakage).

---

### Notebook Pipeline

| Notebook | Task |
|----------|------|
| `01_data_merge.ipynb` | Join FBref + FotMob on (player, match_date, team) using fuzzy name matching |
| `02_eda.ipynb` | Feature distributions, correlation matrix, PCA projection |
| `03_feature_engineering.ipynb` | Z-score normalization, composite score construction, team aggregation |
| `04_regression.ipynb` | Predict composite scores — Ridge + Random Forest; time-series CV |
| `05_clustering.ipynb` | K-Means player archetypes; Silhouette + Elbow evaluation |
| `06_classification.ipynb` | Predict Win/Draw/Loss from aggregated team features; class-weighted |

---

### Data Merge Strategy (`01_data_merge.ipynb`)

**Join key:** `(player_name fuzzy, match_date, home_team/away_team)`

- Use `rapidfuzz` or `thefuzz` for player name fuzzy matching (FBref vs FotMob names often differ)
- Match on date ± 0 days; team name matching can use exact or fuzzy
- Output: one row per player per match, with both FBref season stats and FotMob match stats

---

### Composite Score Formulas (`03_feature_engineering.ipynb`)

All input metrics must be **z-score normalized** (per metric, across the training set) before applying weights.

**ATT (Attacker) Score:**
```
ATT = 0.25*(Goals + Assists) + 0.20*xG + 0.15*xA + 0.15*Dribbles
    + 0.10*Shots + 0.10*ChancesCreated + 0.05*Recoveries
```

**MID (Midfielder) Score:**
```
MID = 0.20*ProgPass + 0.20*ChancesCreated + 0.15*xA + 0.15*(Goals + Assists)
    + 0.15*TacklesWon + 0.10*Interceptions + 0.05*Recoveries
```

**DEF (Defender) Score:**
```
DEF = 0.25*TacklesWon + 0.20*AerialDuelsWon + 0.20*Clearances
    + 0.15*Interceptions + 0.10*Blocks + 0.10*ProgPass
```

**GK (Goalkeeper) Score:**
```
GK = 0.30*Saves + 0.25*xGOT + 0.15*DivingSaves + 0.15*SavesInsideBox
   + 0.10*HighClaims + 0.05*SweeperActions
```

**FotMob column name mappings:**
- `Goals + Assists` → `goals` + `goal_assist` (outfield_players)
- `xG` → `expected_goals`
- `xA` → `expected_assists`
- `Dribbles` → `successful_dribbles`
- `ChancesCreated` → `chances_created`
- `Recoveries` → `recoveries`
- `ProgPass` → `accurate_passes` (use FBref `progressive_passes` if available)
- `TacklesWon` → `tackles_won`
- `Interceptions` → `interceptions`
- `AerialDuelsWon` → `aerial_duels_won`
- `Clearances` → `clearances`
- `Blocks` → `shot_blocks`
- `Saves` → `saves`
- `xGOT` → `xgot_faced`
- `DivingSaves` → `diving_save`
- `SavesInsideBox` → `saves_inside_box`
- `HighClaims` → `high_claim`
- `SweeperActions` → `acted_as_sweeper`

**Team feature aggregation:** For each match, sum composite scores of starting XI per position group → 4 features per team → 8 features total for classification (home vs away).

---

### Regression (`04_regression.ipynb`) — Amine

- **Target:** composite score values (continuous)
- **Models:** Ridge Regression, Random Forest Regressor
- **Hyperparameter tuning:** GridSearchCV / RandomizedSearchCV with time-series CV
- **Evaluation:** RMSE, MAE, R²
- **Baseline:** predict mean composite score per position

---

### Clustering (`05_clustering.ipynb`) — Nathaniel

- **Target:** player archetype discovery (unsupervised)
- **Algorithm:** K-Means
- **Evaluation:** Silhouette Score, Elbow method (inertia vs k)
- **Input features:** normalized composite scores + position indicator
- **Goal:** identify player clusters (e.g., "clinical striker", "defensive midfielder")

---

### Classification (`06_classification.ipynb`) — Nathaniel

- **Target:** match outcome (Win/Draw/Loss) — 3-class
- **Input:** aggregated team composite scores per match
- **Models:** Logistic Regression, Random Forest Classifier, optionally SVM
- **Class imbalance:** always use `class_weight='balanced'`
- **Evaluation:** accuracy, macro F1, confusion matrix
- **Baseline:** predict most frequent class ("Home Win")

---

## Key Constraints and Pitfalls

1. **Never use random train/test splits** — always temporal splits. Shuffling creates data leakage (future matches inform past predictions).
2. **Z-score normalization fit on train only** — apply the same scaler to test; never fit on test data.
3. **Multicollinearity** — many FotMob metrics are correlated (e.g., goals and xG). Use Ridge regression (L2) to handle this; check VIF if needed.
4. **Class imbalance** — The Premier League has ~45% home wins, ~25% draws, ~30% away wins. Always use `class_weight='balanced'` for classifiers.
5. **Data alignment** — FBref is season-level (one row per player per season); FotMob is match-level. Merge carefully: FBref stats serve as contextual features, FotMob stats drive the per-match composite score.
6. **Player name mismatches** — FBref uses accented names, FotMob may differ. Use fuzzy matching with threshold ≥ 85.
7. **FotMob rate limiting** — scraper already handles this with jitter + retries. Do not increase MAX_CONCURRENT beyond 4.

---

## Data Status

| Source | League | Seasons | Status |
|--------|--------|---------|--------|
| FBref | Premier League | 2021-2025 (4 seasons) | ✅ Complete |
| FBref | La Liga | 2021-2025 | ✅ Complete (not used) |
| FBref | Bundesliga | 2021-2025 | ✅ Complete (not used) |
| FotMob | Premier League (47) | 2024/25 | ✅ Complete |
| FotMob | Premier League (47) | 2021/22, 2022/23, 2023/24 | ❌ Not yet scraped |

---

## Team Responsibilities

| Area | Owner |
|------|-------|
| FotMob Premier League scrape (remaining 3 seasons) | Amine |
| Data merge (01) + EDA (02) | Both |
| Feature engineering (03) | Both |
| Regression (04) + evaluation metrics | Amine |
| Clustering (05) | Nathaniel |
| Classification (06) | Nathaniel |
| Final report write-up | Both |

---

## Timeline

| Date | Milestone |
|------|-----------|
| March 20, 2026 | Milestone 3 begins — regression + clustering |
| ~March 28, 2026 | FotMob scrape complete + data merge done |
| ~April 5, 2026 | Regression + clustering results ready |
| ~April 12, 2026 | Classification complete, full pipeline validated |
| **April 18, 2026** | **Final report + presentation due** |
