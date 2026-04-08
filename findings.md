# GOALS — Pipeline Findings

**Course:** EECE5644 — Introduction to Machine Learning and Pattern Recognition
**Team:** Amine Kebichi, Nathaniel Maw
**Data source:** FotMob Premier League (Seasons 2021/22 – 2024/25)
**Pipeline run date:** April 8, 2026

---

## 1. Data Collection

### FotMob Scraping
- **Scope:** Premier League (league ID 47), 4 seasons: 2021/22, 2022/23, 2023/24, 2024/25
- **Matches scraped:** 1,089 of 1,520 total (~72% coverage)
- **Missing ~28%:** Served via FotMob authenticated API only — not accessible without login credentials
- **Scraper:** idempotent (caches raw JSON), rate-limited with jitter + retries

### Raw data per season (post-scrape, pre-preprocessing)

| Season | Outfield rows | GK rows | Fixtures |
|--------|--------------|---------|----------|
| 2021/22 | ~5,600 | ~560 | 380 |
| 2022/23 | ~5,200 | ~520 | 380 |
| 2023/24 | ~5,300 | ~530 | 380 |
| 2024/25 | ~5,200 | ~520 | 380 |
| **Total** | **21,430** | **2,138** | **1,520** |

Raw outfield schema: 68 columns × 21,430 rows. Raw GK schema: 53 columns × 2,138 rows.

---

## 2. Preprocessing (`02_preprocessing.ipynb`)

### Team name normalisation
Two mismatches found between `team_name` (player data) and `home_team`/`away_team` (fixture data):

| Raw name | Corrected to |
|----------|-------------|
| `Brighton and Hove Albion` | `Brighton & Hove Albion` |
| `Bournemouth` | `AFC Bournemouth` |

After correction: **zero mismatches** across all 4 seasons for both outfield and GK.

### Quality filtering
| Filter | Outfield before | After | Dropped |
|--------|----------------|-------|---------|
| `minutes_played >= 30` | 21,430 | 21,242 | 188 (0.9%) |
| `rating_title` not null | 21,242 | 21,242 | 0 (0.0%) |
| GK (both filters) | 2,138 | 2,138 | 0 (0.0%) |

All GK rows had ≥30 minutes (goalkeepers nearly always play the full match).

### Column drops
20 outfield columns and 5 GK columns removed. Drop criteria:

| Category | Columns |
|----------|---------|
| >97% null — rare events | `missed_penalty`, `owngoal`, `penalties_won`, `conceded_penalties`, `clearance_off_the_line`, `errors_led_to_goal`, `last_man_tackle`, `shots_woodwork` |
| >80% null — too sparse | `Offsides`, `big_chance_missed_title`, `big_chance_created_team_title`, `corners` |
| >70% null — not in any composite formula | `blocked_shots`, `expected_goals_on_target_variant` |
| 40–50% null — not needed | `headed_clearance`, `was_fouled`, `expected_goals_non_penalty`, `xg_and_xa`, `passes_into_final_third`, `touches_opp_box` |

### Match context features added
Five features derived per player-match and appended to both outfield and GK:

| Feature | Description | Method |
|---------|-------------|--------|
| `home_away` | 1 = home, 0 = away | `team_name == home_team` |
| `opponent` | Opposing team name | `np.where` on `home_away` |
| `team_goals_scored` | Goals by player's team | Sum `goals` per match+team from outfield |
| `opp_goals_scored` | Goals by opponent | Joined from same lookup |
| `result` | W / D / L | Comparison of goals columns |

Note: `team_goals_scored` is derived from summing the outfield `goals` column per (match, team). GK data has no `goals` column, so this lookup is joined from the outfield aggregate.

### Result distribution (match-team level, all seasons)

| Result | Overall | 2021/22 | 2022/23 | 2023/24 | 2024/25 |
|--------|---------|---------|---------|---------|---------|
| Win | 35.0% | 35.6% | 34.4% | 35.8% | 34.4% |
| Loss | 35.0% | 35.6% | 34.4% | 35.8% | 34.4% |
| Draw | 29.9% | 28.8% | 31.3% | 28.5% | 31.3% |

Win and loss percentages are equal by construction (every win has a corresponding loss). Draw rate is consistent at ~29–31% across seasons.

### Post-preprocessing dimensions

| Dataset | Rows | Columns |
|---------|------|---------|
| `outfield_clean.parquet` | 21,242 | 53 |
| `gk_clean.parquet` | 2,138 | 53 |

### Remaining null rates (outfield, post-drop)

| Column | Null % | Notes |
|--------|--------|-------|
| `shot_accuracy_total` | 52.0% | Kept — required for shot_accuracy ratio |
| `shot_accuracy` | 52.0% | Kept — composite input |
| `accurate_crosses_total` | 47.0% | Kept — cross_accuracy denominator |
| `accurate_crosses` | 47.0% | Kept — composite input |
| `dribbles_succeeded_total` | 42.1% | Kept — dribble_success_rate denominator |
| `dribbles_succeeded` | 42.1% | Kept — composite input |
| `ShotsOffTarget` / `ShotsOnTarget` | 41.9% | Kept — shot_accuracy numerator |
| `expected_goals` | 41.9% | Kept — ATT composite input |
| `long_balls_accurate_total` | 18.8% | Kept — ratio denominator |
| `long_balls_accurate` | 18.8% | Kept — composite input |
| `expected_assists` | 15.1% | Kept — MID/ATT composite input |
| `duel_won` / `duel_lost` | 3.6–4.1% | Kept — rolling feature |
| `ground_duels_won` / `_total` | 2.6% | Kept — ratio denominator |

All remaining nulls are filled with training-set medians in `03_feature_engineering.ipynb` after the temporal split, to avoid data leakage.

---

## 3. Feature Engineering (`03_feature_engineering.ipynb`)

### Temporal train / test split

| Split | Seasons | Outfield rows | GK rows |
|-------|---------|--------------|---------|
| Train | 2021/22, 2022/23, 2023/24 | 16,038 | 1,614 |
| Test | 2024/25 | 5,204 | 524 |

### Position mapping
FotMob numeric `position_id` mapped to 4 outfield groups:

| Group | position_id values | Count (train) |
|-------|--------------------|---------------|
| defender | 32–38, 51–52, 58–59, 62, 71, 79 | 6,543 |
| midfielder | 63–68, 73–77, 95 | 3,693 |
| winger | 72, 78, 82–88, 94, 96, 107 | 3,782 |
| forward | 103–106, 114–116 | 2,020 |

3 position IDs (0, 31, 39) were unmapped and defaulted to `midfielder`. These account for 12 rows total.

### Ratio features

#### Outfield (7 ratios)

| Feature | Numerator | Denominator |
|---------|-----------|-------------|
| `pass_accuracy` | `accurate_passes` | `accurate_passes_total` |
| `long_ball_accuracy` | `long_balls_accurate` | `long_balls_accurate_total` |
| `cross_accuracy` | `accurate_crosses` | `accurate_crosses_total` |
| `dribble_success_rate` | `dribbles_succeeded` | `dribbles_succeeded_total` |
| `aerial_win_rate` | `aerials_won` | `aerials_won_total` |
| `ground_duel_win_rate` | `ground_duels_won` | `ground_duels_won_total` |
| `shot_accuracy` | `ShotsOnTarget` | `shot_accuracy_total` |

#### GK (4 ratios + save_rate)

| Feature | Formula |
|---------|---------|
| `pass_accuracy` | `accurate_passes / accurate_passes_total` |
| `long_ball_accuracy` | `long_balls_accurate / long_balls_accurate_total` |
| `aerial_win_rate` | `aerials_won / aerials_won_total` |
| `ground_duel_win_rate` | `ground_duels_won / ground_duels_won_total` |
| `save_rate` | `saves / (saves + goals_conceded)` |

All 0/0 division NaNs filled with **training-set means** only. Denominator columns dropped after ratio creation.

GK ratio outlier clipping: `roll5_aerial_win_rate`, `roll5_ground_duel_win_rate`, `roll5_save_rate`, `roll5_goals_prevented` clipped to ±5 standard deviations (training distribution) to suppress outliers from sparse denominators.

### Null imputation
After ratio computation, all remaining numeric nulls filled with **training-set medians**. Result: **zero nulls** in both train and test for both outfield and GK.

### 5-match rolling windows
Per player, sorted by `match_date`:
- Applied `.shift(1).rolling(5, min_periods=1).mean()` — current match is never in its own window
- 32 outfield features rolled; 34 GK features rolled
- Rows where rolling is NaN (first appearance, no prior matches) dropped

Post-rolling dimensions:

| Dataset | Rows | Columns |
|---------|------|---------|
| Outfield train | 15,288 | 86 |
| Outfield test | 4,748 | 86 |
| GK train | 1,549 | 88 |
| GK test | 483 | 88 |

### Composite score construction (Scheme A — canonical)

All input metrics z-score normalised per metric using **training-set mean and std**. Weights applied per position group.

#### ATT (Winger / Forward)
```
score = 0.25*(roll5_goals + roll5_assists) + 0.20*roll5_expected_goals
      + 0.15*roll5_expected_assists + 0.15*roll5_dribbles_succeeded
      + 0.10*roll5_chances_created  + 0.05*roll5_recoveries
      + 0.10*roll5_shot_accuracy
```

#### MID (Midfielder)
```
score = 0.20*roll5_accurate_passes + 0.20*roll5_chances_created
      + 0.15*roll5_expected_assists + 0.15*(roll5_goals + roll5_assists)
      + 0.15*roll5_tackles + 0.10*roll5_interceptions + 0.05*roll5_recoveries
```

#### DEF (Defender)
```
score = 0.25*roll5_tackles + 0.20*roll5_aerials_won
      + 0.20*roll5_clearances + 0.15*roll5_interceptions
      + 0.10*roll5_shot_blocks + 0.10*roll5_accurate_passes
```

#### GK (Goalkeeper)
```
score = 0.30*roll5_saves + 0.25*roll5_expected_goals_on_target_faced
      + 0.15*roll5_keeper_diving_save + 0.15*roll5_saves_inside_box
      + 0.10*roll5_keeper_high_claim  + 0.05*roll5_keeper_sweeper
```

### Composite score distribution (training set)

| Position | Count | Mean | Std | Min | Median | Max |
|----------|-------|------|-----|-----|--------|-----|
| Defender | 6,251 | 0.408 | 0.522 | −1.245 | 0.393 | 3.374 |
| Midfielder | 3,527 | 0.233 | 0.486 | −1.190 | 0.177 | 2.569 |
| Winger | 3,597 | 0.339 | 0.629 | −1.224 | 0.258 | 5.520 |
| Forward | 1,913 | 0.311 | 0.643 | −0.971 | 0.199 | 4.060 |
| GK | 1,549 | 0.000 | 0.702 | −2.022 | −0.061 | 3.149 |

### Sensitivity analysis — composite weight schemes

Three weight schemes compared by Pearson correlation and RMSE against FotMob `rating_title` (independent benchmark):

| Scheme | Description | r (all outfield) | RMSE |
|--------|-------------|-------------------|------|
| A | Proposal weights (canonical) | **+0.191** | 6.731 |
| B | Equal weights | +0.179 | 6.747 |
| C | Data-driven (|r| with rating) | +0.199 | 6.743 |

**Decision:** Scheme A (proposal weights) chosen as canonical. Scheme C has marginally higher r (+0.008) but RMSE is higher than Scheme A and it overfits the correlation with FotMob ratings (which themselves are partly subjective). Scheme A reflects domain knowledge about position roles.

Per-position breakdown — Scheme A:

| Position | r vs rating_title | n |
|----------|-------------------|---|
| Midfielder | +0.311 | 3,527 |
| Winger | +0.293 | 3,597 |
| Forward | +0.248 | 1,913 |
| Defender | +0.071 | 6,251 |

Defender correlation is notably weaker — FotMob ratings for defenders likely incorporate defensive actions not fully captured by our formula (e.g., positioning, press resistance).

### Scaling
`RobustScaler` fit per position group on **training data only**, applied to both train and test. Robust scaling chosen over StandardScaler to reduce sensitivity to outlier performances (hat-tricks, clean sheets in dominant wins).

Final scaled dataset dimensions:

| File | Rows | Columns |
|------|------|---------|
| `outfield_train_scaled.parquet` | 15,288 | 87 |
| `outfield_test_scaled.parquet` | 4,748 | 87 |
| `gk_train_scaled.parquet` | 1,549 | 89 |
| `gk_test_scaled.parquet` | 483 | 89 |

---

## 4. Exploratory Data Analysis (`04_eda.ipynb`)

### Feature correlations
High-correlation pairs identified (|r| > 0.7):

| Feature A | Feature B | r |
|-----------|-----------|---|
| `roll5_accurate_passes` | `roll5_touches` | **0.914** |
| `roll5_duel_won` | `roll5_ground_duels_won` | 0.716 |

The passes–touches correlation is expected (players who touch the ball more naturally complete more passes). Ridge regression's L2 penalty handles this multicollinearity. For Random Forest, redundant features dilute individual importances but do not harm predictive performance.

### Top teams by mean composite score (train: 2021/22–2023/24)

| Rank | Team | Mean composite |
|------|------|---------------|
| 1 | Chelsea | 0.592 |
| 2 | Manchester City | 0.572 |
| 3 | Everton | 0.547 |
| 4 | Manchester United | 0.529 |
| 5 | AFC Bournemouth | 0.472 |
| 6 | Liverpool | 0.448 |
| 7 | Arsenal | 0.414 |
| 8 | Newcastle United | 0.410 |
| 9 | Brighton & Hove Albion | 0.361 |
| 10 | Crystal Palace | 0.338 |

Note: Composite scores are scaled and position-normalised. High scores for Everton and Bournemouth may reflect specific match contexts captured by rolling averages (e.g., consistent minutes from rated players) rather than overall team quality.

---

## 5. Regression (`05_regression.ipynb`)

**Target:** `composite_score` (continuous)
**CV strategy:** `TimeSeriesSplit(n_splits=5)` within training seasons
**Feature sets:**
- *Baseline*: 34 rolling features + `home_away` + `minutes_played`
- *Enhanced*: baseline + `roll5_team_goals_scored` + `roll5_opp_goals_scored` (team/opponent strength context)

### Full results table (test set)

| Model | Features | RMSE | MAE | R² | n |
|-------|----------|------|-----|-----|---|
| **RF grouped** | **Enhanced** | **0.1022 avg** | — | **0.947** | 4,292 |
| RF global | Enhanced | 0.2027 | 0.1396 | 0.853 | 4,292 |
| RF global | Baseline | 0.2365 | 0.1538 | 0.829 | 4,748 |
| Ridge | Enhanced | 0.3176 | 0.2470 | 0.638 | 4,292 |
| Ridge | Baseline | 0.3464 | 0.2636 | 0.633 | 4,748 |
| Baseline (pos mean) | — | 0.5671 | 0.4403 | 0.018 | 4,748 |

### Position-grouped RF (best model) — test results

| Position | RMSE | MAE | R² | n (test) |
|----------|------|-----|-----|----------|
| Defender | 0.0923 | 0.0646 | **0.966** | 1,747 |
| Midfielder | 0.1163 | 0.0860 | **0.941** | 977 |
| Winger | 0.1372 | 0.0926 | **0.942** | 1,061 |
| Forward | 0.1409 | 0.1084 | **0.938** | 507 |
| **Overall (avg R²)** | — | — | **0.947** | 4,292 |

Best hyperparameters per position (RF grouped enhanced):

| Position | n_estimators | max_depth | min_samples_leaf | max_features |
|----------|-------------|-----------|-----------------|--------------|
| Defender | 200 | 10 | 8 | 0.5 |
| Midfielder | 100 | 8 | 5 | 0.5 |
| Winger | 200 | 10 | 8 | 0.5 |
| Forward | 200 | 6 | 3 | 0.5 |

### Best Ridge hyperparameter
Both baseline and enhanced Ridge: `alpha = 100`.

### Feature importances — RF enhanced global model (top 10)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `roll5_defensive_actions` | 0.2003 |
| 2 | `roll5_ShotsOnTarget` | 0.1190 |
| 3 | `roll5_touches` | 0.1051 |
| 4 | `roll5_assists` | 0.0932 |
| 5 | `roll5_goals` | 0.0730 |
| 6 | `roll5_expected_assists` | 0.0536 |
| 7 | `roll5_chances_created` | 0.0443 |
| 8 | `roll5_duel_won` | 0.0372 |
| 9 | `roll5_accurate_passes` | 0.0371 |
| 10 | `roll5_expected_goals` | 0.0327 |

`roll5_defensive_actions` is the top feature because it appears across all positions (defenders, midfielders, and GKs all record defensive actions), making it a strong general discriminator in the global model.

### Interpretation
The high R² (0.947 grouped) reflects that composite scores are computed from a deterministic weighted combination of the same rolling features used as inputs — the RF is largely recovering the formula structure. This is expected and desirable: it confirms the pipeline is consistent end-to-end. The regression predictions will be used as player-performance estimates in the team aggregation stage, where the model captures non-linear position-specific interactions that the linear formula cannot.

The enhanced features (`roll5_team_goals_scored`, `roll5_opp_goals_scored`) improved R² by ~2 percentage points globally (0.829 → 0.853), confirming that opponent and team context adds signal beyond individual rolling stats alone.

---

## 6. Saved Artefacts

| File | Location | Description |
|------|----------|-------------|
| `outfield_clean.parquet` | `data/processed/` | 21,242 rows × 53 cols |
| `gk_clean.parquet` | `data/processed/` | 2,138 rows × 53 cols |
| `outfield_train_scaled.parquet` | `data/processed/datasets/` | 15,288 rows × 87 cols |
| `outfield_test_scaled.parquet` | `data/processed/datasets/` | 4,748 rows × 87 cols |
| `gk_train_scaled.parquet` | `data/processed/datasets/` | 1,549 rows × 89 cols |
| `gk_test_scaled.parquet` | `data/processed/datasets/` | 483 rows × 89 cols |
| `scalers_outfield.pkl` | `data/models/` | 4 RobustScalers (one per position group) |
| `scaler_gk.pkl` | `data/models/` | Single RobustScaler for GK |
| `position_map.pkl` | `data/models/` | Dict mapping FotMob position_id → group |
| `ratio_fill_means.pkl` | `data/models/` | Training means for ratio 0/0 fills |
| `imputation_medians.pkl` | `data/models/` | Training medians for null imputation |
| `rf_regression_defender.pkl` | `data/models/` | RF regressor for defender composite score |
| `rf_regression_midfielder.pkl` | `data/models/` | RF regressor for midfielder composite score |
| `rf_regression_winger.pkl` | `data/models/` | RF regressor for winger composite score |
| `rf_regression_forward.pkl` | `data/models/` | RF regressor for forward composite score |

---

## 7. Team Aggregation (`07_team_aggregation.ipynb`)

### Design
Aggregates per-player composite scores to one row per match with home/away features side-by-side.

**Features computed per team per match (prefixed `h_` / `a_`):**

| Feature | Description |
|---------|-------------|
| `mean_composite` | Mean composite score of all outfield players |
| `weighted_composite` | Minutes-weighted mean composite |
| `att_composite` | Mean composite of forwards |
| `att2_composite` | Mean composite of wingers |
| `mid_composite` | Mean composite of midfielders |
| `def_composite` | Mean composite of defenders |
| `top3_composite` | Mean of top 3 composites (key performers) |
| `composite_std` | Std of composites (lineup cohesion) |
| `gk_composite` | GK composite score |
| `n_defenders/midfielders/wingers/forwards` | Formation shape counts |
| `roll5_attack_str` | Rolling 5-match avg goals scored (team form) |
| `roll5_defence_str` | Rolling 5-match avg goals conceded (defensive form) |

**Target:** `outcome` = H (home win), D (draw), A (away win).

Note: `roll5_attack_str` and `roll5_defence_str` are computed at team level with `.shift(1)` to exclude the current match — no leakage.

### Match coverage
- Total unique matches after rolling (player-level drop): **1,059**
- Matches with both home + away data (pivot inner join): **1,044**
- Matches missing GK data (filled with GK composite mean): **71**

### Outcome distribution

| Outcome | Overall | 2021/22 | 2022/23 | 2023/24 | 2024/25 |
|---------|---------|---------|---------|---------|---------|
| Away win (A) | 35.2% | 36.3% | 34.6% | 36.0% | 33.6% |
| Home win (H) | 34.7% | 35.2% | 34.2% | 35.6% | 33.6% |
| Draw (D) | 30.2% | 28.5% | 31.2% | 28.5% | 32.8% |

### Train / test split

| Split | Seasons | Matches | H | D | A |
|-------|---------|---------|---|---|---|
| Train | 2021/22–2023/24 | 794 | 278 | 233 | 283 |
| Test | 2024/25 | 250 | 84 | 82 | 84 |

### Feature nulls (position-specific composites)
Nulls occur when a team has no recorded player in a given position group for that match:

| Feature | Nulls (match-level) |
|---------|---------------------|
| `h_att2_composite` / `a_att2_composite` | 86 each |
| `h_att_composite` / `a_att_composite` | 35–36 each |
| `h_mid_composite` / `a_mid_composite` | 9–12 each |
| `h_def_composite` / `a_def_composite` | 3–6 each |

These will be imputed with training-set means in the classification notebook.

### Composite differential validation
Mean `h_mean_composite − a_mean_composite` by outcome — confirms the features carry signal:

| Outcome | Mean composite diff |
|---------|---------------------|
| H (home win) | **+0.172** |
| D (draw) | −0.024 |
| A (away win) | **−0.178** |

### Top feature correlations with outcome (H=+1, D=0, A=−1)

| Rank | Feature | r |
|------|---------|---|
| 1 | `h_mean_composite` | +0.338 |
| 2 | `h_weighted_composite` | +0.319 |
| 3 | `a_mean_composite` | −0.318 |
| 4 | `a_weighted_composite` | −0.299 |
| 5 | `h_top3_composite` | +0.299 |
| 6 | `a_top3_composite` | −0.286 |
| 7 | `h_mid_composite` | +0.254 |
| 8 | `a_mid_composite` | −0.252 |
| 9 | `h_roll5_attack_str` | +0.242 |
| 10 | `h_att2_composite` | +0.216 |

Home team GK composite shows a counterintuitive negative correlation (r = −0.174) — likely because GKs in winning teams face fewer shots and thus have fewer opportunities to record saves and accumulate composite score.

### Output files

| File | Rows | Columns | Size |
|------|------|---------|------|
| `match_features_train.parquet` | 794 | 37 | 176 KB |
| `match_features_test.parquet` | 250 | 37 | 71 KB |

---

## 8. Pending Stages

| Stage | Notebook | Owner | Status |
|-------|----------|-------|--------|
| Clustering | `06_clustering.ipynb` | Nathaniel | Not built |
| Classification | `08_classification.ipynb` | Nathaniel | Not built |
