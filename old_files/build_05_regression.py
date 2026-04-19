"""Creates notebooks/05_regression.ipynb"""
import json
from pathlib import Path

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src, cell_id=None):
    c = {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": src}
    if cell_id:
        c["id"] = cell_id
    cells.append(c)

md("""\
# 05 — Regression
Predict player composite scores using Ridge Regression and Random Forest.
Two feature variants: baseline (rolling stats only) vs enhanced (+ team/opponent
strength context). Position-grouped models vs single global model.

| Metric | Description |
|--------|-------------|
| RMSE   | Root mean squared error |
| MAE    | Mean absolute error |
| R²     | Coefficient of determination |

**CV strategy:** `TimeSeriesSplit(n_splits=5)` within training seasons.
""")

code("""\
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DS_DIR    = Path('../data/processed/datasets')
MODEL_DIR = Path('../data/models')

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110

of_train = pd.read_parquet(DS_DIR / 'outfield_train_scaled.parquet')
of_test  = pd.read_parquet(DS_DIR / 'outfield_test_scaled.parquet')

# Sort by date to respect temporal order
of_train = of_train.sort_values('match_date').reset_index(drop=True)
of_test  = of_test.sort_values('match_date').reset_index(drop=True)

print(f'Train: {of_train.shape}   Test: {of_test.shape}')
print(f'Target (composite_score) train nulls: {of_train["composite_score"].isnull().sum()}')
""")

md("## 1 — Feature sets")

code("""\
# Identifiers and target — never features
NON_FEATURE = {
    'match_id', 'round', 'match_date', 'home_team', 'away_team',
    'player_id', 'player_name', 'team_id', 'team_name', 'shirt_number',
    'position_id', 'position_id_int', 'position_group', 'is_goalkeeper', 'season',
    'opponent', 'result', 'rating_title', 'composite_score',
}

# Baseline: rolling stats + home_away + minutes_played
BASELINE_FEAT = [c for c in of_train.columns
                 if c not in NON_FEATURE and c.startswith('roll5_')]
BASELINE_FEAT += ['home_away', 'minutes_played']

# Enhanced: baseline + team/opponent strength rolling averages
# (team_goals_scored and opp_goals_scored are per-match; we compute rolling averages below)
ENHANCED_EXTRA = ['roll5_team_goals_scored', 'roll5_opp_goals_scored']

def add_strength_features(df):
    df = df.copy().sort_values(['player_id', 'match_date'])
    # Rolling team/opp strength (per player history — proxy for fixture difficulty)
    for col in ['team_goals_scored', 'opp_goals_scored']:
        rolled = (
            df.groupby('player_id')[col]
            .apply(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        df[f'roll5_{col}'] = rolled
    df[[f'roll5_team_goals_scored', f'roll5_opp_goals_scored']] = (
        df[[f'roll5_team_goals_scored', f'roll5_opp_goals_scored']].fillna(
            df[['team_goals_scored', 'opp_goals_scored']].mean()
        )
    )
    return df

of_train = add_strength_features(of_train)
of_test  = add_strength_features(of_test)

ENHANCED_FEAT = BASELINE_FEAT + [c for c in ENHANCED_EXTRA if c in of_train.columns]

print(f'Baseline features : {len(BASELINE_FEAT)}')
print(f'Enhanced features : {len(ENHANCED_FEAT)}')
""")

md("""\
## 2 — Baseline model
Predict training-set mean composite score per position group.
""")

code("""\
def evaluate(y_true, y_pred, label=''):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f'{label:<35}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  (n={len(y_true):,})')
    return {'label': label, 'rmse': rmse, 'mae': mae, 'r2': r2, 'n': len(y_true)}

results = []

# Predict position-group mean
of_train_valid = of_train.dropna(subset=['composite_score'])
of_test_valid  = of_test.dropna(subset=['composite_score'])

pos_means = of_train_valid.groupby('position_group')['composite_score'].mean()
baseline_pred_train = of_train_valid['position_group'].map(pos_means)
baseline_pred_test  = of_test_valid['position_group'].map(pos_means).fillna(of_train_valid['composite_score'].mean())

results.append(evaluate(of_train_valid['composite_score'], baseline_pred_train, 'Baseline (pos mean) — train'))
results.append(evaluate(of_test_valid['composite_score'],  baseline_pred_test,  'Baseline (pos mean) — test'))
""")

md("## 3 — Ridge Regression")

code("""\
tscv = TimeSeriesSplit(n_splits=5)

def run_ridge(X_tr, y_tr, X_te, y_te, label):
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100, 500]}
    ridge = GridSearchCV(Ridge(), param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    ridge.fit(X_tr, y_tr)
    best = ridge.best_estimator_
    print(f'  Best alpha: {best.alpha}')
    results.append(evaluate(y_tr, best.predict(X_tr), f'Ridge {label} — train'))
    results.append(evaluate(y_te, best.predict(X_te), f'Ridge {label} — test'))
    return best

of_tr = of_train_valid.dropna(subset=BASELINE_FEAT)
of_te = of_test_valid.dropna(subset=BASELINE_FEAT)

print('Ridge — baseline features:')
ridge_baseline = run_ridge(
    of_tr[BASELINE_FEAT], of_tr['composite_score'],
    of_te[BASELINE_FEAT], of_te['composite_score'],
    'baseline'
)

of_tr_e = of_train_valid.dropna(subset=ENHANCED_FEAT)
of_te_e = of_test_valid.dropna(subset=ENHANCED_FEAT)

print()
print('Ridge — enhanced features:')
ridge_enhanced = run_ridge(
    of_tr_e[ENHANCED_FEAT], of_tr_e['composite_score'],
    of_te_e[ENHANCED_FEAT], of_te_e['composite_score'],
    'enhanced'
)
""")

md("## 4 — Random Forest Regressor")

code("""\
from scipy.stats import randint, uniform

def run_rf(X_tr, y_tr, X_te, y_te, label, n_iter=30):
    param_dist = {
        'n_estimators' : [100, 200, 300],
        'max_depth'    : [4, 6, 8, 10, None],
        'min_samples_leaf': randint(5, 30),
        'max_features' : ['sqrt', 0.5],
    }
    rf = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_dist, n_iter=n_iter, cv=tscv,
        scoring='neg_root_mean_squared_error',
        random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    best = rf.best_estimator_
    print(f'  Best params: {rf.best_params_}')
    results.append(evaluate(y_tr, best.predict(X_tr), f'RF {label} — train'))
    results.append(evaluate(y_te, best.predict(X_te), f'RF {label} — test'))
    return best

print('Random Forest — baseline features:')
rf_baseline = run_rf(
    of_tr[BASELINE_FEAT], of_tr['composite_score'],
    of_te[BASELINE_FEAT], of_te['composite_score'],
    'baseline'
)

print()
print('Random Forest — enhanced features:')
rf_enhanced = run_rf(
    of_tr_e[ENHANCED_FEAT], of_tr_e['composite_score'],
    of_te_e[ENHANCED_FEAT], of_te_e['composite_score'],
    'enhanced'
)
""")

md("## 5 — Position-grouped RF (best model)")

code("""\
grouped_models = {}
grouped_results = []

for pos in ['defender', 'midfielder', 'winger', 'forward']:
    print(f'\\n--- {pos} ---')
    tr_pos = of_tr_e[of_tr_e['position_group'] == pos]
    te_pos = of_te_e[of_te_e['position_group'] == pos]
    if len(tr_pos) < 50:
        print(f'  Skipped — only {len(tr_pos)} train rows')
        continue

    param_dist = {
        'n_estimators'    : [100, 200, 300],
        'max_depth'       : [4, 6, 8, 10, None],
        'min_samples_leaf': randint(3, 20),
        'max_features'    : ['sqrt', 0.5],
    }
    rf_pos = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_dist, n_iter=20, cv=tscv,
        scoring='neg_root_mean_squared_error',
        random_state=42, n_jobs=-1
    )
    rf_pos.fit(tr_pos[ENHANCED_FEAT], tr_pos['composite_score'])
    best = rf_pos.best_estimator_
    print(f'  Best params: {rf_pos.best_params_}')

    r = evaluate(te_pos['composite_score'], best.predict(te_pos[ENHANCED_FEAT]),
                 f'RF grouped {pos} — test')
    grouped_results.append(r)
    grouped_models[pos] = best

# Save position-grouped RF models
for pos, model in grouped_models.items():
    path = MODEL_DIR / f'rf_regression_{pos}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'✓ rf_regression_{pos}.pkl saved')

print()
print('RF grouped enhanced — overall test R²:',
      round(np.mean([r['r2'] for r in grouped_results]), 4))
""")

md("## 6 — Results summary table")

code("""\
results_df = pd.DataFrame(results)
print(results_df[results_df['label'].str.contains('test')].sort_values('r2', ascending=False).to_string(index=False))
""")

md("## 7 — Feature importance (best RF model — enhanced global)")

code("""\
importances = pd.Series(rf_enhanced.feature_importances_, index=ENHANCED_FEAT).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
importances.head(25).plot.barh(ax=ax, color='steelblue')
ax.invert_yaxis()
ax.set_title('RF enhanced — top 25 feature importances')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.show()

print('Top 10 features:')
print(importances.head(10).round(4).to_string())
""")

md("## 8 — Per-position test R² bar chart")

code("""\
fig, ax = plt.subplots(figsize=(8, 5))
gr_df = pd.DataFrame(grouped_results)
gr_df = gr_df.sort_values('r2', ascending=False)
bars = ax.bar(gr_df['label'].str.replace('RF grouped ', '').str.replace(' — test', ''),
              gr_df['r2'], color='steelblue', edgecolor='white')
ax.bar_label(bars, fmt='%.3f', padding=3)
ax.set_ylabel('R²')
ax.set_title('RF grouped enhanced — test R² by position')
ax.set_ylim(0, max(gr_df['r2'].max() * 1.3, 0.2))
plt.tight_layout()
plt.show()
""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path(__file__).parent / '05_regression.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'Written: {out}')
