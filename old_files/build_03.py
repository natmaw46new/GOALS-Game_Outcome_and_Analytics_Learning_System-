"""Creates notebooks/03_feature_engineering.ipynb"""
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

# ── Cell 0: Header ─────────────────────────────────────────────────────────────
md("""\
# 03 — Feature Engineering
Temporal train/test split → ratio features → position mapping → 5-match rolling
windows (shift-1, no leakage) → composite scores (ATT / MID / DEF / GK) →
position-grouped RobustScaler → sensitivity analysis (3 weight schemes).

| Output | Path |
|--------|------|
| `outfield_train_scaled.parquet` | `data/processed/datasets/` |
| `outfield_test_scaled.parquet`  | `data/processed/datasets/` |
| `gk_train_scaled.parquet`       | `data/processed/datasets/` |
| `gk_test_scaled.parquet`        | `data/processed/datasets/` |
| `scalers_outfield.pkl`          | `data/models/` |
| `scaler_gk.pkl`                 | `data/models/` |
| `position_map.pkl`              | `data/models/` |
""")

# ── Cell 1: Imports ────────────────────────────────────────────────────────────
code("""\
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import RobustScaler

PROC_DIR    = Path('../data/processed')
DS_DIR      = PROC_DIR / 'datasets'
MODEL_DIR   = Path('../data/models')
DS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SEASONS = {'2021_2022', '2022_2023', '2023_2024'}
TEST_SEASONS  = {'2024_2025'}

pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', '{:.3f}'.format)
print('✓ Ready')
""")

# ── Cell 2: Load clean parquets ────────────────────────────────────────────────
md("## 1 — Load clean data")
code("""\
outfield = pd.read_parquet(PROC_DIR / 'outfield_clean.parquet')
gk       = pd.read_parquet(PROC_DIR / 'gk_clean.parquet')

print(f'Outfield : {outfield.shape}')
print(f'GK       : {gk.shape}')
print()
print('Seasons present:')
print('  outfield:', sorted(outfield['season'].unique()))
print('  gk      :', sorted(gk['season'].unique()))
""")

# ── Cell 3: Train/test split ───────────────────────────────────────────────────
md("""\
## 2 — Temporal train / test split
- **Train:** 2021/22, 2022/23, 2023/24
- **Test:** 2024/25

All scalers and fill values are fit on train only.
""")
code("""\
def split_temporal(df):
    train = df[df['season'].isin(TRAIN_SEASONS)].copy()
    test  = df[df['season'].isin(TEST_SEASONS)].copy()
    return train, test

of_train, of_test = split_temporal(outfield)
gk_train, gk_test = split_temporal(gk)

for label, tr, te in [('outfield', of_train, of_test), ('gk', gk_train, gk_test)]:
    print(f'{label}  train {tr.shape}  test {te.shape}')
""")

# ── Cell 4: Position mapping ───────────────────────────────────────────────────
md("""\
## 3 — Position mapping
FotMob numeric `position_id` → 4 outfield groups (defender / midfielder / winger / forward).
The ranges below were derived by cross-referencing known player names with their IDs.

| Group | position_id range |
|-------|-------------------|
| defender | 32–38, 51–52, 58–59, 62, 71, 79 |
| midfielder | 63–68, 73–77, 95 |
| winger | 72, 78, 82–88, 94, 96, 107 |
| forward | 103–106, 114–116 |
""")
code("""\
POSITION_MAP = {
    # defenders — CBs, LBs, RBs, wing-backs
    32: 'defender', 33: 'defender', 34: 'defender', 35: 'defender',
    36: 'defender', 37: 'defender', 38: 'defender',
    51: 'defender', 52: 'defender', 58: 'defender', 59: 'defender',
    62: 'defender', 71: 'defender', 79: 'defender',
    # midfielders — DMs, CMs, box-to-box
    63: 'midfielder', 64: 'midfielder', 65: 'midfielder', 66: 'midfielder',
    67: 'midfielder', 68: 'midfielder',
    73: 'midfielder', 74: 'midfielder', 75: 'midfielder', 76: 'midfielder',
    77: 'midfielder', 95: 'midfielder',
    # wingers — wide AMs, inside forwards, attacking midfielders
    72: 'winger', 78: 'winger', 82: 'winger', 83: 'winger',
    84: 'winger', 85: 'winger', 86: 'winger', 87: 'winger',
    88: 'winger', 94: 'winger', 96: 'winger', 107: 'winger',
    # forwards — STs, CFs, second-strikers
    103: 'forward', 104: 'forward', 105: 'forward', 106: 'forward',
    114: 'forward', 115: 'forward', 116: 'forward',
}

def map_positions(df):
    df = df.copy()
    df['position_id_int'] = df['position_id'].astype('Int64')
    df['position_group'] = df['position_id_int'].map(POSITION_MAP).fillna('midfielder')
    return df

of_train = map_positions(of_train)
of_test  = map_positions(of_test)

print('Train position distribution:')
print(of_train['position_group'].value_counts())
print()
print('Unmapped position_ids (assigned to midfielder):')
unmapped = of_train.loc[~of_train['position_id_int'].isin(POSITION_MAP), 'position_id_int'].value_counts()
print(unmapped)

# Save position map for inference
with open(MODEL_DIR / 'position_map.pkl', 'wb') as f:
    pickle.dump(POSITION_MAP, f)
print('✓ position_map.pkl saved')
""")

# ── Cell 5: Ratio features — outfield ─────────────────────────────────────────
md("""\
## 4 — Ratio features
Compute rates from numerator/denominator pairs; fill 0/0 divisions with
**training-set mean** (computed here, applied to both train and test).
Denominator columns are dropped after ratio creation.
""")
code("""\
OUTFIELD_RATIOS = [
    # (new_col, numerator, denominator)
    ('pass_accuracy',        'accurate_passes',     'accurate_passes_total'),
    ('long_ball_accuracy',   'long_balls_accurate',  'long_balls_accurate_total'),
    ('cross_accuracy',       'accurate_crosses',     'accurate_crosses_total'),
    ('dribble_success_rate', 'dribbles_succeeded',   'dribbles_succeeded_total'),
    ('aerial_win_rate',      'aerials_won',          'aerials_won_total'),
    ('ground_duel_win_rate', 'ground_duels_won',     'ground_duels_won_total'),
    ('shot_accuracy',        'ShotsOnTarget',        'shot_accuracy_total'),   # total shots proxy
]

GK_RATIOS = [
    ('pass_accuracy',        'accurate_passes',     'accurate_passes_total'),
    ('long_ball_accuracy',   'long_balls_accurate',  'long_balls_accurate_total'),
    ('aerial_win_rate',      'aerials_won',          'aerials_won_total'),
    ('ground_duel_win_rate', 'ground_duels_won',     'ground_duels_won_total'),
]


def compute_ratios(train, test, ratios):
    train, test = train.copy(), test.copy()
    fill_means = {}
    denom_cols = set()

    for new_col, num, denom in ratios:
        if num not in train.columns or denom not in train.columns:
            print(f'  SKIP {new_col} — missing {num} or {denom}')
            continue
        denom_cols.add(denom)

        # compute ratio — NaN where denom == 0
        train[new_col] = np.where(train[denom] > 0, train[num] / train[denom], np.nan)
        test[new_col]  = np.where(test[denom]  > 0, test[num]  / test[denom],  np.nan)

        # fill NaN with training mean
        fill_val = train[new_col].mean()
        fill_means[new_col] = fill_val
        train[new_col] = train[new_col].fillna(fill_val)
        test[new_col]  = test[new_col].fillna(fill_val)

    # drop denominator columns (numerators kept for composite scores)
    denom_cols_present = [c for c in denom_cols if c in train.columns]
    train = train.drop(columns=denom_cols_present)
    test  = test.drop(columns=denom_cols_present)

    print(f'  Ratios computed: {[r[0] for r in ratios]}')
    print(f'  Dropped denominators: {denom_cols_present}')
    return train, test, fill_means


of_train, of_test, of_ratio_means = compute_ratios(of_train, of_test, OUTFIELD_RATIOS)
gk_train, gk_test, gk_ratio_means = compute_ratios(gk_train, gk_test, GK_RATIOS)

# save fill means for inference
with open(MODEL_DIR / 'ratio_fill_means.pkl', 'wb') as f:
    pickle.dump({'outfield': of_ratio_means, 'gk': gk_ratio_means}, f)
print('✓ ratio_fill_means.pkl saved')
""")

# ── Cell 6: GK save_rate separately ───────────────────────────────────────────
code("""\
# GK save_rate — shots faced inferred from goals_conceded + saves
# saves_inside_box already present; no explicit shot_faced col — use saves / (saves + goals_conceded)
def add_gk_save_rate(train, test):
    train, test = train.copy(), test.copy()
    for df in (train, test):
        denom = df['saves'] + df['goals_conceded']
        df['save_rate'] = np.where(denom > 0, df['saves'] / denom, np.nan)

    fill_val = train['save_rate'].mean()
    train['save_rate'] = train['save_rate'].fillna(fill_val)
    test['save_rate']  = test['save_rate'].fillna(fill_val)
    return train, test

gk_train, gk_test = add_gk_save_rate(gk_train, gk_test)
print('save_rate added — sample:')
print(gk_train['save_rate'].describe().to_string())
""")

# ── Cell 7: Null fill for remaining numeric cols ───────────────────────────────
md("""\
## 5 — Null imputation
Fill remaining null numeric values with **training-set medians**. Fit on train only.
""")
code("""\
def impute_with_train_median(train, test):
    train, test = train.copy(), test.copy()
    num_cols = train.select_dtypes(include='number').columns.tolist()
    medians  = train[num_cols].median()
    train[num_cols] = train[num_cols].fillna(medians)
    test[num_cols]  = test[num_cols].fillna(medians)
    remaining_train = train[num_cols].isnull().sum().sum()
    remaining_test  = test[num_cols].isnull().sum().sum()
    print(f'  Nulls after imputation — train: {remaining_train}, test: {remaining_test}')
    return train, test, medians

of_train, of_test, of_medians = impute_with_train_median(of_train, of_test)
gk_train, gk_test, gk_medians = impute_with_train_median(gk_train, gk_test)

with open(MODEL_DIR / 'imputation_medians.pkl', 'wb') as f:
    pickle.dump({'outfield': of_medians, 'gk': gk_medians}, f)
print('✓ imputation_medians.pkl saved')
""")

# ── Cell 8: 5-match rolling windows ───────────────────────────────────────────
md("""\
## 6 — 5-match rolling windows (shift-1)
Per player, sort by `match_date`, apply `.shift(1).rolling(5).mean()` so the current
match is **never** included in its own window. Drop rows where rolling is NaN
(each player's first ≤5 matches that have no prior history).
""")
code("""\
def add_rolling_features(df: pd.DataFrame, roll_cols: list[str]) -> pd.DataFrame:
    df = df.sort_values(['player_id', 'match_date']).copy()

    roll_cols_present = [c for c in roll_cols if c in df.columns]

    rolled = (
        df.groupby('player_id')[roll_cols_present]
        .apply(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    rolled.columns = [f'roll5_{c}' for c in roll_cols_present]

    df = pd.concat([df, rolled], axis=1)

    # Drop rows where ALL rolling features are NaN (first appearance of player)
    roll_col_names = rolled.columns.tolist()
    df = df.dropna(subset=roll_col_names, how='all').reset_index(drop=True)

    return df


# Numeric columns to roll (exclude identifiers, context, and ratio cols already computed)
ROLL_EXCLUDE = {
    'match_id', 'round', 'match_date', 'home_team', 'away_team',
    'player_id', 'player_name', 'team_id', 'team_name', 'shirt_number',
    'position_id', 'position_id_int', 'position_group', 'is_goalkeeper', 'season',
    'home_away', 'opponent', 'team_goals_scored', 'opp_goals_scored', 'result',
    'minutes_played', 'rating_title',
}

of_num_cols  = [c for c in of_train.select_dtypes(include='number').columns if c not in ROLL_EXCLUDE]
gk_num_cols  = [c for c in gk_train.select_dtypes(include='number').columns if c not in ROLL_EXCLUDE]

print(f'Outfield columns to roll: {len(of_num_cols)}')
print(f'GK columns to roll:       {len(gk_num_cols)}')

of_train = add_rolling_features(of_train, of_num_cols)
of_test  = add_rolling_features(of_test,  of_num_cols)
gk_train = add_rolling_features(gk_train, gk_num_cols)
gk_test  = add_rolling_features(gk_test,  gk_num_cols)

print()
for label, tr, te in [('outfield', of_train, of_test), ('gk', gk_train, gk_test)]:
    print(f'{label}  train {tr.shape}  test {te.shape}')
""")

# ── Cell 9: Composite score construction ──────────────────────────────────────
md("""\
## 7 — Composite score construction
Apply position-specific weighted formulas from the proposal. Each input metric is
z-score normalised using **training-set mean and std** (fit on train, applied to both).

Three weight schemes are tested in the sensitivity analysis cell.
Scheme A (proposal weights) is the canonical scheme.
""")
code("""\
# ── Scheme A: proposal weights ──────────────────────────────────────────────
WEIGHTS_A = {
    'defender': {
        'roll5_matchstats.headers.tackles': 0.25,
        'roll5_aerials_won'               : 0.20,
        'roll5_clearances'                : 0.20,
        'roll5_interceptions'             : 0.15,
        'roll5_shot_blocks'               : 0.10,
        'roll5_accurate_passes'           : 0.10,
    },
    'midfielder': {
        'roll5_accurate_passes'           : 0.20,
        'roll5_chances_created'           : 0.20,
        'roll5_expected_assists'          : 0.15,
        'roll5_goals'                     : 0.075,
        'roll5_assists'                   : 0.075,
        'roll5_matchstats.headers.tackles': 0.15,
        'roll5_interceptions'             : 0.10,
        'roll5_recoveries'                : 0.05,
    },
    'winger': {
        'roll5_goals'             : 0.125,
        'roll5_assists'           : 0.125,
        'roll5_expected_goals'    : 0.20,
        'roll5_expected_assists'  : 0.15,
        'roll5_dribbles_succeeded': 0.15,
        'roll5_chances_created'   : 0.10,
        'roll5_recoveries'        : 0.05,
        'roll5_shot_accuracy'     : 0.10,
    },
    'forward': {
        'roll5_goals'             : 0.125,
        'roll5_assists'           : 0.125,
        'roll5_expected_goals'    : 0.20,
        'roll5_expected_assists'  : 0.15,
        'roll5_dribbles_succeeded': 0.15,
        'roll5_chances_created'   : 0.10,
        'roll5_recoveries'        : 0.05,
        'roll5_shot_accuracy'     : 0.10,
    },
}

# ── Scheme B: equal weights ──────────────────────────────────────────────────
def equal_weights(w_dict):
    return {pos: {col: 1/len(cols) for col in cols}
            for pos, cols in w_dict.items()}

WEIGHTS_B = equal_weights(WEIGHTS_A)

# ── GK weights (single scheme — no position sub-groups) ─────────────────────
GK_WEIGHTS_A = {
    'roll5_saves'                          : 0.30,
    'roll5_expected_goals_on_target_faced' : 0.25,
    'roll5_keeper_diving_save'             : 0.15,
    'roll5_saves_inside_box'               : 0.15,
    'roll5_keeper_high_claim'              : 0.10,
    'roll5_keeper_sweeper'                 : 0.05,
}
GK_WEIGHTS_B = {col: 1/len(GK_WEIGHTS_A) for col in GK_WEIGHTS_A}


def compute_composite(df: pd.DataFrame, weights_by_pos: dict, z_params: dict = None):
    \"\"\"
    Compute composite score per row using position-specific weights.
    z_params: dict of {col: (mean, std)} from training set.
    Returns (df_with_score, z_params).
    \"\"\"
    df = df.copy()
    scores = pd.Series(np.nan, index=df.index)
    z_params_out = {}

    all_cols = set()
    for cols in weights_by_pos.values():
        all_cols.update(cols.keys())

    # Fit z-score params on full passed df (caller slices train only for fitting)
    if z_params is None:
        z_params = {}
        for col in all_cols:
            if col in df.columns:
                z_params[col] = (df[col].mean(), df[col].std(ddof=0))
    z_params_out = z_params

    for pos, weights in weights_by_pos.items():
        mask = df['position_group'] == pos
        if mask.sum() == 0:
            continue
        score = pd.Series(0.0, index=df[mask].index)
        for col, w in weights.items():
            if col not in df.columns:
                continue
            mu, sigma = z_params.get(col, (0.0, 1.0))
            z = (df.loc[mask, col] - mu) / (sigma if sigma > 0 else 1.0)
            score += w * z
        scores[mask] = score

    df['composite_score'] = scores
    return df, z_params_out


def compute_gk_composite(df: pd.DataFrame, weights: dict, z_params: dict = None):
    df = df.copy()
    if z_params is None:
        z_params = {}
        for col in weights:
            if col in df.columns:
                z_params[col] = (df[col].mean(), df[col].std(ddof=0))

    score = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col not in df.columns:
            print(f'  Missing GK col: {col}')
            continue
        mu, sigma = z_params.get(col, (0.0, 1.0))
        z = (df[col] - mu) / (sigma if sigma > 0 else 1.0)
        score += w * z

    df['composite_score'] = score
    return df, z_params


# ── Fit on train, apply to test ──────────────────────────────────────────────
of_train, of_z_params_A = compute_composite(of_train, WEIGHTS_A)
of_test,  _             = compute_composite(of_test,  WEIGHTS_A, z_params=of_z_params_A)

gk_train, gk_z_params_A = compute_gk_composite(gk_train, GK_WEIGHTS_A)
gk_test,  _             = compute_gk_composite(gk_test,  GK_WEIGHTS_A, z_params=gk_z_params_A)

print('Composite score (train) by position:')
print(of_train.groupby('position_group')['composite_score'].describe().round(3))
print()
print('GK composite score (train):')
print(gk_train['composite_score'].describe().round(3))
""")

# ── Cell 10: GK ratio clipping ─────────────────────────────────────────────────
code("""\
# Clip GK extreme ratio values (sparse denominators produce outliers)
GK_CLIP_COLS = ['roll5_aerial_win_rate', 'roll5_ground_duel_win_rate',
                'roll5_save_rate', 'roll5_goals_prevented']

for col in GK_CLIP_COLS:
    for df in (gk_train, gk_test):
        if col not in df.columns:
            continue
    mu    = gk_train[col].mean() if col in gk_train.columns else 0
    sigma = gk_train[col].std()  if col in gk_train.columns else 1
    lo, hi = mu - 5*sigma, mu + 5*sigma
    if col in gk_train.columns:
        gk_train[col] = gk_train[col].clip(lo, hi)
    if col in gk_test.columns:
        gk_test[col]  = gk_test[col].clip(lo, hi)

print('GK ratio clipping applied (±5 std).')
""")

# ── Cell 11: Sensitivity analysis ─────────────────────────────────────────────
md("""\
## 8 — Sensitivity analysis: 3 weight schemes
Compare composite score vs FotMob `rating_title` across weight schemes using
**Pearson correlation** and RMSE on the training set (no model, just the raw
weighted composite vs the rating). This validates whether the proposed weights
produce a more discriminating performance signal than equal weighting.

Scheme C is data-driven: weights proportional to |Pearson r| of each rolling feature
with `rating_title`.
""")
code("""\
from scipy.stats import pearsonr

def scheme_c_weights(df: pd.DataFrame, weights_A: dict) -> dict:
    \"\"\"Compute correlation-based weights per position group.\"\"\"
    weights_C = {}
    for pos, cols in weights_A.items():
        mask = df['position_group'] == pos
        sub  = df[mask]
        r_vals = {}
        for col in cols:
            if col in sub.columns:
                valid = sub[[col, 'rating_title']].dropna()
                if len(valid) > 10:
                    r, _ = pearsonr(valid[col], valid['rating_title'])
                    r_vals[col] = abs(r)
                else:
                    r_vals[col] = 0.0
        total = sum(r_vals.values()) or 1
        weights_C[pos] = {col: r/total for col, r in r_vals.items()}
    return weights_C

WEIGHTS_C = scheme_c_weights(of_train, WEIGHTS_A)

# Compute composite for each scheme
def composite_corr_rmse(df: pd.DataFrame, weights: dict, label: str):
    scored_df, _ = compute_composite(df, weights)
    valid = scored_df[['composite_score', 'rating_title']].dropna()
    if len(valid) < 2:
        print(f'{label}: insufficient data')
        return
    r, _ = pearsonr(valid['composite_score'], valid['rating_title'])
    rmse = np.sqrt(((valid['composite_score'] - valid['rating_title'])**2).mean())
    print(f'{label:<12}  r={r:+.4f}  RMSE={rmse:.4f}  (n={len(valid):,})')

print('Composite score vs FotMob rating_title (train, all outfield):')
composite_corr_rmse(of_train, WEIGHTS_A, 'Scheme A')
composite_corr_rmse(of_train, WEIGHTS_B, 'Scheme B (equal)')
composite_corr_rmse(of_train, WEIGHTS_C, 'Scheme C (data)')

print()
print('Per-position breakdown — Scheme A:')
for pos in WEIGHTS_A:
    sub = of_train[of_train['position_group'] == pos]
    composite_corr_rmse(sub, {pos: WEIGHTS_A[pos]}, f'  {pos}')
""")

# ── Cell 12: RobustScaler ──────────────────────────────────────────────────────
md("""\
## 9 — Position-grouped RobustScaler
Fit one `RobustScaler` per outfield position group on training data, transform both
train and test. Scale numeric rolling + ratio features (not identifiers or context cols).
""")
code("""\
SCALE_EXCLUDE = {
    'match_id', 'round', 'match_date', 'home_team', 'away_team',
    'player_id', 'player_name', 'team_id', 'team_name', 'shirt_number',
    'position_id', 'position_id_int', 'position_group', 'is_goalkeeper', 'season',
    'home_away', 'opponent', 'team_goals_scored', 'opp_goals_scored', 'result',
    'minutes_played', 'rating_title', 'composite_score',
}

def fit_scale_grouped(train, test, group_col='position_group'):
    train, test = train.copy(), test.copy()
    scalers = {}

    num_cols = [c for c in train.select_dtypes(include='number').columns
                if c not in SCALE_EXCLUDE]

    for grp in train[group_col].unique():
        mask_tr = train[group_col] == grp
        mask_te = test[group_col]  == grp

        scaler = RobustScaler()
        train.loc[mask_tr, num_cols] = scaler.fit_transform(train.loc[mask_tr, num_cols])
        if mask_te.sum() > 0:
            test.loc[mask_te, num_cols] = scaler.transform(test.loc[mask_te, num_cols])
        scalers[grp] = scaler
        print(f'  {grp}: scaler fit on {mask_tr.sum():,} rows, applied to {mask_te.sum():,} test rows')

    return train, test, scalers


def fit_scale_single(train, test):
    train, test = train.copy(), test.copy()
    num_cols = [c for c in train.select_dtypes(include='number').columns
                if c not in SCALE_EXCLUDE]
    scaler = RobustScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])
    test[num_cols]  = scaler.transform(test[num_cols])
    return train, test, scaler


print('Outfield — position-grouped scaling:')
of_train, of_test, scalers_outfield = fit_scale_grouped(of_train, of_test)

print()
print('GK — single scaler:')
gk_train, gk_test, scaler_gk = fit_scale_single(gk_train, gk_test)

# Save scalers
with open(MODEL_DIR / 'scalers_outfield.pkl', 'wb') as f:
    pickle.dump(scalers_outfield, f)
with open(MODEL_DIR / 'scaler_gk.pkl', 'wb') as f:
    pickle.dump(scaler_gk, f)
print()
print('✓ scalers_outfield.pkl saved')
print('✓ scaler_gk.pkl saved')
""")

# ── Cell 13: Final shape summary ───────────────────────────────────────────────
md("## 10 — Final summary")
code("""\
print('='*60)
for label, tr, te in [
        ('outfield_train', of_train, None),
        ('outfield_test',  of_test,  None),
        ('gk_train',       gk_train, None),
        ('gk_test',        gk_test,  None),
    ]:
    df = tr
    print(f'  {label:<18} : {df.shape[0]:>6,} rows  x  {df.shape[1]:>3} cols')
print('='*60)
print()
print('composite_score nulls — outfield train:', of_train['composite_score'].isnull().sum())
print('composite_score nulls — gk train      :', gk_train['composite_score'].isnull().sum())
""")

# ── Cell 14: Export ─────────────────────────────────────────────────────────────
md("## 11 — Export")
code("""\
of_train.to_parquet(DS_DIR / 'outfield_train_scaled.parquet', index=False)
of_test.to_parquet( DS_DIR / 'outfield_test_scaled.parquet',  index=False)
gk_train.to_parquet(DS_DIR / 'gk_train_scaled.parquet',       index=False)
gk_test.to_parquet( DS_DIR / 'gk_test_scaled.parquet',        index=False)

for name in ['outfield_train_scaled', 'outfield_test_scaled', 'gk_train_scaled', 'gk_test_scaled']:
    size = (DS_DIR / f'{name}.parquet').stat().st_size / 1e6
    print(f'✓ {name}.parquet saved  ({size:.1f} MB)')
""")

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path(__file__).parent / '03_feature_engineering.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'Written: {out}')
