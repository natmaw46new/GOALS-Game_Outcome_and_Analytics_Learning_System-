"""Creates notebooks/07_team_aggregation.ipynb"""
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

# ── Header ─────────────────────────────────────────────────────────────────────
md("""\
# 07 — Team Aggregation
Aggregates per-player composite scores to one row per match, with home-team
and away-team features side-by-side. Output feeds directly into the
Win/Draw/Loss classifier.

| Output | Path |
|--------|------|
| `match_features_train.parquet` | `data/processed/` |
| `match_features_test.parquet`  | `data/processed/` |

**Target column:** `outcome` — `H` (home win), `D` (draw), `A` (away win).
""")

# ── Cell 1: Imports ────────────────────────────────────────────────────────────
code("""\
import pandas as pd
import numpy as np
from pathlib import Path

DS_DIR   = Path('../data/processed/datasets')
PROC_DIR = Path('../data/processed')

TRAIN_SEASONS = {'2021_2022', '2022_2023', '2023_2024'}
TEST_SEASONS  = {'2024_2025'}

pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', '{:.3f}'.format)
print('✓ Ready')
""")

# ── Cell 2: Load data ──────────────────────────────────────────────────────────
md("## 1 — Load scaled datasets")
code("""\
of_train = pd.read_parquet(DS_DIR / 'outfield_train_scaled.parquet')
of_test  = pd.read_parquet(DS_DIR / 'outfield_test_scaled.parquet')
gk_train = pd.read_parquet(DS_DIR / 'gk_train_scaled.parquet')
gk_test  = pd.read_parquet(DS_DIR / 'gk_test_scaled.parquet')

# Combine for rolling computation (split again at export)
outfield = pd.concat([of_train, of_test], ignore_index=True)
gk       = pd.concat([gk_train, gk_test], ignore_index=True)

print(f'Outfield: {outfield.shape}')
print(f'GK:       {gk.shape}')
print(f'Seasons:  {sorted(outfield[\"season\"].unique())}')
""")

# ── Cell 3: Rolling team strength ──────────────────────────────────────────────
md("""\
## 2 — Rolling team strength features
`team_goals_scored` and `opp_goals_scored` in the player data are raw
match-level values (leakage if used directly). We compute a 5-match
rolling average at the team level, sorted by date, using `.shift(1)` so
the current match is excluded from its own window.
""")
code("""\
# Unique team-match observations (same for all players in team)
team_match = (
    outfield[['match_id', 'team_name', 'match_date', 'season',
              'team_goals_scored', 'opp_goals_scored']]
    .drop_duplicates(subset=['match_id', 'team_name'])
    .sort_values(['team_name', 'match_date'])
    .reset_index(drop=True)
)

# Rolling 5-match avg goals scored / conceded per team
for col, new_col in [('team_goals_scored', 'roll5_attack_str'),
                     ('opp_goals_scored',  'roll5_defence_str')]:
    team_match[new_col] = (
        team_match.groupby('team_name')[col]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

# Fill first-match NaNs with overall mean (computed on all data — benign, no future leakage)
team_match['roll5_attack_str']  = team_match['roll5_attack_str'].fillna(
    team_match['roll5_attack_str'].mean())
team_match['roll5_defence_str'] = team_match['roll5_defence_str'].fillna(
    team_match['roll5_defence_str'].mean())

print('Team-match strength table:', team_match.shape)
print(team_match[['team_name','match_date','team_goals_scored',
                  'roll5_attack_str','roll5_defence_str']].head(8).to_string())
""")

# ── Cell 4: GK per-match composite ────────────────────────────────────────────
md("## 3 — GK composite score per team per match")
code("""\
# If multiple GKs appear (substitution), keep the one with more minutes
gk_match = (
    gk.sort_values('minutes_played', ascending=False)
    .drop_duplicates(subset=['match_id', 'team_name'], keep='first')
    [['match_id', 'team_name', 'composite_score', 'season']]
    .rename(columns={'composite_score': 'gk_composite'})
)

print(f'GK match records: {len(gk_match)} (unique match+team)')
missing = outfield[['match_id','team_name']].drop_duplicates().merge(
    gk_match[['match_id','team_name']], on=['match_id','team_name'], how='left', indicator=True)
print(f'Matches missing GK data: {(missing["_merge"] == "left_only").sum()}')
""")

# ── Cell 5: Outfield team aggregation ─────────────────────────────────────────
md("""\
## 4 — Outfield aggregation to team level

For each (match, team) compute:
- `mean_composite` — mean composite of all outfield players
- `weighted_composite` — minutes-weighted composite
- `att_composite` — mean composite of wingers + forwards
- `mid_composite` — mean composite of midfielders
- `def_composite` — mean composite of defenders
- `top3_composite` — mean of top 3 composites (key performers)
- `composite_std` — spread of composites (lineup cohesion)
- `n_defenders`, `n_midfielders`, `n_wingers`, `n_forwards` — formation shape
""")
code("""\
def team_agg(grp):
    cs  = grp['composite_score'].values
    mp  = grp['minutes_played'].values
    pos = grp['position_group'].values

    weighted = np.sum(cs * mp) / np.sum(mp) if np.sum(mp) > 0 else np.nan

    def pos_mean(p):
        mask = pos == p
        return cs[mask].mean() if mask.any() else np.nan

    top3 = np.sort(cs)[-3:].mean() if len(cs) >= 3 else cs.mean()

    return pd.Series({
        'mean_composite'  : cs.mean(),
        'weighted_composite': weighted,
        'att_composite'   : pos_mean('forward'),   # forward = clinical strikers
        'att2_composite'  : pos_mean('winger'),    # winger = wide attackers
        'mid_composite'   : pos_mean('midfielder'),
        'def_composite'   : pos_mean('defender'),
        'top3_composite'  : top3,
        'composite_std'   : cs.std() if len(cs) > 1 else 0.0,
        'n_defenders'     : int((pos == 'defender').sum()),
        'n_midfielders'   : int((pos == 'midfielder').sum()),
        'n_wingers'       : int((pos == 'winger').sum()),
        'n_forwards'      : int((pos == 'forward').sum()),
        'n_players'       : len(cs),
        # match context (same value for all players in team-match)
        'home_away'       : int(grp['home_away'].iloc[0]),
        'result'          : grp['result'].iloc[0],
        'season'          : grp['season'].iloc[0],
        'match_date'      : grp['match_date'].iloc[0],
        'home_team'       : grp['home_team'].iloc[0],
        'away_team'       : grp['away_team'].iloc[0],
    })


team_features = (
    outfield.groupby(['match_id', 'team_name'])
    .apply(team_agg)
    .reset_index()
)

print(f'Team-match features: {team_features.shape}')
print(team_features.head(4).to_string())
""")

# ── Cell 6: Join GK + rolling strength ────────────────────────────────────────
md("## 5 — Join GK composite and team strength")
code("""\
team_features = team_features.merge(
    gk_match[['match_id', 'team_name', 'gk_composite']],
    on=['match_id', 'team_name'], how='left'
)

team_features = team_features.merge(
    team_match[['match_id', 'team_name', 'roll5_attack_str', 'roll5_defence_str']],
    on=['match_id', 'team_name'], how='left'
)

# Fill missing GK composite with overall GK mean
gk_mean = gk['composite_score'].mean()
team_features['gk_composite'] = team_features['gk_composite'].fillna(gk_mean)

print('Nulls after join:')
print(team_features.isnull().sum()[team_features.isnull().sum() > 0])
print()
print(f'Total team-match rows: {len(team_features)}  ({team_features[\"match_id\"].nunique()} matches)')
""")

# ── Cell 7: Pivot to match level ───────────────────────────────────────────────
md("""\
## 6 — Pivot to one row per match
Split into home-team rows and away-team rows, then join on `match_id`.
Home features get prefix `h_`, away features get prefix `a_`.
""")
code("""\
FEAT_COLS = [
    'mean_composite', 'weighted_composite',
    'att_composite', 'att2_composite', 'mid_composite', 'def_composite',
    'top3_composite', 'composite_std', 'gk_composite',
    'n_defenders', 'n_midfielders', 'n_wingers', 'n_forwards',
    'roll5_attack_str', 'roll5_defence_str',
]

home_rows = team_features[team_features['home_away'] == 1].copy()
away_rows = team_features[team_features['home_away'] == 0].copy()

home_feat = home_rows[['match_id'] + FEAT_COLS + ['result', 'season', 'match_date',
                                                    'home_team', 'away_team']].copy()
away_feat = away_rows[['match_id'] + FEAT_COLS].copy()

home_feat = home_feat.rename(columns={c: f'h_{c}' for c in FEAT_COLS})
away_feat = away_feat.rename(columns={c: f'a_{c}' for c in FEAT_COLS})

match_df = home_feat.merge(away_feat, on='match_id', how='inner')

# Outcome from home team perspective: W→H, D→D, L→A
outcome_map = {'W': 'H', 'D': 'D', 'L': 'A'}
match_df['outcome'] = match_df['result'].map(outcome_map)
match_df = match_df.drop(columns=['result'])

print(f'Match-level dataset: {match_df.shape}')
print()
print('Outcome distribution:')
print(match_df['outcome'].value_counts(normalize=True).mul(100).round(1))
print()
print(match_df[['match_id','home_team','away_team','outcome',
                'h_mean_composite','a_mean_composite',
                'h_gk_composite','a_gk_composite']].head(6).to_string())
""")

# ── Cell 8: Sanity checks ──────────────────────────────────────────────────────
md("## 7 — Sanity checks")
code("""\
# Outcome distribution per season
print('Outcome distribution per season:')
print(match_df.groupby('season')['outcome']
      .value_counts(normalize=True).mul(100).round(1)
      .unstack().to_string())
print()

# Feature nulls
null_check = match_df[[f'h_{c}' for c in FEAT_COLS] +
                       [f'a_{c}' for c in FEAT_COLS]].isnull().sum()
print('Feature nulls:')
print(null_check[null_check > 0] if null_check.any() else '  None — all clean ✓')
print()

# Mean composite differential — home vs away
match_df['composite_diff'] = match_df['h_mean_composite'] - match_df['a_mean_composite']
print('Composite differential by outcome (positive = home team stronger):')
print(match_df.groupby('outcome')['composite_diff'].mean().round(3))
""")

# ── Cell 9: Feature correlation with outcome ───────────────────────────────────
code("""\
from scipy.stats import pointbiserialr

# Encode outcome for correlation check
outcome_enc = match_df['outcome'].map({'H': 1, 'D': 0, 'A': -1})

print('Pearson r vs outcome encoding (H=1, D=0, A=-1):')
corr_rows = []
for col in [f'h_{c}' for c in FEAT_COLS] + [f'a_{c}' for c in FEAT_COLS]:
    valid = match_df[[col]].join(outcome_enc.rename('outcome_enc')).dropna()
    r = valid.corr().iloc[0, 1]
    corr_rows.append({'feature': col, 'r': r})

corr_df = pd.DataFrame(corr_rows).sort_values('r', key=abs, ascending=False)
print(corr_df.head(20).to_string(index=False))
""")

# ── Cell 10: Train / test split and export ─────────────────────────────────────
md("## 8 — Train / test split and export")
code("""\
match_train = match_df[match_df['season'].isin(TRAIN_SEASONS)].reset_index(drop=True)
match_test  = match_df[match_df['season'].isin(TEST_SEASONS)].reset_index(drop=True)

print(f'Train matches: {len(match_train)}')
print(f'Test  matches: {len(match_test)}')
print()
print('Train outcome distribution:')
print(match_train['outcome'].value_counts())
print()
print('Test outcome distribution:')
print(match_test['outcome'].value_counts())

match_train.to_parquet(PROC_DIR / 'match_features_train.parquet', index=False)
match_test.to_parquet( PROC_DIR / 'match_features_test.parquet',  index=False)

for name, df in [('match_features_train', match_train), ('match_features_test', match_test)]:
    size = (PROC_DIR / f'{name}.parquet').stat().st_size / 1e3
    print(f'✓ {name}.parquet  ({len(df)} rows x {df.shape[1]} cols  {size:.0f} KB)')
""")

# ── Write notebook ──────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path(__file__).parent / '07_team_aggregation.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'Written: {out}')
