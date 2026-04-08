"""Creates notebooks/02_preprocessing.ipynb"""
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

# ── Cell 0: Header ────────────────────────────────────────────────────────────
md("""\
# 02 — Preprocessing
Loads all four FotMob Premier League seasons, adds match context features
(home/away flag, opponent, goals scored, W/D/L result), fixes team name
inconsistencies, drops sparse columns, and exports clean parquets.

| Output | Path |
|--------|------|
| `outfield_clean.parquet` | `data/processed/` |
| `gk_clean.parquet` | `data/processed/` |
""")

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
code("""\
import pandas as pd
import numpy as np
from pathlib import Path

FOTMOB_BASE = Path('../data/47')
SEASONS     = ['2021_2022', '2022_2023', '2023_2024', '2024_2025']
OUT_DIR     = Path('../data/processed')
OUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', '{:.3f}'.format)
print('✓ Ready')
""")

# ── Cell 2: Load all seasons ──────────────────────────────────────────────────
md("## 1 — Load all seasons")

code("""\
outfield_frames = []
gk_frames       = []
fixture_frames  = []

for season in SEASONS:
    base = FOTMOB_BASE / season / 'output'
    o = pd.read_parquet(base / 'outfield_players.parquet')
    g = pd.read_parquet(base / 'goalkeepers.parquet')
    f = pd.read_parquet(base / 'fixtures.parquet')
    o['season'] = season
    g['season'] = season
    f['season'] = season
    outfield_frames.append(o)
    gk_frames.append(g)
    fixture_frames.append(f)

outfield = pd.concat(outfield_frames, ignore_index=True)
gk       = pd.concat(gk_frames,       ignore_index=True)
fixtures = pd.concat(fixture_frames,  ignore_index=True)

print(f'Outfield : {outfield.shape}')
print(f'GK       : {gk.shape}')
print(f'Fixtures : {fixtures.shape}')
""")

# ── Cell 3: Team name normalisation ──────────────────────────────────────────
md("""\
## 2 — Team name normalisation
`team_name` in player data uses slightly different spellings than
`home_team`/`away_team` in fixture data. Fix before joining.
""")

code("""\
TEAM_NAME_MAP = {
    'Brighton and Hove Albion': 'Brighton & Hove Albion',
    'Bournemouth'             : 'AFC Bournemouth',
}

for df in (outfield, gk):
    df['team_name'] = df['team_name'].replace(TEAM_NAME_MAP)

# Verify — collect all known team names from fixture columns, should print nothing
all_fixture_teams = pd.concat([
    outfield['home_team'], outfield['away_team']
]).unique()

for df, label in [(outfield, 'outfield'), (gk, 'gk')]:
    bad = df[~df['team_name'].isin(all_fixture_teams)]['team_name'].unique()
    print(f'{label} mismatches remaining: {bad.tolist()}')
""")

# ── Cell 4: Match context ─────────────────────────────────────────────────────
md("""\
## 3 — Add match context features
Adds per-player-per-match:
- `home_away` — 1 = home, 0 = away
- `opponent` — opposing team name
- `team_goals_scored` / `opp_goals_scored` — goals in the match (summed from player stats)
- `result` — W / D / L from the player's team perspective
""")

code("""\
# Build a match-level score lookup from outfield data (GK has no 'goals' column)
match_goals = (
    outfield.groupby(['match_id', 'team_name'])['goals']
    .sum()
    .reset_index(name='team_goals_scored')
)
opp_goals = match_goals.rename(columns={
    'team_name'        : 'opponent',
    'team_goals_scored': 'opp_goals_scored',
})

def add_match_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # home_away flag
    df['home_away'] = (df['team_name'] == df['home_team']).astype(int)

    # opponent
    df['opponent'] = np.where(df['home_away'] == 1, df['away_team'], df['home_team'])

    # merge pre-computed scores (works for both outfield and GK)
    df = df.merge(match_goals, on=['match_id', 'team_name'], how='left')
    df = df.merge(opp_goals,   on=['match_id', 'opponent'],  how='left')

    # result
    def result(row):
        if row['team_goals_scored'] > row['opp_goals_scored']:
            return 'W'
        elif row['team_goals_scored'] == row['opp_goals_scored']:
            return 'D'
        return 'L'

    df['result'] = df.apply(result, axis=1)
    return df


outfield = add_match_context(outfield)
gk       = add_match_context(gk)

print('Context columns added.')
print(outfield[['team_name', 'home_away', 'opponent',
                'team_goals_scored', 'opp_goals_scored', 'result']].head(8).to_string())
""")

# ── Cell 5: Result distribution check ────────────────────────────────────────
code("""\
# Sanity check: result distribution at match level (not player level)
match_results = (
    outfield.drop_duplicates(subset=['match_id', 'team_name'])
    [['season', 'result']]
)
print('Result distribution (match-team level):')
print(match_results['result'].value_counts(normalize=True).mul(100).round(1))
print()
print('Per season:')
print(match_results.groupby('season')['result'].value_counts(normalize=True).mul(100).round(1).unstack())
""")

# ── Cell 6: Quality filter ────────────────────────────────────────────────────
md("""\
## 4 — Quality filter
Remove players with fewer than 30 minutes played (insufficient data)
and any rows with a null `rating_title` (FotMob couldn't generate a rating).
""")

code("""\
def quality_filter(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n0 = len(df)
    df = df[df['minutes_played'] >= 30].copy()
    n1 = len(df)
    df = df[df['rating_title'].notna()].copy()
    n2 = len(df)
    print(f'{label}: {n0:,} → {n1:,} (min 30 min) → {n2:,} (has rating)  dropped {n0-n2:,} ({(n0-n2)/n0*100:.1f}%)')
    return df

outfield = quality_filter(outfield, 'outfield')
gk       = quality_filter(gk,       'gk')
""")

# ── Cell 7: Column audit ──────────────────────────────────────────────────────
md("""\
## 5 — Column audit
Drop columns that are too sparse to be useful and are not needed by the
composite score formulas. Threshold: **> 80% null** (event rarities like
missed penalties, own goals) and known non-informative columns.
""")

code("""\
# Columns required for composite scores or ratio features — never drop these
KEEP_ALWAYS = {
    # identifiers / context
    'match_id', 'round', 'match_date', 'home_team', 'away_team',
    'player_id', 'player_name', 'team_id', 'team_name', 'shirt_number',
    'position_id', 'is_goalkeeper', 'season',
    'home_away', 'opponent', 'team_goals_scored', 'opp_goals_scored', 'result',
    'minutes_played', 'rating_title',
    # ATT score inputs
    'goals', 'assists', 'expected_goals', 'expected_assists',
    'dribbles_succeeded', 'dribbles_succeeded_total',
    'chances_created',  'recoveries',
    # MID score inputs  (accurate_passes used as ProgPass proxy)
    'accurate_passes', 'accurate_passes_total',
    'interceptions',
    # DEF score inputs
    'aerials_won', 'aerials_won_total',
    'clearances', 'shot_blocks',
    'matchstats.headers.tackles',
    # ratio feature denominators
    'ground_duels_won', 'ground_duels_won_total',
    'long_balls_accurate', 'long_balls_accurate_total',
    'accurate_crosses', 'accurate_crosses_total',
    'ShotsOnTarget', 'ShotsOffTarget',
    'shot_accuracy', 'shot_accuracy_total',
    # extra context
    'touches', 'dispossessed', 'defensive_actions',
    'duel_won', 'duel_lost', 'fouls',
}

DROP_COLS = [
    # > 97% null — rare events
    'missed_penalty', 'owngoal', 'penalties_won', 'conceded_penalties',
    'clearance_off_the_line', 'errors_led_to_goal', 'last_man_tackle', 'shots_woodwork',
    # > 80% null — too sparse
    'Offsides', 'big_chance_missed_title', 'big_chance_created_team_title', 'corners',
    # > 70% null — not in any composite formula
    'blocked_shots', 'expected_goals_on_target_variant',
    # 40-50% null and not needed
    'headed_clearance', 'was_fouled', 'expected_goals_non_penalty',
    'xg_and_xa', 'passes_into_final_third', 'touches_opp_box',
]

def drop_sparse(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f'{label}: dropped {len(cols_to_drop)} columns → {df.shape[1]} remaining')
    return df

outfield = drop_sparse(outfield, 'outfield')
gk       = drop_sparse(gk,       'gk')
""")

# ── Cell 8: Null summary ──────────────────────────────────────────────────────
code("""\
print('Outfield — remaining null rates (>0%):')
null_pct = (outfield.isnull().sum() / len(outfield) * 100).sort_values(ascending=False)
print(null_pct[null_pct > 0].to_string())
""")

# ── Cell 9: Final shape summary ───────────────────────────────────────────────
md("## 6 — Final summary")

code("""\
print('='*55)
for df, label in [(outfield, 'outfield'), (gk, 'gk')]:
    print(f'  {label:<10} : {df.shape[0]:>6,} rows  x  {df.shape[1]:>2} cols')
    print(f'             seasons : {df[\"season\"].value_counts().to_dict()}')
    print()
print('='*55)
""")

# ── Cell 10: Export ───────────────────────────────────────────────────────────
md("## 7 — Export")

code("""\
outfield.to_parquet(OUT_DIR / 'outfield_clean.parquet', index=False)
gk.to_parquet(OUT_DIR / 'gk_clean.parquet', index=False)

for name in ['outfield_clean', 'gk_clean']:
    size = (OUT_DIR / f'{name}.parquet').stat().st_size / 1e6
    print(f'✓ {name}.parquet saved  ({size:.1f} MB)')
""")

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path(__file__).parent / '02_preprocessing.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'Written: {out}')
