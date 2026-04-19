"""
raw_cleaner.py
=============
Deletes all raw/ folders and their contents under data/

Usage:
    python raw_cleaner
.py
"""

import shutil
from pathlib import Path

DATA_ROOT = Path('data')

for league_dir in sorted(DATA_ROOT.iterdir()):
    for season_dir in sorted(league_dir.iterdir()):
        raw_dir = season_dir / 'raw'
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
            print(f'Deleted {raw_dir}')

print('\nDone.')