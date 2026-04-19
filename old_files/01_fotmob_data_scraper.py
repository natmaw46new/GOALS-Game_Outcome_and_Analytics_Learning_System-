"""
FotMob Multi-Season Scraper
============================
Automates scraping across multiple leagues and seasons.
Converts the notebook cells into a single callable function
and loops over every (league, season) combination you define.

Usage:
    python fotmob_scraper.py

Output structure:
    data/
      {league_id}/
        {season}/
          raw/          ← cached raw JSON per match (never re-fetched)
          output/
            fixtures.parquet
            player_stats.parquet
            outfield_players.parquet
            goalkeepers.parquet
"""

import hashlib
import hmac
import time
import random
import json
import asyncio
import logging
import requests
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm.auto import tqdm

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('fotmob')

# ══════════════════════════════════════════════════════════════
#  CONFIG — Edit this section only
# ══════════════════════════════════════════════════════════════

SCRAPE_TARGETS = [
    # (league_id, season_string)
    # Premier League
    (47, '2021/2022'),
    (47, '2022/2023'),
    (47, '2023/2024'),
    (47, '2024/2025'),   # test season — still scraped, split happens in preprocessing
    # Uncomment to add more leagues:
    # (87, '2021/2022'),  # La Liga
    # (87, '2022/2023'),
    # (87, '2023/2024'),
    # (87, '2024/2025'),
    # (54, '2021/2022'),  # Bundesliga
]

BASE_URL = 'https://www.fotmob.com/api/data'  # updated: /api/data/ prefix as of 2025

MAX_CONCURRENT = 4
JITTER_MIN     = 1.0
JITTER_MAX     = 3.0
MAX_RETRIES    = 5
BACKOFF_BASE   = 2

# ── Category + GK maps (unchanged from your notebook) ─────────
CATEGORY_MAP = {
    'Top stats'       : 'top_stats',
    'Attack'          : 'attack',
    'Defence'         : 'defense',
    'Defense'         : 'defense',
    'Passes'          : 'passes',
    'Duels'           : 'duels',
    'Goalkeeping'     : 'goalkeeping',
    'Physical metrics': 'physical_metrics',
}

GK_GOALKEEPING = {
    'saves', 'goals_conceded', 'xgot_faced', 'goals_prevented',
    'diving_save', 'saves_inside_box', 'acted_as_sweeper',
    'punches', 'throws', 'high_claim',
}
GK_PASSING = {'accurate_passes', 'accurate_long_balls', 'passes_into_final_third', 'accurate_crosses'}
GK_DEFENSE = {'recoveries', 'clearances', 'defensive_actions', 'tackles', 'interceptions', 'shot_blocks'}
GK_DUELS   = {'ground_duels_won', 'aerials_won', 'was_fouled', 'fouls_committed', 'aerial_duels_won'}

EXCLUDE_FROM_WIDE = {
    'physical_metrics_distance_covered', 'physical_metrics_topspeed',
    'physical_metrics_running', 'physical_metrics_sprinting',
    'physical_metrics_walking', 'physical_metrics_number_of_sprints',
    'fantasy_points', 'Shotmap',
}

ID_COLS = [
    'match_id', 'round', 'match_date',
    'home_team', 'away_team',
    'player_id', 'player_name', 'team_id', 'team_name',
    'shirt_number', 'position_id', 'is_goalkeeper',
]

PAGE_HEADERS = {
    'User-Agent'     : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept'         : 'text/html,application/xhtml+xml,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Referer'        : 'https://www.fotmob.com/',
}


# ══════════════════════════════════════════════════════════════
#  AUTH + CLIENT
#  FotMob replaced the old x-mas HMAC header with X-Fm-Req:
#  a Base64-encoded JSON token signed with MD5 + secret lyrics.
# ══════════════════════════════════════════════════════════════

# The secret is the opening verse of Never Gonna Give You Up.
# Line endings must be \n (not \r\n) and no trailing newline.
_FM_SECRET = (
    "We're no strangers to love\n"
    "You know the rules and so do I\n"
    "A full commitment's what I'm thinking of\n"
    "You wouldn't get this from any other guy\n"
    "I just wanna tell you how I'm feeling\n"
    "Gotta make you understand\n"
    "Never gonna give you up\n"
    "Never gonna let you down\n"
    "Never gonna run around and desert you\n"
    "Never gonna make you cry\n"
    "Never gonna say goodbye\n"
    "Never gonna tell a lie and hurt you"
)

def _make_xfmreq_token(api_path: str, params: dict = None) -> str:
    """
    Build the X-Fm-Req header value FotMob now requires.

    Steps:
      1. Build the full URL path including query string
      2. Create body = {"url": path, "code": epoch_ms}
      3. signature = MD5(JSON.stringify(body) + secret_lyrics).upper()
      4. token = base64(JSON.stringify({"body": body, "signature": sig}))
    """
    import json, hashlib, base64, urllib.parse

    # Reconstruct the full path with query string (mirrors what JS does)
    if params:
        qs   = urllib.parse.urlencode(params)
        path = f"{api_path}?{qs}"
    else:
        path = api_path

    code = int(time.time() * 1000)
    body = {"url": path, "code": code}

    # MD5(stringify(body) + lyrics) — no spaces, no trailing newline
    raw       = json.dumps(body, separators=(',', ':')) + _FM_SECRET
    signature = hashlib.md5(raw.encode('utf-8')).hexdigest().upper()

    token_obj = {"body": body, "signature": signature}
    return base64.b64encode(
        json.dumps(token_obj, separators=(',', ':')).encode('utf-8')
    ).decode('utf-8')


def make_headers(path: str, params: dict = None) -> dict:
    return {
        'X-Fm-Req'       : _make_xfmreq_token(path, params),
        'User-Agent'     : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept'         : 'application/json, text/plain, */*',
        'Accept-Encoding': 'gzip, deflate',
        'Referer'        : 'https://www.fotmob.com/',
        'Origin'         : 'https://www.fotmob.com',
    }


class FotMobClient:
    def __init__(self):
        self._session   = None
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(ssl=True, limit=MAX_CONCURRENT),
        )
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    async def get(self, path: str, params: dict = None) -> dict:
        url = f'{BASE_URL}{path}'
        for attempt in range(1, MAX_RETRIES + 1):
            async with self._semaphore:
                await asyncio.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
                try:
                    async with self._session.get(
                        url, params=params, headers=make_headers(path, params)
                    ) as resp:
                        if resp.status == 200:
                            return await resp.json(content_type=None)
                        elif resp.status in (429, 503):
                            await asyncio.sleep(BACKOFF_BASE ** attempt + random.random())
                        elif resp.status >= 500:
                            await asyncio.sleep(BACKOFF_BASE ** attempt)
                        else:
                            raise ValueError(f'HTTP {resp.status} on {url}')
                except aiohttp.ClientConnectorError:
                    await asyncio.sleep(BACKOFF_BASE ** attempt)
        raise RuntimeError(f'Exhausted {MAX_RETRIES} retries for {url}')


# ══════════════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════════════

async def get_season_id(client: FotMobClient, league_id: int, season_label: str) -> str | None:
    """
    FotMob identifies past seasons by a numeric ID, not a year string.
    This hits the league endpoint without a season param to get the full
    season list, then finds the ID whose name matches season_label.

    season_label examples: '2021/2022', '2022/2023', '2023/2024'
    """
    data = await client.get('/leagues', params={'id': league_id, 'ccode3': 'USA_MA'})

    # Try every known key that might hold the season list
    seasons = (
        data.get('allAvailableSeasons')
        or data.get('availableSeasons')
        or data.get('stats', {}).get('seasonStatLinks')
        or []
    )

    # Log raw structure so we can see exactly what FotMob returns
    log.info('Season list sample (first 5): %s', seasons[:5])

    for s in seasons:
        # Handle both plain strings ('2021/2022') and dicts ({'id':..., 'year':...})
        if isinstance(s, str):
            name = s
            sid  = s
        elif isinstance(s, dict):
            name = s.get('year') or s.get('name') or s.get('label') or s.get('season') or ''
            sid  = s.get('id') or s.get('seasonId') or s.get('season') or name
        else:
            continue

        if season_label in name or name in season_label:
            log.info('Found season ID %s for "%s"', sid, season_label)
            return str(sid)

    log.warning('Could not find season ID for "%s" — raw list: %s', season_label, seasons[:8])
    return None


async def fetch_all_fixtures(league_id: int, season: str, raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    async with FotMobClient() as client:
        # Step 1: resolve the numeric season ID for this year label
        season_id = await get_season_id(client, league_id, season)

        params = {'id': league_id, 'ccode3': 'USA_MA'}
        if season_id:
            params['season'] = season_id   # numeric ID for past seasons
        else:
            params['season'] = season       # fallback — works for current season

        data = await client.get('/leagues', params=params)

    with open(raw_dir / 'league_meta.json', 'w') as f:
        json.dump(data, f, indent=2)

    all_matches = data.get('fixtures', {}).get('allMatches', [])
    if not all_matches:
        log.warning('No matches found for league=%s season=%s', league_id, season)
        return pd.DataFrame()

    rows = []
    for m in all_matches:
        rows.append({
            'match_id'  : m.get('id'),
            'round'     : m.get('round'),
            'page_url'  : m.get('pageUrl', ''),
            'match_date': m.get('status', {}).get('utcTime', ''),
            'finished'  : m.get('status', {}).get('finished', False),
            'home_team' : m.get('home', {}).get('name', ''),
            'home_id'   : m.get('home', {}).get('id'),
            'away_team' : m.get('away', {}).get('name', ''),
            'away_id'   : m.get('away', {}).get('id'),
        })

    df = pd.DataFrame(rows).dropna(subset=['match_id'])
    df['match_id'] = df['match_id'].astype(int)
    df = df.sort_values(['round', 'match_date']).reset_index(drop=True)
    df.to_parquet(out_dir / 'fixtures.parquet', index=False)
    df.to_csv(out_dir / 'fixtures.csv', index=False)
    log.info('Fixtures: %d matches across %d rounds', len(df), df['round'].nunique())
    return df


# ══════════════════════════════════════════════════════════════
#  PARSER (unchanged from notebook)
# ══════════════════════════════════════════════════════════════

def fetch_match_page(page_url: str) -> dict | None:
    full_url = f'https://www.fotmob.com{page_url}'
    try:
        resp = requests.get(full_url, headers=PAGE_HEADERS, timeout=20)
        if resp.status_code != 200:
            log.warning('HTTP %s for %s', resp.status_code, full_url)
            return None
        tag = BeautifulSoup(resp.text, 'html.parser').find('script', {'id': '__NEXT_DATA__'})
        if not tag:
            log.warning('No __NEXT_DATA__ at %s', full_url)
            return None
        return json.loads(tag.string)
    except Exception as exc:
        log.warning('Error fetching %s: %s', full_url, exc)
        return None


def parse_match_page(next_data: dict, match_row: pd.Series) -> list[dict]:
    rows         = []
    page_props   = next_data.get('props', {}).get('pageProps', {})
    content      = page_props.get('content', {})
    player_stats = content.get('playerStats', {})

    if not player_stats:
        return rows

    match_ctx = {
        'match_id'  : match_row['match_id'],
        'round'     : match_row['round'],
        'match_date': match_row['match_date'],
        'home_team' : match_row['home_team'],
        'away_team' : match_row['away_team'],
    }

    for pid_str, pdata in player_stats.items():
        if not isinstance(pdata, dict):
            continue

        is_gk = pdata.get('isGoalkeeper', False)
        player_ctx = {
            'player_id'    : pdata.get('id', pid_str),
            'player_name'  : pdata.get('name', ''),
            'team_id'      : pdata.get('teamId'),
            'team_name'    : pdata.get('teamName', ''),
            'is_goalkeeper': is_gk,
            'shirt_number' : pdata.get('shirtNumber'),
            'position_id'  : pdata.get('positionId'),
        }

        for block in pdata.get('stats', []):
            if not isinstance(block, dict):
                continue

            raw_title  = block.get('title', '')
            block_key  = block.get('key', '')
            stats_dict = block.get('stats', {})

            if not isinstance(stats_dict, dict):
                continue

            for metric_name, metric_data in stats_dict.items():
                if not isinstance(metric_data, dict):
                    continue

                metric_key = metric_data.get('key', '')
                stat       = metric_data.get('stat', {})
                value      = stat.get('value')
                total      = stat.get('total')
                stat_type  = stat.get('type', '')

                if is_gk and block_key == 'top_stats':
                    if metric_key in GK_GOALKEEPING:
                        category = 'goalkeeping'
                    elif metric_key in GK_PASSING:
                        category = 'passes'
                    elif metric_key in GK_DEFENSE:
                        category = 'defense'
                    elif metric_key in GK_DUELS:
                        category = 'duels'
                    else:
                        category = 'top_stats'
                else:
                    category = CATEGORY_MAP.get(raw_title, block_key)

                rows.append({
                    **match_ctx,
                    **player_ctx,
                    'category'  : category,
                    'metric'    : metric_name,
                    'metric_key': metric_key,
                    'value'     : value,
                    'total'     : total,
                    'stat_type' : stat_type,
                })

    return rows


# ══════════════════════════════════════════════════════════════
#  SCRAPE + WIDE BUILD (unchanged logic, now takes dirs as args)
# ══════════════════════════════════════════════════════════════

def scrape_full_season(fixtures_df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    finished = fixtures_df[fixtures_df['finished']].copy().reset_index(drop=True)
    all_rows = []
    failed   = []

    already_cached = sum(
        1 for _, r in finished.iterrows()
        if (raw_dir / (str(r['match_id']) + '.json')).exists()
    )
    log.info('Finished: %d | Cached: %d | New: %d',
             len(finished), already_cached, len(finished) - already_cached)

    for _, match_row in tqdm(finished.iterrows(), total=len(finished), desc='Scraping', unit='match'):
        match_id = match_row['match_id']
        raw_path = raw_dir / f'{match_id}.json'

        if raw_path.exists():
            with open(raw_path) as f:
                next_data = json.load(f)
        else:
            time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
            next_data = fetch_match_page(match_row['page_url'])
            if next_data is None:
                failed.append(match_id)
                continue
            with open(raw_path, 'w') as f:
                json.dump(next_data, f)

        content      = next_data.get('props', {}).get('pageProps', {}).get('content', {})
        player_stats = content.get('playerStats')
        if not player_stats:
            continue

        rows = parse_match_page(next_data, match_row)
        if rows:
            all_rows.extend(rows)
        else:
            failed.append(match_id)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df['match_id']   = df['match_id'].astype(int)
        df['round']      = df['round'].astype(int)
        df['value']      = pd.to_numeric(df['value'], errors='coerce')
        df['total']      = pd.to_numeric(df['total'], errors='coerce')
        df['match_date'] = pd.to_datetime(df['match_date'], utc=True, errors='coerce')

    log.info('Rows: %d | Matches: %d | Players: %d | Failed: %d',
             len(df), df['match_id'].nunique() if not df.empty else 0,
             df['player_id'].nunique() if not df.empty else 0, len(failed))
    return df


def build_wide(df: pd.DataFrame, is_gk: bool, min_matches: int = 3, min_pct: float = 0.0) -> pd.DataFrame:
    subset = df[
        (df['is_goalkeeper'] == is_gk) &
        (~df['metric_key'].isin(EXCLUDE_FROM_WIDE)) &
        (df['metric_key'].notna()) &
        (df['metric_key'] != '')
    ].copy()

    metric_stats = (
        subset.groupby('metric_key')
        .agg(n_matches=('match_id', 'nunique'), pct_valid=('value', lambda x: x.notna().mean()), category=('category', 'first'))
        .reset_index()
    )
    included = metric_stats[
        (metric_stats['n_matches'] >= min_matches) &
        (metric_stats['pct_valid'] >= min_pct)
    ]
    cat_order = ['top_stats', 'attack', 'passes', 'defense', 'duels', 'goalkeeping']
    included = included.copy()
    included['cat_order'] = included['category'].map({c: i for i, c in enumerate(cat_order)}).fillna(99)
    included = included.sort_values(['cat_order', 'n_matches'], ascending=[True, False])
    valid_keys = included['metric_key'].tolist()

    subset = subset[subset['metric_key'].isin(valid_keys)]
    value_wide = subset.pivot_table(index=ID_COLS, columns='metric_key', values='value', aggfunc='first').reset_index()
    value_wide.columns.name = None

    frac = subset[subset['stat_type'] == 'fractionWithPercentage']
    if not frac.empty:
        total_wide = frac.pivot_table(index=ID_COLS, columns='metric_key', values='total', aggfunc='first').reset_index()
        total_wide.columns.name = None
        rename_map = {c: f'{c}_total' for c in total_wide.columns if c not in ID_COLS}
        total_wide = total_wide.rename(columns=rename_map)
        wide = value_wide.merge(total_wide[ID_COLS + list(rename_map.values())], on=ID_COLS, how='left')
    else:
        wide = value_wide

    val_cols   = [c for c in valid_keys if c in wide.columns]
    total_cols = [c for c in wide.columns if c.endswith('_total')]
    ordered    = ID_COLS + val_cols + total_cols
    ordered    = [c for c in ordered if c in wide.columns]
    return wide[ordered].sort_values(['round', 'match_id', 'player_id']).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  MAIN LOOP — runs every (league, season) combo
# ══════════════════════════════════════════════════════════════

def run_one(league_id: int, season: str):
    """Scrape and export one (league, season) combination."""
    label   = f'League {league_id} | {season}'
    raw_dir = Path('data') / str(league_id) / season.replace('/', '_') / 'raw'
    out_dir = Path('data') / str(league_id) / season.replace('/', '_') / 'output'
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info('━━━ START %s ━━━', label)

    # 1. Fixtures
    done_marker = out_dir / 'fixtures.parquet'
    if done_marker.exists():
        fixtures_df = pd.read_parquet(done_marker)
        log.info('Fixtures already fetched (%d rows), skipping API call', len(fixtures_df))
    else:
        try:
            fixtures_df = asyncio.run(fetch_all_fixtures(league_id, season, raw_dir, out_dir))
        except RuntimeError:
            # Jupyter / already-running loop fallback
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            fixtures_df = loop.run_until_complete(fetch_all_fixtures(league_id, season, raw_dir, out_dir))

    if fixtures_df.empty:
        log.warning('No fixtures — skipping %s', label)
        return

    # 2. Scrape match pages
    stats_df = scrape_full_season(fixtures_df, raw_dir)

    log.info('stats_df shape: %s | empty: %s', stats_df.shape, stats_df.empty)

    if stats_df.empty:
        log.warning('No stats returned — checking one raw JSON to diagnose...')
        # Print the first cached raw file so we can inspect its structure
        raw_files = list(raw_dir.glob('*.json'))
        if raw_files:
            import json as _json
            with open(raw_files[0]) as _f:
                _sample = _json.load(_f)
            # Show top-level keys only
            props = _sample.get('props', {}).get('pageProps', {}).get('content', {})
            log.warning('First raw JSON top-level keys: %s', list(_sample.keys()))
            log.warning('pageProps.content keys: %s', list(props.keys()))
        log.warning('Skipping %s', label)
        return

    # 3. Save raw long-format
    try:
        stats_df.to_parquet(out_dir / 'player_stats.parquet', index=False, engine='pyarrow')
        stats_df.to_csv(out_dir / 'player_stats.csv', index=False)
        log.info('Saved player_stats — %d rows', len(stats_df))
    except Exception as e:
        log.error('Failed saving player_stats: %s', e)
        return

    # 4. Build and save wide-format
    try:
        outfield_df = build_wide(stats_df, is_gk=False)
        gk_df       = build_wide(stats_df, is_gk=True)
        outfield_df.to_parquet(out_dir / 'outfield_players.parquet', index=False, engine='pyarrow')
        outfield_df.to_csv(out_dir / 'outfield_players.csv', index=False)
        gk_df.to_parquet(out_dir / 'goalkeepers.parquet', index=False, engine='pyarrow')
        gk_df.to_csv(out_dir / 'goalkeepers.csv', index=False)
        log.info('Saved wide-format — outfield: %d rows, GK: %d rows', len(outfield_df), len(gk_df))
    except Exception as e:
        log.error('Failed building/saving wide format: %s', e)
        return

    log.info('━━━ DONE  %s — outfield: %d rows, GK: %d rows ━━━',
             label, len(outfield_df), len(gk_df))


if __name__ == '__main__':
    print(f'Scraping {len(SCRAPE_TARGETS)} (league, season) combinations...\n')
    for i, (league_id, season) in enumerate(SCRAPE_TARGETS, 1):
        print(f'\n[{i}/{len(SCRAPE_TARGETS)}] League {league_id} — {season}')
        run_one(league_id, season)
    print('\n✓ All done.')