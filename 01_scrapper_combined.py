"""
FotMob Multi-Season Scraper
============================
Scraping strategy (two-pass):
  Pass 1 — __NEXT_DATA__ HTML parsing (original method, fast, async)
  Pass 2 — /api/data/matchDetails API (fallback for auth-blocked matches,
            requires browser cookie, synchronous)

The API fallback is only attempted for matches where Pass 1 returns
empty playerStats. If BROWSER_COOKIE is left empty, Pass 2 is skipped
and missing matches are logged as permanently unavailable.

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
import urllib.parse
import base64
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
    (47, '2021/2022'),
    (47, '2022/2023'),
    (47, '2023/2024'),
    (47, '2024/2025'),
]

BASE_URL       = 'https://www.fotmob.com'
API_BASE       = 'https://www.fotmob.com/api/data'

MAX_CONCURRENT = 4
JITTER_MIN     = 1.0
JITTER_MAX     = 3.0
MAX_RETRIES    = 5
BACKOFF_BASE   = 2

# ── API fallback settings ──────────────────────────────────────
# Jitter for the synchronous API fallback pass
API_JITTER_MIN = 3.0
API_JITTER_MAX = 6.0

# Client version hash from x-mas token — update if API starts 403ing
# How to refresh: DevTools → Network → any matchDetails request
# → Request Headers → x-mas → decode base64 → read 'foo' field
FM_CLIENT_VERSION = "production:dda5f4d07deb53ec94eb7009cbab58a6149c4ac3"

# Browser cookie for Cloudflare Turnstile bypass
# How to get: DevTools → Network → matchDetails request
# → Request Headers → copy full 'cookie' value
# Leave empty string to skip API fallback entirely
# Token typically lasts 24-48 hours — refresh if you see 403s
BROWSER_COOKIE = ""  # ← paste your cookie here

# ── Category + GK maps ────────────────────────────────────────
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
    'User-Agent'     : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36',
    'Accept'         : 'text/html,application/xhtml+xml,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Referer'        : 'https://www.fotmob.com/',
}


# ══════════════════════════════════════════════════════════════
#  AUTH — x-mas token (replaces old X-Fm-Req)
# ══════════════════════════════════════════════════════════════

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


def _make_xmas_token(api_path: str, params: dict = None) -> str:
    """
    Build the x-mas header token FotMob requires.
    Must be regenerated fresh for every single request.
    Includes 'foo' (client version) field required by current API.
    """
    if params:
        qs   = urllib.parse.urlencode(params)
        path = f"{api_path}?{qs}"
    else:
        path = api_path

    code      = int(time.time() * 1000)
    body      = {"url": path, "code": code, "foo": FM_CLIENT_VERSION}
    raw       = json.dumps(body, separators=(',', ':')) + _FM_SECRET
    signature = hashlib.md5(raw.encode('utf-8')).hexdigest().upper()
    token_obj = {"body": body, "signature": signature}
    return base64.b64encode(
        json.dumps(token_obj, separators=(',', ':')).encode('utf-8')
    ).decode('utf-8')


def make_headers(path: str, params: dict = None) -> dict:
    """Headers for async fixture/league API calls."""
    return {
        'x-mas'          : _make_xmas_token(path, params),
        'User-Agent'     : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36',
        'Accept'         : 'application/json, text/plain, */*',
        'Accept-Encoding': 'gzip, deflate',
        'Referer'        : 'https://www.fotmob.com/',
        'Origin'         : 'https://www.fotmob.com',
    }


def make_api_fallback_headers(path: str, params: dict = None) -> dict:
    """
    Headers for synchronous API fallback calls.
    Includes browser cookie for Cloudflare Turnstile bypass.
    """
    headers = {
        'x-mas'          : _make_xmas_token(path, params),
        'User-Agent'     : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36',
        'Accept'         : '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer'        : 'https://www.fotmob.com/',
        'sec-fetch-dest' : 'empty',
        'sec-fetch-mode' : 'cors',
        'sec-fetch-site' : 'same-origin',
    }
    if BROWSER_COOKIE:
        headers['cookie'] = BROWSER_COOKIE
    return headers


# ══════════════════════════════════════════════════════════════
#  ASYNC CLIENT (Pass 1 — fixtures + HTML scraping)
# ══════════════════════════════════════════════════════════════

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
        url = f'{API_BASE}{path}'
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
    data    = await client.get('/leagues', params={'id': league_id, 'ccode3': 'USA_MA'})
    seasons = (
        data.get('allAvailableSeasons')
        or data.get('availableSeasons')
        or data.get('stats', {}).get('seasonStatLinks')
        or []
    )
    log.info('Season list sample (first 5): %s', seasons[:5])
    for s in seasons:
        if isinstance(s, str):
            name = s; sid = s
        elif isinstance(s, dict):
            name = s.get('year') or s.get('name') or s.get('label') or s.get('season') or ''
            sid  = s.get('id') or s.get('seasonId') or s.get('season') or name
        else:
            continue
        if season_label in name or name in season_label:
            log.info('Found season ID %s for "%s"', sid, season_label)
            return str(sid)
    log.warning('Could not find season ID for "%s"', season_label)
    return None


async def fetch_all_fixtures(league_id: int, season: str, raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    async with FotMobClient() as client:
        season_id = await get_season_id(client, league_id, season)
        params    = {'id': league_id, 'ccode3': 'USA_MA'}
        if season_id:
            params['season'] = season_id
        else:
            params['season'] = season
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
#  PASS 1 — __NEXT_DATA__ HTML PARSER
# ══════════════════════════════════════════════════════════════

def fetch_match_page(page_url: str) -> dict | None:
    """Fetch match page and extract __NEXT_DATA__ JSON."""
    full_url = f'{BASE_URL}{page_url}'
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
    """Parse __NEXT_DATA__ response into long-format rows."""
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
                    if metric_key in GK_GOALKEEPING:   category = 'goalkeeping'
                    elif metric_key in GK_PASSING:     category = 'passes'
                    elif metric_key in GK_DEFENSE:     category = 'defense'
                    elif metric_key in GK_DUELS:       category = 'duels'
                    else:                              category = 'top_stats'
                else:
                    category = CATEGORY_MAP.get(raw_title, block_key)

                rows.append({
                    **match_ctx, **player_ctx,
                    'category'  : category,
                    'metric'    : metric_name,
                    'metric_key': metric_key,
                    'value'     : value,
                    'total'     : total,
                    'stat_type' : stat_type,
                })

    return rows


# ══════════════════════════════════════════════════════════════
#  PASS 2 — /api/data/matchDetails API FALLBACK
# ══════════════════════════════════════════════════════════════

def fetch_match_api(match_id: int) -> tuple:
    """
    Fallback: fetch player stats via /api/data/matchDetails.
    Returns (data, status) where status is:
      'ok'           — playerStats found and non-empty
      'auth_blocked' — response ok but playerStats empty/None
      'failed'       — HTTP error or exception
    """
    path   = '/api/data/matchDetails'
    params = {'matchId': match_id}
    url    = f'{BASE_URL}{path}'

    try:
        r = requests.get(
            url,
            params=params,
            headers=make_api_fallback_headers(path, params),
            timeout=20
        )
        if r.status_code != 200:
            return None, f'failed_http_{r.status_code}'

        data    = r.json()
        content = data.get('content', {})
        ps      = content.get('playerStats')

        if ps and len(ps) > 0:
            return data, 'ok'
        else:
            return data, 'auth_blocked'

    except Exception as e:
        return None, f'failed_exception_{str(e)[:60]}'


def parse_match_api(data: dict) -> list[dict]:
    """
    Parse /api/data/matchDetails response into long-format rows.
    Produces identical schema to parse_match_page so build_wide()
    works completely unchanged.
    """
    rows    = []
    general = data.get('general', {})
    content = data.get('content', {})
    ps      = content.get('playerStats', {})

    if not ps:
        return rows

    match_ctx = {
        'match_id'  : int(general.get('matchId', 0)),
        'round'     : str(general.get('leagueRoundName', '')),
        'match_date': general.get('matchTimeUTCDate', ''),
        'home_team' : general.get('homeTeam', {}).get('name', ''),
        'away_team' : general.get('awayTeam', {}).get('name', ''),
    }

    for pid_str, pdata in ps.items():
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
                    if metric_key in GK_GOALKEEPING:   category = 'goalkeeping'
                    elif metric_key in GK_PASSING:     category = 'passes'
                    elif metric_key in GK_DEFENSE:     category = 'defense'
                    elif metric_key in GK_DUELS:       category = 'duels'
                    else:                              category = 'top_stats'
                else:
                    category = CATEGORY_MAP.get(raw_title, block_key)

                rows.append({
                    **match_ctx, **player_ctx,
                    'category'  : category,
                    'metric'    : metric_name,
                    'metric_key': metric_key,
                    'value'     : value,
                    'total'     : total,
                    'stat_type' : stat_type,
                })

    return rows


# ══════════════════════════════════════════════════════════════
#  TWO-PASS SCRAPER
# ══════════════════════════════════════════════════════════════

def scrape_full_season(fixtures_df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """
    Two-pass scraper:
      Pass 1 — __NEXT_DATA__ HTML parsing (fast, async-style sequential)
      Pass 2 — /api/data/matchDetails API fallback (slower, needs cookie)

    Pass 2 only runs if BROWSER_COOKIE is set and Pass 1 left
    matches with empty playerStats.
    """
    finished = fixtures_df[fixtures_df['finished']].copy().reset_index(drop=True)
    all_rows = []
    failed   = []

    # ── Track per-match outcome ───────────────────────────────
    pass1_ok         = []   # match IDs successfully parsed via Pass 1
    pass1_empty      = []   # match IDs where __NEXT_DATA__ had no playerStats
    pass2_ok         = []   # match IDs recovered via Pass 2
    pass2_blocked    = []   # match IDs still empty after Pass 2
    pass2_failed     = []   # match IDs that errored in Pass 2

    already_cached = sum(
        1 for _, r in finished.iterrows()
        if (raw_dir / f'{int(r["match_id"])}.json').exists()
    )
    log.info('Finished: %d | Cached: %d | New: %d',
             len(finished), already_cached, len(finished) - already_cached)

    # ══════════════════════════════════════════════════════════
    #  PASS 1 — __NEXT_DATA__ HTML
    # ══════════════════════════════════════════════════════════
    log.info('─── PASS 1: __NEXT_DATA__ HTML scraping ───')

    for _, match_row in tqdm(finished.iterrows(), total=len(finished),
                             desc='Pass 1', unit='match'):
        match_id = int(match_row['match_id'])
        raw_path = raw_dir / f'{match_id}.json'

        # Load from cache or fetch fresh
        if raw_path.exists():
            with open(raw_path) as f:
                next_data = json.load(f)
        else:
            time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
            next_data = fetch_match_page(match_row['page_url'])
            if next_data is None:
                failed.append(match_id)
                pass1_empty.append(match_id)
                continue
            with open(raw_path, 'w') as f:
                json.dump(next_data, f)

        # Check if this is an __NEXT_DATA__ response or an API response
        # (handles re-runs where some matches were already recovered via Pass 2)
        if 'props' in next_data:
            # __NEXT_DATA__ format
            content      = next_data.get('props', {}).get('pageProps', {}).get('content', {})
            player_stats = content.get('playerStats')
            if not player_stats:
                pass1_empty.append(match_id)
                continue
            rows = parse_match_page(next_data, match_row)
        elif 'content' in next_data:
            # Already an API response cached from a previous Pass 2 run
            player_stats = next_data.get('content', {}).get('playerStats')
            if not player_stats:
                pass1_empty.append(match_id)
                continue
            rows = parse_match_api(next_data)
        else:
            pass1_empty.append(match_id)
            continue

        if rows:
            all_rows.extend(rows)
            pass1_ok.append(match_id)
        else:
            pass1_empty.append(match_id)

    log.info('Pass 1 complete — ok: %d | empty: %d | fetch_failed: %d',
             len(pass1_ok), len(pass1_empty), len(failed))

    # ══════════════════════════════════════════════════════════
    #  PASS 2 — /api/data/matchDetails API FALLBACK
    # ══════════════════════════════════════════════════════════
    if not pass1_empty:
        log.info('Pass 1 recovered all matches — skipping Pass 2')

    elif not BROWSER_COOKIE:
        log.warning(
            'Pass 2 skipped — %d matches have no playerStats but '
            'BROWSER_COOKIE is not set. Set it in CONFIG to recover these.',
            len(pass1_empty)
        )

    else:
        log.info('─── PASS 2: API fallback for %d matches ───', len(pass1_empty))
        log.info('Using /api/data/matchDetails with browser cookie')

        # Build lookup for match metadata
        match_lookup = finished.set_index('match_id')

        for match_id in tqdm(pass1_empty, desc='Pass 2', unit='match'):
            raw_path = raw_dir / f'api_{match_id}.json'

            # Skip if already recovered in a previous Pass 2 run
            if raw_path.exists():
                with open(raw_path) as f:
                    cached = json.load(f)
                ps = cached.get('content', {}).get('playerStats', {})
                if ps and len(ps) > 0:
                    rows = parse_match_api(cached)
                    if rows:
                        all_rows.extend(rows)
                        pass2_ok.append(match_id)
                        continue

            time.sleep(random.uniform(API_JITTER_MIN, API_JITTER_MAX))

            data, status = fetch_match_api(match_id)

            if status == 'ok':
                # Save under api_ prefix to distinguish from __NEXT_DATA__ cache
                with open(raw_path, 'w') as f:
                    json.dump(data, f)
                rows = parse_match_api(data)
                if rows:
                    all_rows.extend(rows)
                    pass2_ok.append(match_id)
                else:
                    pass2_blocked.append(match_id)

            elif status == 'auth_blocked':
                pass2_blocked.append(match_id)

            else:
                pass2_failed.append(match_id)
                log.warning('Pass 2 failed for %d: %s', match_id, status)

        log.info('Pass 2 complete — recovered: %d | blocked: %d | failed: %d',
                 len(pass2_ok), len(pass2_blocked), len(pass2_failed))

    # ── Final stats ───────────────────────────────────────────
    total_ok      = len(pass1_ok) + len(pass2_ok)
    total_missing = len(pass1_empty) - len(pass2_ok) - len(pass2_blocked) if BROWSER_COOKIE else len(pass1_empty)
    total_blocked = len(pass2_blocked) if BROWSER_COOKIE else 0

    log.info(
        'Season total — matches ok: %d | auth_blocked: %d | missing: %d',
        total_ok, total_blocked, len(pass1_empty) - len(pass2_ok)
    )

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df['match_id']   = df['match_id'].astype(int)
        df['round']      = df['round'].astype(str)
        df['value']      = pd.to_numeric(df['value'], errors='coerce')
        df['total']      = pd.to_numeric(df['total'], errors='coerce')
        df['match_date'] = pd.to_datetime(df['match_date'], utc=True, errors='coerce')

    log.info('Rows: %d | Matches: %d | Players: %d',
             len(df),
             df['match_id'].nunique() if not df.empty else 0,
             df['player_id'].nunique() if not df.empty else 0)

    return df


# ══════════════════════════════════════════════════════════════
#  WIDE FORMAT BUILDER (unchanged)
# ══════════════════════════════════════════════════════════════

def build_wide(df: pd.DataFrame, is_gk: bool,
               min_matches: int = 3, min_pct: float = 0.0) -> pd.DataFrame:
    subset = df[
        (df['is_goalkeeper'] == is_gk) &
        (~df['metric_key'].isin(EXCLUDE_FROM_WIDE)) &
        (df['metric_key'].notna()) &
        (df['metric_key'] != '')
    ].copy()

    metric_stats = (
        subset.groupby('metric_key')
        .agg(
            n_matches  =('match_id',  'nunique'),
            pct_valid  =('value',     lambda x: x.notna().mean()),
            category   =('category',  'first')
        )
        .reset_index()
    )
    included = metric_stats[
        (metric_stats['n_matches'] >= min_matches) &
        (metric_stats['pct_valid'] >= min_pct)
    ].copy()

    cat_order = ['top_stats', 'attack', 'passes', 'defense', 'duels', 'goalkeeping']
    included['cat_order'] = included['category'].map(
        {c: i for i, c in enumerate(cat_order)}
    ).fillna(99)
    included  = included.sort_values(['cat_order', 'n_matches'], ascending=[True, False])
    valid_keys = included['metric_key'].tolist()

    subset     = subset[subset['metric_key'].isin(valid_keys)]
    value_wide = subset.pivot_table(
        index=ID_COLS, columns='metric_key', values='value', aggfunc='first'
    ).reset_index()
    value_wide.columns.name = None

    frac = subset[subset['stat_type'] == 'fractionWithPercentage']
    if not frac.empty:
        total_wide = frac.pivot_table(
            index=ID_COLS, columns='metric_key', values='total', aggfunc='first'
        ).reset_index()
        total_wide.columns.name = None
        rename_map = {c: f'{c}_total' for c in total_wide.columns if c not in ID_COLS}
        total_wide = total_wide.rename(columns=rename_map)
        wide = value_wide.merge(
            total_wide[ID_COLS + list(rename_map.values())], on=ID_COLS, how='left'
        )
    else:
        wide = value_wide

    val_cols   = [c for c in valid_keys   if c in wide.columns]
    total_cols = [c for c in wide.columns if c.endswith('_total')]
    ordered    = ID_COLS + val_cols + total_cols
    ordered    = [c for c in ordered if c in wide.columns]
    return wide[ordered].sort_values(
        ['round', 'match_id', 'player_id']
    ).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
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
    fix_path = out_dir / 'fixtures.parquet'
    if fix_path.exists():
        fixtures_df = pd.read_parquet(fix_path)
        log.info('Fixtures already fetched (%d rows), skipping API call', len(fixtures_df))
    else:
        try:
            fixtures_df = asyncio.run(
                fetch_all_fixtures(league_id, season, raw_dir, out_dir)
            )
        except RuntimeError:
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            fixtures_df = loop.run_until_complete(
                fetch_all_fixtures(league_id, season, raw_dir, out_dir)
            )

    if fixtures_df.empty:
        log.warning('No fixtures — skipping %s', label)
        return

    # 2. Two-pass scrape
    stats_df = scrape_full_season(fixtures_df, raw_dir)

    if stats_df.empty:
        log.warning('No stats returned — skipping %s', label)
        return

    # 3. Save long-format
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
        gk_df.to_parquet(out_dir / 'goalkeepers.parquet',    index=False, engine='pyarrow')
        gk_df.to_csv(out_dir / 'goalkeepers.csv',            index=False)
        log.info('━━━ DONE %s — outfield: %d rows | GK: %d rows ━━━',
                 label, len(outfield_df), len(gk_df))
    except Exception as e:
        log.error('Failed building/saving wide format: %s', e)


if __name__ == '__main__':
    if not BROWSER_COOKIE:
        print("\n⚠️  BROWSER_COOKIE is not set.")
        print("   Pass 1 (__NEXT_DATA__) will run but Pass 2 (API fallback) will be skipped.")
        print("   To enable Pass 2, paste your browser cookie into BROWSER_COOKIE in CONFIG.\n")
    else:
        print("\n✅ BROWSER_COOKIE is set — both Pass 1 and Pass 2 will run.\n")

    print(f'Scraping {len(SCRAPE_TARGETS)} (league, season) combinations...\n')
    for i, (league_id, season) in enumerate(SCRAPE_TARGETS, 1):
        print(f'\n[{i}/{len(SCRAPE_TARGETS)}] League {league_id} — {season}')
        run_one(league_id, season)
    print('\n✓ All done.')