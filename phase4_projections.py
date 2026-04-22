"""Phase 4: Pull Preseason Projections.

Coverage hierarchy (per spec):
  2000–2005 : Marcel projected W% — expected to be unavailable; flags as null.
  2006–2025 : ZiPS projected W% from FanGraphs; Steamer as fallback.

All unavailable or unmatched projections → projected_wpct = null, flagged in
data_audit.txt.  No imputation.

Outputs: outputs/projections_table.csv
"""

import os
import re
import time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils import bbref_get, log_audit, OUTPUT_DIR, HEADERS

EVENT_TABLE = os.path.join(OUTPUT_DIR, 'event_table.csv')
OUT_PATH    = os.path.join(OUTPUT_DIR, 'projections_table.csv')

# FanGraphs team IDs (used in some API URLs)
FG_TEAM_IDS = {
    'ARI': 15, 'ATL': 16, 'BAL': 2,  'BOS': 3,  'CHC': 17, 'CWS': 4,
    'CIN': 18, 'CLE': 5,  'COL': 19, 'DET': 6,  'HOU': 21, 'KCR': 7,
    'LAD': 22, 'LAA': 1,  'ANA': 1,  'MIA': 28, 'FLA': 28, 'MIL': 23,
    'MIN': 8,  'MON': 27, 'NYM': 25, 'NYY': 9,  'OAK': 10, 'PHI': 26,
    'PIT': 27, 'SDP': 29, 'SEA': 11, 'SFG': 30, 'STL': 24, 'TBR': 12,
    'TBD': 12, 'TEX': 13, 'TOR': 14, 'WSN': 20,
}


# ── Marcel (2000–2005) ─────────────────────────────────────────────────────────
def get_marcel_wpct(team, year):
    """
    Marcel team-level W% projections for 2000–2005.

    Marcel was designed as a player-level system and team W% aggregates were
    never published in a consistently archived, machine-readable format.
    Returning None and flagging for every season in this range.
    """
    log_audit(
        f'PROJECTION_UNAVAILABLE: Marcel team W% not available in scrapable form '
        f'for {team} {year}. Setting projected_wpct=null.', 'PROJECTION'
    )
    return None, 'unavailable'


# ── ZiPS via FanGraphs ─────────────────────────────────────────────────────────
def get_zips_wpct(team, year):
    """
    Attempt to scrape ZiPS projected W% for team-season from FanGraphs.

    FanGraphs keeps historical depth-chart / projected-standings pages.
    We try two URL patterns:
      1. The projections API endpoint (JSON, preferred)
      2. The depth-charts HTML page
    Returns (projected_wpct float or None, source str).
    """
    # Attempt 1: FanGraphs projections API (works for recent years)
    api_url = (
        f'https://www.fangraphs.com/api/projections'
        f'?type=zips&stats=pit&pos=all&team=0&players=0&lg=all&season={year}'
    )
    time.sleep(2)
    try:
        resp = requests.get(api_url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            # data is a list of player projection dicts; team-level W% is not
            # directly in this endpoint. Fall through to HTML approach.
    except Exception:
        pass

    # Attempt 2: FanGraphs projected standings / depth charts (HTML)
    # URL pattern for historical projected standings varies; try both formats.
    fg_team_id = FG_TEAM_IDS.get(team)
    if fg_team_id is None:
        log_audit(
            f'PROJECTION_UNAVAILABLE: No FanGraphs team ID for {team} {year}.',
            'PROJECTION'
        )
        return None, 'unavailable'

    urls_to_try = [
        # Post-2015 format
        (f'https://www.fangraphs.com/depthcharts.aspx'
         f'?position=Team&teamid={fg_team_id}&statgroup=ZiPS&type=DC&season={year}'),
        # Older format
        (f'https://www.fangraphs.com/projections.aspx'
         f'?pos=all&stats=bat&type=zips&team={fg_team_id}&lg=all&players=0&season={year}'),
    ]

    for url in urls_to_try:
        time.sleep(2)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                continue
            wpct = _parse_fg_projected_wpct(resp.text, team, year)
            if wpct is not None:
                return wpct, 'ZiPS'
        except Exception as exc:
            log_audit(f'FanGraphs ZiPS scrape error {team} {year}: {exc}', 'WARNING')

    log_audit(
        f'PROJECTION_UNAVAILABLE: ZiPS W% not found for {team} {year} '
        f'after trying FanGraphs URL patterns.', 'PROJECTION'
    )
    return None, 'unavailable'


def _parse_fg_projected_wpct(html, team, year):
    """
    Attempt to extract a projected win percentage from FanGraphs HTML.
    Returns float or None.
    """
    soup = BeautifulSoup(html, 'lxml')

    # Look for a table or data element containing win% or projected W
    # FanGraphs depth charts have a "Projected Record" section with W and L columns.
    # Pattern: find cells near "W" and "L" column headers.
    for table in soup.find_all('table'):
        headers = [th.get_text(strip=True).lower() for th in table.find_all('th')]
        if 'w' in headers and 'l' in headers:
            try:
                df = pd.read_html(str(table))[0]
                df.columns = [str(c).lower().strip() for c in df.columns]
                if 'w' in df.columns and 'l' in df.columns:
                    # Sum projected W and L across all rows (team totals)
                    w = pd.to_numeric(df['w'], errors='coerce').sum()
                    l = pd.to_numeric(df['l'], errors='coerce').sum()
                    if w > 0 or l > 0:
                        return w / (w + l)
            except Exception:
                pass

    # Also check for text like "82-80" or "W% .509"
    text = soup.get_text(' ')
    wpct_match = re.search(r'W%[:\s]+\.?([0-9]{3,4})', text)
    if wpct_match:
        val = float('0.' + wpct_match.group(1).lstrip('0') or '0')
        if 0.2 < val < 0.8:
            return val

    return None


# ── Steamer via FanGraphs (fallback) ──────────────────────────────────────────
def get_steamer_wpct(team, year):
    """
    Try Steamer projected W% from FanGraphs as ZiPS fallback.
    Steamer available on FanGraphs ~2013+.
    """
    if year < 2013:
        log_audit(
            f'PROJECTION_UNAVAILABLE: Steamer not available before 2013. '
            f'{team} {year} projected_wpct=null.', 'PROJECTION'
        )
        return None, 'unavailable'

    fg_team_id = FG_TEAM_IDS.get(team)
    if fg_team_id is None:
        return None, 'unavailable'

    url = (f'https://www.fangraphs.com/depthcharts.aspx'
           f'?position=Team&teamid={fg_team_id}&statgroup=Steamer&type=DC&season={year}')
    time.sleep(2)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            wpct = _parse_fg_projected_wpct(resp.text, team, year)
            if wpct is not None:
                return wpct, 'Steamer'
    except Exception as exc:
        log_audit(f'Steamer scrape error {team} {year}: {exc}', 'WARNING')

    log_audit(
        f'PROJECTION_UNAVAILABLE: Steamer W% not found for {team} {year}.',
        'PROJECTION'
    )
    return None, 'unavailable'


# ── Dispatch per-event ─────────────────────────────────────────────────────────
def get_projection(team, year):
    """
    Apply coverage hierarchy and return (projected_wpct, projection_source).
    """
    if year <= 2005:
        return get_marcel_wpct(team, year)

    # Try ZiPS first
    wpct, source = get_zips_wpct(team, year)
    if wpct is not None:
        return wpct, source

    # Steamer fallback
    wpct, source = get_steamer_wpct(team, year)
    return wpct, source


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    print("Phase 4: Pulling preseason projections ...")
    log_audit('='*60, 'PHASE4_START')

    if not os.path.exists(EVENT_TABLE):
        print(f"  ERROR: {EVENT_TABLE} not found. Run Phase 1 first.")
        return

    events = pd.read_csv(EVENT_TABLE)
    print(f"  Loaded {len(events)} events from event_table.csv")

    # Load metrics for actual W% values
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics_table.csv')
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
    else:
        log_audit('Phase 4: metrics_table.csv not found; residuals cannot be calculated.', 'WARNING')
        metrics = pd.DataFrame()

    # Cache projections by (team, year) to avoid duplicate scrapes
    proj_cache = {}
    records = []

    for _, ev in events.iterrows():
        fid  = int(ev['firing_id'])
        team = str(ev['team'])
        year = int(ev['season'])
        key  = (team, year)

        if key not in proj_cache:
            print(f"    Fetching projection: {team} {year} ...")
            wpct, source = get_projection(team, year)
            proj_cache[key] = (wpct, source)
        else:
            wpct, source = proj_cache[key]

        # Look up actual W% for residual calculation
        if len(metrics) > 0:
            m_row = metrics[metrics['firing_id'] == fid]
            actual_pre  = m_row['actual_wpct_pre'].iloc[0]  if len(m_row) else np.nan
            actual_post = m_row['actual_wpct_post'].iloc[0] if len(m_row) else np.nan
        else:
            actual_pre  = np.nan
            actual_post = np.nan

        actual_pre  = float(actual_pre)  if not pd.isna(actual_pre)  else np.nan
        actual_post = float(actual_post) if not pd.isna(actual_post) else np.nan
        wpct_f      = float(wpct)        if wpct is not None         else np.nan

        res_pre   = actual_pre  - wpct_f if not np.isnan(actual_pre)  and not np.isnan(wpct_f) else np.nan
        res_post  = actual_post - wpct_f if not np.isnan(actual_post) and not np.isnan(wpct_f) else np.nan
        res_delta = res_post    - res_pre if not np.isnan(res_post)   and not np.isnan(res_pre) else np.nan

        records.append({
            'firing_id':                fid,
            'team':                     team,
            'season':                   year,
            'projected_wpct':           wpct_f if not np.isnan(wpct_f) else None,
            'projection_source':        source,
            'projection_residual_pre':  res_pre   if not np.isnan(res_pre)   else None,
            'projection_residual_post': res_post  if not np.isnan(res_post)  else None,
            'projection_residual_delta': res_delta if not np.isnan(res_delta) else None,
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_PATH, index=False)

    n          = len(df_out)
    n_avail    = df_out['projected_wpct'].notna().sum()
    n_unavail  = n - n_avail
    src_counts = df_out['projection_source'].value_counts().to_dict()

    print(f"Phase 4 complete — {n} records written to projections_table.csv")
    print(f"  Projection coverage: {n_avail}/{n} events have projected_wpct")
    print(f"  Sources: {src_counts}")

    log_audit(
        f'Phase 4 complete. {n} projection records. {n_avail} with projected_wpct. '
        f'{n_unavail} unavailable. Sources: {src_counts}. '
        f'NOTE: FanGraphs HTML structure varies by year — manual verification of '
        f'projection values strongly recommended, especially 2006–2014.',
        'PHASE4_END'
    )


if __name__ == '__main__':
    run()
