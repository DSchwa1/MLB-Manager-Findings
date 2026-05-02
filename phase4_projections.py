"""Phase 4: Preseason Quality Baseline.

Uses prior-year Pythagorean W% (from MLB Stats API final standings) as the
preseason expected win percentage for each firing event's team.

Prior-year Pythagorean W% is a well-validated, reproducible baseline for team
quality that is available for all seasons 1999–2025, giving full coverage
across the 2000–2025 study period.

Outputs: outputs/projections_table.csv
"""

import os
import numpy as np
import pandas as pd
import requests

from utils import (
    pythagorean_wpct, log_audit, OUTPUT_DIR,
    BBREF_TO_MLB_ID, MLB_API_BASE,
)

EVENT_TABLE = os.path.join(OUTPUT_DIR, 'event_table.csv')
OUT_PATH    = os.path.join(OUTPUT_DIR, 'projections_table.csv')

# Invert BBREF_TO_MLB_ID so we can look up bbref abbrev from numeric team ID.
# Multiple BBRef codes may map to the same MLB ID (e.g. ANA/LAA -> 108);
# keep the most current abbreviation.
_MLB_ID_TO_BBREF = {}
for abbrev, mlb_id in BBREF_TO_MLB_ID.items():
    _MLB_ID_TO_BBREF[mlb_id] = abbrev   # last write wins; later abbrevs are current

_standings_cache = {}


def get_prior_year_standings(year):
    """
    Fetch final regular-season standings for all teams in (year).
    Returns dict mapping MLB team ID -> {'rs': int, 'ra': int, 'w': int, 'l': int}.
    """
    if year in _standings_cache:
        return _standings_cache[year]

    url = (f'{MLB_API_BASE}/standings'
           f'?leagueId=103,104&season={year}&standingsTypes=regularSeason')
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log_audit(f'MLB API standings error for {year}: {exc}', 'ERROR')
        _standings_cache[year] = {}
        return {}

    result = {}
    for division in data.get('records', []):
        for tr in division.get('teamRecords', []):
            tid = tr['team']['id']
            result[tid] = {
                'w':  tr.get('wins', 0),
                'l':  tr.get('losses', 0),
                'rs': tr.get('runsScored', 0),
                'ra': tr.get('runsAllowed', 0),
            }

    _standings_cache[year] = result
    return result


def get_prior_year_pyth(team_bbref, season):
    """
    Return prior-year Pythagorean W% for team_bbref entering (season).
    Uses final standings from (season - 1).
    Returns (float or None, source_str).
    """
    prior_year = season - 1
    mlb_id = BBREF_TO_MLB_ID.get(team_bbref)
    if mlb_id is None:
        log_audit(
            f'PROJECTION_UNAVAILABLE: No MLB team ID for {team_bbref}. '
            f'projected_wpct=null.', 'PROJECTION'
        )
        return None, 'unavailable'

    standings = get_prior_year_standings(prior_year)
    record = standings.get(mlb_id)
    if record is None:
        log_audit(
            f'PROJECTION_UNAVAILABLE: {team_bbref} (id={mlb_id}) not found in '
            f'{prior_year} standings. projected_wpct=null.', 'PROJECTION'
        )
        return None, 'unavailable'

    rs, ra = record['rs'], record['ra']
    if rs == 0 and ra == 0:
        log_audit(
            f'PROJECTION_UNAVAILABLE: Zero runs for {team_bbref} {prior_year}. '
            f'projected_wpct=null.', 'PROJECTION'
        )
        return None, 'unavailable'

    pyth = pythagorean_wpct(rs, ra)
    return pyth, f'prior_year_pyth_{prior_year}'


def run():
    print("Phase 4: Computing preseason quality baselines (prior-year Pythagorean W%) ...")
    log_audit('='*60, 'PHASE4_START')

    if not os.path.exists(EVENT_TABLE):
        print(f"  ERROR: {EVENT_TABLE} not found. Run Phase 1 first.")
        return

    events = pd.read_csv(EVENT_TABLE)
    print(f"  Loaded {len(events)} events from event_table.csv")

    metrics_path = os.path.join(OUTPUT_DIR, 'metrics_table.csv')
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
    else:
        log_audit('Phase 4: metrics_table.csv not found; residuals cannot be calculated.', 'WARNING')
        metrics = pd.DataFrame()

    proj_cache = {}
    records = []

    for _, ev in events.iterrows():
        fid  = int(ev['firing_id'])
        team = str(ev['team'])
        year = int(ev['season'])
        key  = (team, year)

        if key not in proj_cache:
            print(f"    {team} {year}: fetching prior-year ({year-1}) Pythagorean W% ...")
            wpct, source = get_prior_year_pyth(team, year)
            proj_cache[key] = (wpct, source)
        else:
            wpct, source = proj_cache[key]

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
            'firing_id':                 fid,
            'team':                      team,
            'season':                    year,
            'projected_wpct':            wpct_f if not np.isnan(wpct_f) else None,
            'projection_source':         source,
            'projection_residual_pre':   res_pre   if not np.isnan(res_pre)   else None,
            'projection_residual_post':  res_post  if not np.isnan(res_post)  else None,
            'projection_residual_delta': res_delta if not np.isnan(res_delta) else None,
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_PATH, index=False)

    n_avail   = df_out['projected_wpct'].notna().sum()
    n_unavail = len(df_out) - n_avail
    src_counts = df_out['projection_source'].value_counts().to_dict()

    print(f"Phase 4 complete — {len(df_out)} records written to projections_table.csv")
    print(f"  Coverage: {n_avail}/{len(df_out)} events have projected_wpct")
    print(f"  Sources: {src_counts}")

    log_audit(
        f'Phase 4 complete. {len(df_out)} projection records. {n_avail} with '
        f'projected_wpct. {n_unavail} unavailable. Sources: {src_counts}.',
        'PHASE4_END'
    )


if __name__ == '__main__':
    run()
