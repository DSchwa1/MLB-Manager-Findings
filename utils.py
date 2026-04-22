"""Shared utilities for the MLB Manager Firings analysis pipeline."""

import os
import time
import requests
import numpy as np
from datetime import datetime

# ── Directory paths ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs')
CHARTS_DIR  = os.path.join(OUTPUT_DIR, 'charts')
AUDIT_FILE  = os.path.join(BASE_DIR, 'data_audit.txt')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── HTTP / rate-limiting ───────────────────────────────────────────────────────
BBREF_DELAY = 2.0   # mandatory seconds between all BBRef requests

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    )
}


def bbref_get(url, session=None):
    """Fetch a BBRef URL with mandatory 2-second delay. Returns Response or None."""
    time.sleep(BBREF_DELAY)
    requester = session if session is not None else requests
    try:
        resp = requester.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp
    except Exception as exc:
        log_audit(f"HTTP error fetching {url}: {exc}", category="ERROR")
        return None


# ── Audit logging ──────────────────────────────────────────────────────────────
def log_audit(message, category="NOTE"):
    """Append a timestamped entry to data_audit.txt."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] [{category}] {message}\n"
    with open(AUDIT_FILE, 'a', encoding='utf-8') as fh:
        fh.write(line)


# ── Team-ID mapping ────────────────────────────────────────────────────────────
# Maps Lahman teamID -> Baseball Reference abbreviation. Year-aware where needed.
_STATIC_MAP = {
    'ARZ': 'ARI',
    'ATL': 'ATL',
    'BAL': 'BAL',
    'BOS': 'BOS',
    'CHA': 'CWS',
    'CHN': 'CHC',
    'CIN': 'CIN',
    'CLE': 'CLE',   # covers both Indians and Guardians eras; BBRef uses CLE both
    'CLU': 'CLE',
    'COL': 'COL',
    'DET': 'DET',
    'FLO': 'FLA',
    'HOU': 'HOU',
    'KCA': 'KCR',
    'LAN': 'LAD',
    'MIA': 'MIA',
    'MIL': 'MIL',
    'MIN': 'MIN',
    'MON': 'MON',
    'NYA': 'NYY',
    'NYN': 'NYM',
    'OAK': 'OAK',
    'PHI': 'PHI',
    'PIT': 'PIT',
    'SDN': 'SDP',
    'SEA': 'SEA',
    'SFN': 'SFG',
    'SLN': 'STL',
    'TEX': 'TEX',
    'TOR': 'TOR',
    'WAS': 'WSN',
}


def lahman_to_bbref(lahman_id, year):
    """Convert Lahman teamID + season year to the matching BBRef abbreviation."""
    if lahman_id == 'ANA':
        # Anaheim Angels -> Los Angeles Angels of Anaheim (2005)
        return 'LAA' if year >= 2005 else 'ANA'
    if lahman_id == 'TBA':
        # Tampa Bay Devil Rays -> Tampa Bay Rays (2008)
        return 'TBR' if year >= 2008 else 'TBD'
    return _STATIC_MAP.get(lahman_id, lahman_id)


# ── Pythagorean W% ─────────────────────────────────────────────────────────────
PYTH_EXP = 1.83


def pythagorean_wpct(runs_scored, runs_allowed):
    """Return Pythagorean W% using exponent 1.83.

    Accepts scalars or array-likes. Returns NaN when runs_allowed == 0 or
    when either argument is NaN.
    """
    rs = np.asarray(runs_scored, dtype=float)
    ra = np.asarray(runs_allowed, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(
            (ra == 0) | np.isnan(rs) | np.isnan(ra),
            np.nan,
            rs ** PYTH_EXP / (rs ** PYTH_EXP + ra ** PYTH_EXP)
        )
    # Return scalar if inputs were scalar
    if result.ndim == 0:
        return float(result)
    return result


# ── Game-log cleaning ──────────────────────────────────────────────────────────
import re

MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}


def parse_bbref_date(date_str, year):
    """Parse a BBRef schedule date string (e.g. 'Thursday Apr  1') to datetime.date.

    Returns None on failure.
    """
    if not isinstance(date_str, str):
        return None
    # Strip doubleheader markers like "(1)" or "(2)"
    clean = re.sub(r'\(\d\)', '', date_str).strip()
    # Strip day-of-week prefix
    parts = clean.split()
    # Find month abbreviation
    month = None
    day = None
    for i, p in enumerate(parts):
        if p[:3].capitalize() in MONTH_MAP:
            month = MONTH_MAP[p[:3].capitalize()]
            # next non-empty token is day
            for p2 in parts[i + 1:]:
                if p2.isdigit():
                    day = int(p2)
                    break
            break
    if month is None or day is None:
        return None
    try:
        from datetime import date
        return date(year, month, day)
    except ValueError:
        return None


def clean_schedule_df(df, year):
    """Normalise a raw pybaseball schedule_and_record DataFrame.

    Returns a cleaned DataFrame with columns:
      game_number, game_date, runs_scored, runs_allowed, result
    Only rows with actual game results (W / L / T) are kept.
    """
    import pandas as pd

    df = df.copy()

    # Rename common column variants
    rename = {}
    for col in df.columns:
        cl = col.strip()
        if cl in ('Gm#', 'Gm', 'G#'):
            rename[col] = 'game_number'
        elif cl == 'Date':
            rename[col] = 'date_raw'
        elif cl == 'R':
            rename[col] = 'runs_scored'
        elif cl == 'RA':
            rename[col] = 'runs_allowed'
        elif cl in ('W/L', 'WL'):
            rename[col] = 'result_raw'
    df.rename(columns=rename, inplace=True)

    needed = {'game_number', 'date_raw', 'runs_scored', 'runs_allowed', 'result_raw'}
    missing = needed - set(df.columns)
    if missing:
        log_audit(f"clean_schedule_df: missing columns {missing} for year {year}", "WARNING")

    # Keep only rows with a game result
    if 'result_raw' in df.columns:
        mask = df['result_raw'].astype(str).str.strip().str[:1].isin(['W', 'L', 'T'])
        df = df[mask].copy()

    # Parse result to single character
    if 'result_raw' in df.columns:
        df['result'] = df['result_raw'].astype(str).str.strip().str[:1]

    # Parse dates
    if 'date_raw' in df.columns:
        df['game_date'] = df['date_raw'].apply(lambda d: parse_bbref_date(d, year))

    # Coerce numeric
    for col in ('game_number', 'runs_scored', 'runs_allowed'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing essentials
    essential = [c for c in ('game_number', 'game_date', 'runs_scored', 'runs_allowed', 'result')
                 if c in df.columns]
    df.dropna(subset=essential, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
