"""Phase 1: Build the Event Table.

Primary source: Lahman Managers.csv (seanlahman/baseballdatabank on GitHub).
               Covers through 2016; BBRef scraping supplements 2017–2025.
Enrichment / verification: Baseball Reference team schedule pages.

Outputs: outputs/event_table.csv
         Exclusions and anomalies logged to data_audit.txt
"""

import os
import io
import time
import re
import zipfile
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from utils import (
    bbref_get, log_audit, lahman_to_bbref,
    BASE_DIR, OUTPUT_DIR, clean_schedule_df
)

OUT_PATH = os.path.join(OUTPUT_DIR, 'event_table.csv')

START_YEAR = 2000
END_YEAR   = 2025

# seanlahman/baseballdatabank — the current home of the Lahman DB (covers ~1871–2016)
# chadwickbureau/baseballdatabank no longer exists on GitHub.
_LAHMAN_ZIP_URL   = 'https://github.com/seanlahman/baseballdatabank/archive/master.zip'
_LAHMAN_ZIP_PREFIX = 'baseballdatabank-master/core/'
_LAHMAN_CACHE_DIR  = os.path.join(BASE_DIR, 'lahman_cache')

# In the seanlahman repo the people file is Master.csv, not People.csv
_PEOPLE_FILENAME  = 'Master.csv'


def _ensure_lahman_cache():
    """Download the Lahman zip once and extract all core CSVs to lahman_cache/."""
    os.makedirs(_LAHMAN_CACHE_DIR, exist_ok=True)
    managers_local = os.path.join(_LAHMAN_CACHE_DIR, 'Managers.csv')

    if os.path.exists(managers_local):
        return  # already cached

    print("  Downloading Lahman database zip from seanlahman/baseballdatabank ...")
    resp = requests.get(_LAHMAN_ZIP_URL, allow_redirects=True, timeout=60)
    resp.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    extracted = 0
    for member in z.namelist():
        if member.startswith(_LAHMAN_ZIP_PREFIX) and member.endswith('.csv'):
            filename = member.replace(_LAHMAN_ZIP_PREFIX, '')
            if not filename:
                continue
            local_path = os.path.join(_LAHMAN_CACHE_DIR, filename)
            with z.open(member) as src, open(local_path, 'wb') as dst:
                dst.write(src.read())
            extracted += 1

    print(f"  Cached {extracted} Lahman CSV files to lahman_cache/")
    log_audit(
        f'Lahman zip downloaded from seanlahman/baseballdatabank. '
        f'{extracted} CSVs cached. Coverage: ~1871–2016. '
        f'Seasons 2017–{END_YEAR} will be supplemented via BBRef scraping.',
        'LAHMAN_LOAD'
    )


def _load_lahman_csv(filename):
    _ensure_lahman_cache()
    path = os.path.join(_LAHMAN_CACHE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Lahman cache missing: {filename}")
    return pd.read_csv(path, low_memory=False)


# ── Lahman loaders ─────────────────────────────────────────────────────────────
def load_lahman_managers():
    mgr = _load_lahman_csv('Managers.csv')
    ppl = _load_lahman_csv(_PEOPLE_FILENAME)[['playerID', 'nameFirst', 'nameLast']].copy()
    ppl['manager_name'] = (ppl['nameFirst'].fillna('') + ' '
                           + ppl['nameLast'].fillna('')).str.strip()
    merged = mgr.merge(ppl, on='playerID', how='left')
    merged['yearID']   = merged['yearID'].astype(int)
    merged['inseason'] = merged['inseason'].astype(int)
    merged['G']        = pd.to_numeric(merged['G'], errors='coerce').fillna(0).astype(int)
    return merged[merged['yearID'].between(START_YEAR, END_YEAR)].copy()


# ── Supplement Lahman with BBRef for years beyond Lahman coverage ──────────────
def scrape_bbref_managers_for_year(team_bbref, year):
    """
    Scrape the BBRef team schedule page and infer manager changes from the
    game log. Returns a list of dicts with keys:
      manager_name, games_managed, inseason_order
    (best-effort; falls back gracefully on parse failures)
    """
    url = (f'https://www.baseball-reference.com/teams/'
           f'{team_bbref}/{year}.shtml')
    resp = bbref_get(url)
    if not resp:
        return []
    soup = BeautifulSoup(resp.text, 'lxml')

    # BBRef team pages have a "Managers" note in the team meta section
    # e.g. "Managers: Joe Torre (55-40), Don Mattingly (..."
    results = []
    for tag in soup.find_all(['p', 'div', 'li']):
        text = tag.get_text(' ', strip=True)
        if text.startswith('Managers:') or 'Manager:' in text[:20]:
            # Parse entries like "Name (W-L)" or "Name (G games, W-L)"
            entries = re.findall(r'([A-Z][a-z]+(?:\s[A-Z][a-z\.]+)+)\s*\(([^)]+)\)', text)
            for i, (name, record) in enumerate(entries, start=1):
                # Count games from W-L or total G in record string
                g_match = re.search(r'(\d+)-(\d+)', record)
                g = (int(g_match.group(1)) + int(g_match.group(2))) if g_match else None
                results.append({
                    'manager_name': name.strip(),
                    'games_managed': g,
                    'inseason_order': i,
                })
            break
    return results


# ── Game log retrieval ─────────────────────────────────────────────────────────
_game_log_cache = {}


def get_game_log(year, bbref_team):
    """Return cleaned game-log DataFrame for team-season. Cached per (year, team)."""
    key = (year, bbref_team)
    if key in _game_log_cache:
        return _game_log_cache[key]

    df = None
    try:
        from pybaseball import schedule_and_record
        raw = schedule_and_record(year, bbref_team)
        df = clean_schedule_df(raw, year)
    except Exception as exc:
        log_audit(
            f'pybaseball schedule_and_record failed for {bbref_team} {year}: {exc}. '
            'Falling back to direct BBRef scrape.', 'WARNING'
        )

    if df is None or len(df) == 0:
        df = _scrape_schedule_from_bbref(year, bbref_team)

    _game_log_cache[key] = df
    return df


def _scrape_schedule_from_bbref(year, bbref_team):
    """Direct BBRef schedule scrape fallback."""
    url = f'https://www.baseball-reference.com/teams/{bbref_team}/{year}-schedule.shtml'
    resp = bbref_get(url)
    if not resp:
        log_audit(f'BBRef schedule scrape returned no response for {bbref_team} {year}', 'ERROR')
        return pd.DataFrame()
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', id='team_schedule')
    if table is None:
        log_audit(f'No #team_schedule table found for {bbref_team} {year}', 'ERROR')
        return pd.DataFrame()
    try:
        df_raw = pd.read_html(str(table))[0]
        return clean_schedule_df(df_raw, year)
    except Exception as exc:
        log_audit(f'Failed to parse schedule table for {bbref_team} {year}: {exc}', 'ERROR')
        return pd.DataFrame()


# ── is_outsider determination ──────────────────────────────────────────────────
def determine_is_outsider(replacement_name, team_bbref, fire_year, lahman_managers):
    """
    Attempt to classify replacement manager as insider (0) or outsider (1).

    Logic (in priority order):
      1. If replacement appears in the BBRef coaching staff list for this org
         in fire_year or fire_year-1 or fire_year-2 -> insider (0)
      2. If replacement appears in Lahman Managers for this team in fire_year-1
         through fire_year-3 as a manager (bench roles, etc.) -> insider (0)
      3. Otherwise -> outsider (1)

    All results flagged in data_audit.txt for manual verification.
    Returns (is_outsider int, confidence str)
    """
    # Step 2: Lahman check (replacement managed this org recently)
    recent = lahman_managers[
        (lahman_managers['yearID'].between(fire_year - 3, fire_year - 1))
    ]
    # Match team across all possible lahman IDs for this bbref team
    # (crude name match against manager_name)
    name_parts = replacement_name.lower().split()
    for _, row in recent.iterrows():
        row_bbref = lahman_to_bbref(str(row['teamID']), int(row['yearID']))
        if row_bbref != team_bbref:
            continue
        mgr_n = str(row.get('manager_name', '')).lower()
        if any(p in mgr_n for p in name_parts if len(p) > 2):
            log_audit(
                f'VERIFY_OUTSIDER: {replacement_name} at {team_bbref} {fire_year} '
                f'classified as INSIDER (0) — found in Lahman as prior manager for '
                f'same org.', 'OUTSIDER_CHECK'
            )
            return 0, 'lahman_prior_manager'

    # Step 1: BBRef coaching staff scrape
    is_outsider, confidence = _check_bbref_coaching_staff(
        replacement_name, team_bbref, fire_year
    )
    return is_outsider, confidence


def _check_bbref_coaching_staff(replacement_name, team_bbref, fire_year):
    """Scrape BBRef team page to check if replacement was on coaching staff."""
    name_parts = [p.lower() for p in replacement_name.split() if len(p) > 2]

    for check_year in [fire_year, fire_year - 1]:
        url = f'https://www.baseball-reference.com/teams/{team_bbref}/{check_year}.shtml'
        resp = bbref_get(url)
        if not resp:
            continue
        soup = BeautifulSoup(resp.text, 'lxml')

        # Look for coaches table
        coaches_div = soup.find(id='div_coaches')
        if coaches_div:
            text = coaches_div.get_text(' ', strip=True).lower()
        else:
            # Fall back to searching all page text for coaching sections
            # Look for a "Coaches" header or similar
            coach_header = soup.find(lambda tag: tag.name in ('h2', 'h3', 'strong')
                                     and 'coach' in tag.get_text('').lower())
            text = coach_header.parent.get_text(' ', strip=True).lower() if coach_header else ''

        if text and all(p in text for p in name_parts):
            log_audit(
                f'VERIFY_OUTSIDER: {replacement_name} at {team_bbref} {fire_year} '
                f'classified as INSIDER (0) — found in coaching staff '
                f'(year checked: {check_year}). MANUAL VERIFICATION NEEDED.',
                'OUTSIDER_CHECK'
            )
            return 0, f'bbref_coaching_staff_{check_year}'

    # Not found in coaching staff — classify as outsider
    log_audit(
        f'VERIFY_OUTSIDER: {replacement_name} at {team_bbref} {fire_year} '
        f'classified as OUTSIDER (1) — not found in recent coaching staff. '
        f'MANUAL VERIFICATION NEEDED.', 'OUTSIDER_CHECK'
    )
    return 1, 'not_found_in_coaching_staff'


# ── Post-window calculation ────────────────────────────────────────────────────
def calc_post_window(game_log, game_number_at_firing, next_firing_game_number=None):
    """
    Given a season game log (cleaned DataFrame) and the game number at firing,
    return (post_games_available, truncated_window).

    post_games_available: actual games played in 40-game post window,
                          potentially truncated by second firing or season end.
    truncated_window:     1 if second firing occurs within 40 games; 0 otherwise.
    """
    if game_log is None or len(game_log) == 0:
        return None, None

    # Games after firing (game_number > game_number_at_firing)
    post = game_log[game_log['game_number'] > game_number_at_firing].copy()
    post = post.sort_values('game_number').reset_index(drop=True)

    if len(post) == 0:
        return 0, 0

    # Determine upper bound
    max_post_game = game_number_at_firing + 40

    truncated = 0
    if next_firing_game_number is not None:
        if next_firing_game_number <= max_post_game:
            # Second firing within 40-game window — truncate at that point
            max_post_game = next_firing_game_number
            truncated = 1

    # Count actual games within [game_number_at_firing+1, max_post_game]
    window = post[post['game_number'] <= max_post_game]
    post_games = len(window)

    return post_games, truncated


# ── BBRef supplement for years beyond Lahman ──────────────────────────────────
# All BBRef team IDs for years the Lahman DB may not cover (2024–2025)
ALL_TEAMS_2024 = [
    'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CWS', 'CIN', 'CLE', 'COL',
    'DET', 'HOU', 'KCR', 'LAD', 'LAA', 'MIA', 'MIL', 'MIN', 'NYM',
    'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR',
    'TEX', 'TOR', 'WSN',
]


def build_supplement_from_bbref(lahman_max_year, lahman_managers):
    """
    Scrape BBRef for seasons beyond Lahman coverage.
    Returns a list of event dicts in the same format as the Lahman-sourced events.
    """
    supplement = []
    for year in range(lahman_max_year + 1, END_YEAR + 1):
        for team in ALL_TEAMS_2024:
            time.sleep(2)  # extra caution
            events = _scrape_bbref_single_season_managers(team, year, lahman_managers)
            supplement.extend(events)
    return supplement


def _scrape_bbref_single_season_managers(team_bbref, year, lahman_managers):
    """Return firing events for one team-season scraped from BBRef."""
    url = f'https://www.baseball-reference.com/teams/{team_bbref}/{year}.shtml'
    resp = bbref_get(url)
    if not resp:
        return []
    soup = BeautifulSoup(resp.text, 'lxml')

    # Parse manager entries from team meta section
    mgr_entries = []
    for tag in soup.find_all(['p']):
        txt = tag.get_text(' ', strip=True)
        if txt.startswith('Manager:') or txt.startswith('Managers:'):
            # e.g. "Managers: Bob Melvin (68-50) and Mark Kotsay (46-..."
            entries = re.findall(
                r'([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-\.]+)+)\s*\((\d+)-(\d+)\)',
                txt
            )
            for name, wins, losses in entries:
                g = int(wins) + int(losses)
                mgr_entries.append({'name': name.strip(), 'G': g})
            break

    if len(mgr_entries) <= 1:
        return []  # No mid-season change

    # Multiple managers found — build events
    events = []
    cumulative_games = 0
    for i in range(len(mgr_entries) - 1):  # last manager not fired
        fired_name = mgr_entries[i]['name']
        repl_name  = mgr_entries[i + 1]['name']
        g_fired    = mgr_entries[i]['G']
        game_num   = cumulative_games + g_fired
        cumulative_games += g_fired

        events.append({
            '_source': 'bbref_supplement',
            '_bbref_team': team_bbref,
            '_fired_name': fired_name,
            '_repl_name': repl_name,
            '_pre_games': g_fired,
            '_game_number_at_firing': game_num,
            '_year': year,
        })
    return events


# ── Main builder ───────────────────────────────────────────────────────────────
def run():
    print("Phase 1: Building event table ...")
    log_audit('='*60, 'PHASE1_START')
    log_audit('Phase 1 event table build started.', 'PHASE1_START')

    # Load Lahman
    print("  Loading Lahman Managers + People tables ...")
    lahman_df = load_lahman_managers()
    lahman_max_year = int(lahman_df['yearID'].max())
    print(f"  Lahman coverage: {lahman_df['yearID'].min()}–{lahman_max_year}")

    # Identify mid-season firing events from Lahman
    # A firing event = a manager with inseason < max(inseason) for that team-year
    grp = lahman_df.groupby(['teamID', 'yearID'])
    events_raw = []

    for (team_lahman, year), group in grp:
        group = group.sort_values('inseason').reset_index(drop=True)
        max_inseason = group['inseason'].max()
        if max_inseason == 1:
            continue  # no mid-season change

        # Each manager except the last one was fired
        cumulative_g = 0
        for idx, row in group.iterrows():
            if row['inseason'] == max_inseason:
                break  # last manager, not fired
            fired_g = int(row['G'])
            game_num = cumulative_g + fired_g
            cumulative_g += fired_g

            # Replacement manager
            next_rows = group[group['inseason'] == row['inseason'] + 1]
            repl_name = next_rows.iloc[0]['manager_name'] if len(next_rows) else 'Unknown'

            events_raw.append({
                '_source':                 'lahman',
                '_lahman_team':            team_lahman,
                '_bbref_team':             lahman_to_bbref(team_lahman, year),
                '_fired_name':             row['manager_name'],
                '_repl_name':              repl_name,
                '_pre_games':              fired_g,
                '_game_number_at_firing':  game_num,
                '_year':                   year,
                '_inseason':               int(row['inseason']),
            })

    print(f"  Found {len(events_raw)} firing events from Lahman "
          f"({START_YEAR}–{lahman_max_year}).")

    # Supplement with BBRef scraping for years beyond Lahman
    if lahman_max_year < END_YEAR:
        print(f"  Lahman only covers through {lahman_max_year}. "
              f"Scraping BBRef for {lahman_max_year+1}–{END_YEAR} ...")
        supplement = build_supplement_from_bbref(lahman_max_year, lahman_df)
        print(f"  Found {len(supplement)} additional events from BBRef supplement.")
        events_raw.extend(supplement)

    # For each event, determine the post window, truncation, and second firing
    print("  Pulling game logs and calculating windows ...")
    records = []
    excl_count = 0
    firing_id = 0

    for ev in events_raw:
        year      = ev['_year']
        bbref_tm  = ev['_bbref_team']
        pre_games = ev['_pre_games']
        game_num  = ev['_game_number_at_firing']

        gl = get_game_log(year, bbref_tm)

        if gl is None or len(gl) == 0:
            log_audit(
                f"EXCLUDED firing_id=TBD {ev['_fired_name']} {bbref_tm} {year}: "
                f"no game log available.", 'EXCLUSION'
            )
            excl_count += 1
            continue

        # Check for a subsequent firing in the same team-season
        # (look for another event with same team/year and higher game_number)
        next_game_num = None
        next_fire_date = None
        for ev2 in events_raw:
            if (ev2['_year'] == year and ev2['_bbref_team'] == bbref_tm
                    and ev2['_game_number_at_firing'] > game_num):
                next_game_num = ev2['_game_number_at_firing']
                break

        post_games, truncated = calc_post_window(gl, game_num, next_game_num)

        if post_games is None:
            log_audit(
                f"EXCLUDED {ev['_fired_name']} {bbref_tm} {year}: "
                f"could not calculate post window.", 'EXCLUSION'
            )
            excl_count += 1
            continue

        if post_games < 25:
            log_audit(
                f"EXCLUDED {ev['_fired_name']} {bbref_tm} {year}: "
                f"post_games_available={post_games} < 25 (minimum threshold).", 'EXCLUSION'
            )
            excl_count += 1
            continue

        # Determine firing date from game log (date of last game under fired mgr)
        fired_game_rows = gl[gl['game_number'] == game_num]
        if len(fired_game_rows) > 0 and 'game_date' in fired_game_rows.columns:
            date_fired = fired_game_rows.iloc[0]['game_date']
        else:
            # If exact game not found, use nearest
            below = gl[gl['game_number'] <= game_num]
            date_fired = below.iloc[-1]['game_date'] if len(below) > 0 else None

        if date_fired is None:
            log_audit(
                f"AMBIGUOUS date_fired for {ev['_fired_name']} {bbref_tm} {year}: "
                f"game #{game_num} not found in log. Date set to None.", 'WARNING'
            )

        # Second firing date (if truncated)
        if truncated and next_game_num is not None:
            sf_rows = gl[gl['game_number'] == next_game_num]
            next_fire_date = (sf_rows.iloc[0]['game_date']
                              if len(sf_rows) > 0 else None)

        # Determine is_outsider
        is_out, out_confidence = determine_is_outsider(
            str(ev['_repl_name']), bbref_tm, year, lahman_df
        )

        firing_id += 1
        rec = {
            'firing_id':               firing_id,
            'team':                    bbref_tm,
            'manager_fired':           ev['_fired_name'],
            'manager_replacement':     ev['_repl_name'],
            'date_fired':              date_fired,
            'game_number_at_firing':   game_num,
            'pre_games':               pre_games,
            'post_games_available':    post_games,
            'truncated_window':        int(truncated),
            'second_firing_date':      next_fire_date if truncated else None,
            'is_outsider':             is_out,
            'is_outsider_confidence':  out_confidence,
            'season':                  year,
            'source':                  ev.get('_source', 'lahman'),
        }
        records.append(rec)

    df_out = pd.DataFrame(records)

    if len(df_out) == 0:
        print("  WARNING: no events survived filtering. Check data_audit.txt.")
        log_audit('Phase 1: No events survived filtering.', 'WARNING')
        df_out.to_csv(OUT_PATH, index=False)
        print(f"Phase 1 complete — 0 records written to event_table.csv")
        return

    df_out = df_out.sort_values(['season', 'team', 'game_number_at_firing'])
    df_out.reset_index(drop=True, inplace=True)

    # Re-assign sequential firing_ids after sort
    df_out['firing_id'] = range(1, len(df_out) + 1)

    df_out.to_csv(OUT_PATH, index=False)
    n = len(df_out)
    print(f"Phase 1 complete — {n} records written to event_table.csv "
          f"({excl_count} events excluded; see data_audit.txt)")
    log_audit(
        f'Phase 1 complete. {n} events in event_table.csv. '
        f'{excl_count} excluded. '
        f'NOTE: all is_outsider values require manual verification.',
        'PHASE1_END'
    )


if __name__ == '__main__':
    run()
