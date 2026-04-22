"""Phase 2: Pull Game-Level Data.

For each firing event, pulls game-by-game results for the full pre window
and the 40-game post window (or truncated window as specified in event_table).

Outputs: outputs/game_log.csv
"""

import os
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from utils import (
    bbref_get, log_audit, clean_schedule_df,
    OUTPUT_DIR
)

EVENT_TABLE = os.path.join(OUTPUT_DIR, 'event_table.csv')
OUT_PATH    = os.path.join(OUTPUT_DIR, 'game_log.csv')


# ── Game log retrieval (cached) ────────────────────────────────────────────────
_cache = {}


def _get_raw_log(year, team):
    key = (year, team)
    if key in _cache:
        return _cache[key]

    df = None
    try:
        from pybaseball import schedule_and_record
        raw = schedule_and_record(year, team)
        df = clean_schedule_df(raw, year)
    except Exception as exc:
        log_audit(
            f'pybaseball schedule_and_record failed for {team} {year}: {exc}. '
            'Falling back to BBRef direct scrape.', 'WARNING'
        )

    if df is None or len(df) == 0:
        df = _scrape_direct(year, team)

    _cache[key] = df
    return df


def _scrape_direct(year, team):
    url = f'https://www.baseball-reference.com/teams/{team}/{year}-schedule.shtml'
    resp = bbref_get(url)
    if not resp:
        log_audit(f'BBRef schedule scrape failed for {team} {year}', 'ERROR')
        return pd.DataFrame()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', id='team_schedule')
    if table is None:
        log_audit(f'No #team_schedule table for {team} {year}', 'ERROR')
        return pd.DataFrame()
    try:
        df_raw = pd.read_html(str(table))[0]
        return clean_schedule_df(df_raw, year)
    except Exception as exc:
        log_audit(f'Parse error for {team} {year} schedule: {exc}', 'ERROR')
        return pd.DataFrame()


# ── Extract pre / post windows ─────────────────────────────────────────────────
def extract_windows(gl, firing_id, game_number_at_firing, post_games_available,
                    truncated_window):
    """
    From a full season game log, extract pre and post window rows.

    Pre  : all games where game_number <= game_number_at_firing
    Post : games where game_number > game_number_at_firing,
           up to post_games_available rows (respects truncation).
    """
    rows = []
    gl = gl.copy().sort_values('game_number').reset_index(drop=True)

    pre = gl[gl['game_number'] <= game_number_at_firing].copy()
    post_all = gl[gl['game_number'] > game_number_at_firing].copy()
    post = post_all.head(int(post_games_available))

    for window_label, subset in [('pre', pre), ('post', post)]:
        for seq_num, (_, row) in enumerate(subset.iterrows(), start=1):
            rows.append({
                'firing_id':    firing_id,
                'game_date':    row.get('game_date'),
                'game_number':  seq_num if window_label == 'post' else int(row['game_number']),
                'window':       window_label,
                'runs_scored':  row.get('runs_scored'),
                'runs_allowed': row.get('runs_allowed'),
                'result':       row.get('result'),
            })

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    print("Phase 2: Pulling game-level data ...")
    log_audit('='*60, 'PHASE2_START')

    if not os.path.exists(EVENT_TABLE):
        print(f"  ERROR: {EVENT_TABLE} not found. Run Phase 1 first.")
        return

    events = pd.read_csv(EVENT_TABLE)
    print(f"  Loaded {len(events)} events from event_table.csv")

    all_rows = []
    missing_logs = 0

    # Sort by season+team to maximise cache hits
    events = events.sort_values(['season', 'team']).reset_index(drop=True)

    for i, ev in events.iterrows():
        fid   = int(ev['firing_id'])
        team  = str(ev['team'])
        year  = int(ev['season'])
        gnum  = int(ev['game_number_at_firing'])
        post_g = int(ev['post_games_available'])
        trunc  = int(ev['truncated_window'])

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing event {i+1}/{len(events)} "
                  f"({ev['manager_fired']}, {team} {year}) ...")

        gl = _get_raw_log(year, team)

        if gl is None or len(gl) == 0:
            log_audit(
                f'MISSING game log for firing_id={fid} {ev["manager_fired"]} '
                f'{team} {year}. Rows not added to game_log.csv.', 'ERROR'
            )
            missing_logs += 1
            continue

        # Validate expected pre game count against log
        actual_pre = gl[gl['game_number'] <= gnum]
        if len(actual_pre) != int(ev['pre_games']):
            log_audit(
                f'MISMATCH firing_id={fid} {team} {year}: Lahman pre_games='
                f'{ev["pre_games"]} but game log has {len(actual_pre)} rows '
                f'<= game#{gnum}. Using game log count.', 'WARNING'
            )

        window_rows = extract_windows(gl, fid, gnum, post_g, trunc)
        all_rows.extend(window_rows)

    df_out = pd.DataFrame(all_rows)

    if len(df_out) == 0:
        print("  WARNING: game_log.csv is empty. Check data_audit.txt.")
    else:
        # Type coercion
        df_out['game_date']    = pd.to_datetime(df_out['game_date'], errors='coerce')
        df_out['runs_scored']  = pd.to_numeric(df_out['runs_scored'],  errors='coerce')
        df_out['runs_allowed'] = pd.to_numeric(df_out['runs_allowed'], errors='coerce')

    df_out.to_csv(OUT_PATH, index=False)
    n = len(df_out)
    print(f"Phase 2 complete — {n} records written to game_log.csv "
          f"({missing_logs} events had no retrievable game log)")
    log_audit(
        f'Phase 2 complete. {n} game rows written. '
        f'{missing_logs} events had no game log.', 'PHASE2_END'
    )


if __name__ == '__main__':
    run()
