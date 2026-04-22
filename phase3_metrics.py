"""Phase 3: Calculate Core Metrics.

Reads game_log.csv and event_table.csv.
For each firing event, calculates per-game Pythagorean W%, actual W%,
run differential, and related deltas.

Outputs: outputs/metrics_table.csv
"""

import os
import numpy as np
import pandas as pd

from utils import (
    pythagorean_wpct, log_audit, OUTPUT_DIR
)

EVENT_TABLE = os.path.join(OUTPUT_DIR, 'event_table.csv')
GAME_LOG    = os.path.join(OUTPUT_DIR, 'game_log.csv')
OUT_PATH    = os.path.join(OUTPUT_DIR, 'metrics_table.csv')


def window_metrics(sub):
    """
    Compute summary stats for a single window (pre or post) from a subset of
    game_log rows. Returns a dict of metric values.
    """
    if sub is None or len(sub) == 0:
        nans = {k: np.nan for k in (
            'n_games', 'wins', 'losses', 'actual_wpct',
            'total_rs', 'total_ra', 'pyth_wpct',
            'run_diff_per_game'
        )}
        return nans

    wins   = (sub['result'] == 'W').sum()
    losses = (sub['result'] == 'L').sum()
    n      = wins + losses  # ties excluded from W% calc

    actual_wpct = wins / n if n > 0 else np.nan

    rs_total = sub['runs_scored'].sum()
    ra_total = sub['runs_allowed'].sum()
    pyth     = pythagorean_wpct(rs_total, ra_total)
    rd_pg    = (rs_total - ra_total) / len(sub) if len(sub) > 0 else np.nan

    return {
        'n_games':           len(sub),
        'wins':              wins,
        'losses':            losses,
        'actual_wpct':       actual_wpct,
        'total_rs':          rs_total,
        'total_ra':          ra_total,
        'pyth_wpct':         pyth,
        'run_diff_per_game': rd_pg,
    }


def run():
    print("Phase 3: Calculating core metrics ...")
    log_audit('='*60, 'PHASE3_START')

    for path, label in [(EVENT_TABLE, 'event_table.csv'),
                        (GAME_LOG, 'game_log.csv')]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} not found. Run earlier phases first.")
            return

    events   = pd.read_csv(EVENT_TABLE)
    game_log = pd.read_csv(GAME_LOG)

    # Coerce types
    game_log['runs_scored']  = pd.to_numeric(game_log['runs_scored'],  errors='coerce')
    game_log['runs_allowed'] = pd.to_numeric(game_log['runs_allowed'], errors='coerce')

    records = []

    for _, ev in events.iterrows():
        fid = int(ev['firing_id'])
        ev_log = game_log[game_log['firing_id'] == fid]

        pre_sub  = ev_log[ev_log['window'] == 'pre']
        post_sub = ev_log[ev_log['window'] == 'post']

        pre_m  = window_metrics(pre_sub)
        post_m = window_metrics(post_sub)

        # pyth_gap_at_firing = actual_wpct_pre - pyth_wpct_pre
        # (positive = overperforming Pythagorean expectation going in)
        pyth_gap = (pre_m['actual_wpct'] - pre_m['pyth_wpct']
                    if not np.isnan(pre_m['actual_wpct']) and not np.isnan(pre_m['pyth_wpct'])
                    else np.nan)

        pyth_delta = (post_m['pyth_wpct'] - pre_m['pyth_wpct']
                      if not np.isnan(post_m['pyth_wpct']) and not np.isnan(pre_m['pyth_wpct'])
                      else np.nan)

        rd_delta = (post_m['run_diff_per_game'] - pre_m['run_diff_per_game']
                    if not np.isnan(post_m['run_diff_per_game'])
                       and not np.isnan(pre_m['run_diff_per_game'])
                    else np.nan)

        if np.isnan(pyth_delta):
            log_audit(
                f'WARNING firing_id={fid}: pyth_delta is NaN '
                f'(pre_pyth={pre_m["pyth_wpct"]:.4f if not np.isnan(pre_m["pyth_wpct"]) else "NaN"}, '
                f'post_pyth={post_m["pyth_wpct"]:.4f if not np.isnan(post_m["pyth_wpct"]) else "NaN"})',
                'WARNING'
            )

        rec = {
            'firing_id':                fid,
            'team':                     ev['team'],
            'manager_fired':            ev['manager_fired'],
            'season':                   ev['season'],
            'pre_games':                pre_m['n_games'],
            'post_games':               post_m['n_games'],
            # Pre-window
            'actual_wpct_pre':          pre_m['actual_wpct'],
            'pyth_wpct_pre':            pre_m['pyth_wpct'],
            'run_diff_per_game_pre':    pre_m['run_diff_per_game'],
            'rs_total_pre':             pre_m['total_rs'],
            'ra_total_pre':             pre_m['total_ra'],
            # Post-window
            'actual_wpct_post':         post_m['actual_wpct'],
            'pyth_wpct_post':           post_m['pyth_wpct'],
            'run_diff_per_game_post':   post_m['run_diff_per_game'],
            'rs_total_post':            post_m['total_rs'],
            'ra_total_post':            post_m['total_ra'],
            # Deltas
            'pyth_delta':               pyth_delta,
            'run_diff_delta':           rd_delta,
            # Regression-pressure indicator
            'pyth_gap_at_firing':       pyth_gap,
            # From event table
            'truncated_window':         int(ev['truncated_window']),
            'game_number_at_firing':    int(ev['game_number_at_firing']),
            'is_outsider':              int(ev['is_outsider']),
        }
        records.append(rec)

    df_out = pd.DataFrame(records)
    df_out = df_out.sort_values(['season', 'team']).reset_index(drop=True)
    df_out.to_csv(OUT_PATH, index=False)

    n = len(df_out)
    n_nan = df_out['pyth_delta'].isna().sum()
    print(f"Phase 3 complete — {n} records written to metrics_table.csv "
          f"({n_nan} events have NaN pyth_delta; see data_audit.txt)")
    log_audit(
        f'Phase 3 complete. {n} metrics records. {n_nan} with NaN pyth_delta.',
        'PHASE3_END'
    )


if __name__ == '__main__':
    run()
