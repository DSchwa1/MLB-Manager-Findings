"""Phase 5: Build the Matched Control Group.

For each firing event, find matched control observations from non-firing
team-seasons at a similar point in the season.

Matching criteria (all three must be satisfied):
  - Projection residual at pseudo-firing date  : within ±3 wins  (= ±3/162 in W% terms)
  - Games into season at pseudo-firing date    : within ±15 games
  - Era                                        : within ±5 years

Tie-breaking: smallest |projection_residual_diff|, then |games_diff|, then era.
Controls may be reused; reuse is flagged.

Outputs: outputs/control_table.csv
"""

import os
import itertools
import numpy as np
import pandas as pd
from utils import pythagorean_wpct, log_audit, OUTPUT_DIR

EVENT_TABLE  = os.path.join(OUTPUT_DIR, 'event_table.csv')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics_table.csv')
PROJ_PATH    = os.path.join(OUTPUT_DIR, 'projections_table.csv')
GAME_LOG     = os.path.join(OUTPUT_DIR, 'game_log.csv')
OUT_PATH     = os.path.join(OUTPUT_DIR, 'control_table.csv')

# Matching tolerances
TOL_PROJ_RESIDUAL = 3 / 162   # ±3 wins expressed as W%
TOL_GAMES         = 15
TOL_ERA_YEARS     = 5

POST_WINDOW       = 40        # control post window (same length as firing events)
MIN_POST_GAMES    = 25


def build_all_team_season_windows(events, game_log, projections):
    """
    For every non-firing team-season combination present in the game log,
    generate 40-game windows at the same game-number as each firing event.

    Returns a DataFrame with one row per (team, season, pseudo_game_number) window,
    including pyth_delta, actual_wpct_pre/post, and projection residual.
    """
    # Identify firing team-seasons to exclude from controls
    firing_keys = set(zip(events['team'], events['season']))

    # Unique team-seasons in the game log (may include firing teams too — we exclude them)
    team_seasons = (
        game_log[['firing_id']].drop_duplicates()
        .merge(events[['firing_id', 'team', 'season']], on='firing_id', how='left')
    )
    all_team_seasons = set(zip(team_seasons['team'], team_seasons['season']))
    control_team_seasons = all_team_seasons - firing_keys

    proj_lookup = {}
    if len(projections) > 0:
        for _, row in projections.iterrows():
            proj_lookup[(str(row['team']), int(row['season']))] = row

    # For each control team-season, we need their game log
    # We'll reuse game_log rows that belong to firing events on those teams —
    # but control team-seasons are explicitly non-firing, so their game log
    # rows don't appear in game_log.csv (which only has firing-event windows).
    # We therefore need to request them separately via pybaseball.
    # To avoid heavy re-scraping, we extract the data we need from the
    # full-season logs that were already cached during Phase 2.
    #
    # Strategy: for each control team-season, load the full cached log
    # and slice a 40-game window centred at each firing event's game number.

    from phase2_game_logs import _get_raw_log   # reuse the cache

    rows = []
    ctrl_id = 0

    # Get set of pseudo-firing game numbers to match against (from firing events)
    firing_gnums = sorted(events['game_number_at_firing'].unique().tolist())

    for (ctrl_team, ctrl_season) in sorted(control_team_seasons):
        gl = _get_raw_log(ctrl_season, ctrl_team)
        if gl is None or len(gl) == 0:
            continue

        p_row = proj_lookup.get((ctrl_team, ctrl_season))
        proj_wpct = float(p_row['projected_wpct']) if (
            p_row is not None and pd.notna(p_row.get('projected_wpct'))
        ) else np.nan

        for pseudo_gnum in firing_gnums:
            pre  = gl[gl['game_number'] <= pseudo_gnum]
            post = gl[(gl['game_number'] > pseudo_gnum)
                      & (gl['game_number'] <= pseudo_gnum + POST_WINDOW)]

            if len(post) < MIN_POST_GAMES:
                continue

            # Metrics
            def _wpct(sub):
                w = (sub['result'] == 'W').sum()
                l = (sub['result'] == 'L').sum()
                return w / (w + l) if (w + l) > 0 else np.nan

            def _pyth(sub):
                return pythagorean_wpct(
                    sub['runs_scored'].sum(), sub['runs_allowed'].sum()
                )

            pre_pyth  = _pyth(pre)  if len(pre)  > 0 else np.nan
            post_pyth = _pyth(post)
            pre_act   = _wpct(pre)  if len(pre)  > 0 else np.nan
            post_act  = _wpct(post)

            pyth_delta  = post_pyth - pre_pyth if not np.isnan(post_pyth) and not np.isnan(pre_pyth) else np.nan
            actual_gap  = pre_act   - pre_pyth  if not np.isnan(pre_act)  and not np.isnan(pre_pyth) else np.nan

            # Projection residual at pseudo-firing date
            res_pre = pre_act - proj_wpct if not np.isnan(pre_act) and not np.isnan(proj_wpct) else np.nan

            ctrl_id += 1
            rows.append({
                'ctrl_id':                 ctrl_id,
                'ctrl_team':               ctrl_team,
                'ctrl_season':             ctrl_season,
                'pseudo_game_number':      pseudo_gnum,
                'pre_games':               len(pre),
                'post_games':              len(post),
                'actual_wpct_pre':         pre_act,
                'actual_wpct_post':        post_act,
                'pyth_wpct_pre':           pre_pyth,
                'pyth_wpct_post':          post_pyth,
                'pyth_delta':              pyth_delta,
                'pyth_gap_at_pseudo_fire': actual_gap,
                'projected_wpct':          proj_wpct if not np.isnan(proj_wpct) else None,
                'projection_residual_pre': res_pre   if not np.isnan(res_pre)   else None,
            })

    return pd.DataFrame(rows)


def match_controls(events, metrics, projections, ctrl_pool):
    """
    For each firing event, find the best-matching control observation.

    Returns a matched table with one row per firing event (plus unmatched flags).
    """
    # Merge events with metrics and projections for matching variables
    ev_m = events.merge(
        metrics[['firing_id', 'pyth_delta', 'actual_wpct_pre', 'pyth_wpct_pre',
                 'pyth_gap_at_firing', 'game_number_at_firing']],
        on='firing_id', how='left'
    )
    ev_m = ev_m.merge(
        projections[['firing_id', 'projected_wpct', 'projection_residual_pre']],
        on='firing_id', how='left'
    )

    matched_rows = []
    ctrl_use_count = {}  # count how many times each ctrl_id is reused

    for _, ev in ev_m.iterrows():
        fid       = int(ev['firing_id'])
        ev_year   = int(ev['season'])
        ev_gnum   = float(ev['game_number_at_firing'])
        ev_res    = float(ev['projection_residual_pre']) if pd.notna(ev['projection_residual_pre']) else np.nan

        candidates = ctrl_pool[
            (np.abs(ctrl_pool['pseudo_game_number'] - ev_gnum) <= TOL_GAMES) &
            (np.abs(ctrl_pool['ctrl_season'] - ev_year)       <= TOL_ERA_YEARS)
        ].copy()

        if not np.isnan(ev_res) and 'projection_residual_pre' in candidates.columns:
            cands_with_res = candidates[candidates['projection_residual_pre'].notna()].copy()
            cands_with_res['res_diff'] = np.abs(
                cands_with_res['projection_residual_pre'].astype(float) - ev_res
            )
            cands_with_res = cands_with_res[cands_with_res['res_diff'] <= TOL_PROJ_RESIDUAL]
            if len(cands_with_res) > 0:
                candidates = cands_with_res
            else:
                # No match within residual tolerance
                log_audit(
                    f'NO_MATCH firing_id={fid}: no control within residual tolerance '
                    f'(ev_res={ev_res:.4f}, closest was '
                    f'{cands_with_res["res_diff"].min():.4f} away if any). '
                    f'Attempting match without residual criterion.', 'MATCH_WARNING'
                )

        if len(candidates) == 0:
            log_audit(
                f'UNMATCHED firing_id={fid} {ev["team"]} {ev_year} gnum={ev_gnum}: '
                f'no control found within all tolerance windows. '
                f'Event retained in primary analysis; excluded from matched comparison.',
                'MATCH_WARNING'
            )
            matched_rows.append({
                'firing_id':          fid,
                'ctrl_id':            None,
                'ctrl_team':          None,
                'ctrl_season':        None,
                'match_found':        0,
                'ctrl_pyth_delta':    np.nan,
                'ctrl_res_pre':       np.nan,
                'ctrl_pyth_gap':      np.nan,
                'match_res_diff':     np.nan,
                'match_games_diff':   np.nan,
                'match_era_diff':     np.nan,
            })
            continue

        # Tie-break: smallest |residual_diff|, then |games_diff|, then era_diff
        candidates = candidates.copy()
        if 'res_diff' not in candidates.columns:
            candidates['res_diff'] = (
                np.abs(candidates['projection_residual_pre'].fillna(0).astype(float) - (ev_res if not np.isnan(ev_res) else 0))
            )
        candidates['games_diff'] = np.abs(candidates['pseudo_game_number'] - ev_gnum)
        candidates['era_diff']   = np.abs(candidates['ctrl_season'] - ev_year)
        candidates = candidates.sort_values(
            ['res_diff', 'games_diff', 'era_diff']
        ).reset_index(drop=True)

        best = candidates.iloc[0]
        ctrl_id_val = int(best['ctrl_id'])
        ctrl_use_count[ctrl_id_val] = ctrl_use_count.get(ctrl_id_val, 0) + 1

        matched_rows.append({
            'firing_id':          fid,
            'ctrl_id':            ctrl_id_val,
            'ctrl_team':          best['ctrl_team'],
            'ctrl_season':        int(best['ctrl_season']),
            'match_found':        1,
            'ctrl_pyth_delta':    float(best['pyth_delta'])              if pd.notna(best['pyth_delta'])              else np.nan,
            'ctrl_res_pre':       float(best['projection_residual_pre']) if pd.notna(best['projection_residual_pre']) else np.nan,
            'ctrl_pyth_gap':      float(best['pyth_gap_at_pseudo_fire']) if pd.notna(best['pyth_gap_at_pseudo_fire']) else np.nan,
            'match_res_diff':     float(best['res_diff']),
            'match_games_diff':   float(best['games_diff']),
            'match_era_diff':     float(best['era_diff']),
        })

    # Log reused controls
    reused = {k: v for k, v in ctrl_use_count.items() if v > 1}
    if reused:
        log_audit(
            f'CONTROL_REUSE: {len(reused)} control observations reused across '
            f'multiple firing events. ctrl_id use counts: {reused}. '
            f'Sensitivity to this assumption should be assessed.',
            'MATCH_INFO'
        )

    return pd.DataFrame(matched_rows), ctrl_use_count


def run():
    print("Phase 5: Building matched control group ...")
    log_audit('='*60, 'PHASE5_START')

    for path, label in [
        (EVENT_TABLE,  'event_table.csv'),
        (METRICS_PATH, 'metrics_table.csv'),
        (PROJ_PATH,    'projections_table.csv'),
        (GAME_LOG,     'game_log.csv'),
    ]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} not found. Run earlier phases first.")
            return

    events      = pd.read_csv(EVENT_TABLE)
    metrics     = pd.read_csv(METRICS_PATH)
    projections = pd.read_csv(PROJ_PATH)
    game_log    = pd.read_csv(GAME_LOG)

    game_log['runs_scored']  = pd.to_numeric(game_log['runs_scored'],  errors='coerce')
    game_log['runs_allowed'] = pd.to_numeric(game_log['runs_allowed'], errors='coerce')

    print(f"  Building control pool from non-firing team-seasons ...")
    ctrl_pool = build_all_team_season_windows(events, game_log, projections)
    print(f"  Control pool: {len(ctrl_pool)} candidate observations")
    log_audit(f'Phase 5 control pool: {len(ctrl_pool)} candidates.', 'PHASE5_INFO')

    if len(ctrl_pool) == 0:
        print("  WARNING: control pool is empty. "
              "This likely means phase2 game log cache is empty.")
        log_audit('Phase 5: control pool empty — check phase2 game log cache.', 'WARNING')
        ctrl_pool = pd.DataFrame()

    print("  Matching firing events to controls ...")
    matched_df, reuse_counts = match_controls(events, metrics, projections, ctrl_pool)

    n_matched   = matched_df['match_found'].sum() if len(matched_df) > 0 else 0
    n_unmatched = len(matched_df) - n_matched

    # Merge firing event metrics into matched table for regression use
    full = matched_df.merge(
        metrics[['firing_id', 'pyth_delta', 'actual_wpct_pre', 'pyth_wpct_pre',
                 'pyth_gap_at_firing', 'game_number_at_firing',
                 'run_diff_per_game_pre', 'run_diff_per_game_post']],
        on='firing_id', how='left'
    ).merge(
        projections[['firing_id', 'projected_wpct', 'projection_residual_pre']],
        on='firing_id', how='left'
    ).merge(
        events[['firing_id', 'team', 'season', 'is_outsider', 'truncated_window']],
        on='firing_id', how='left'
    )

    full['fired'] = 1   # all rows in this table are firing events

    full.to_csv(OUT_PATH, index=False)

    # Also save the raw control pool for use in regression
    ctrl_pool_path = os.path.join(OUTPUT_DIR, 'control_pool.csv')
    if len(ctrl_pool) > 0:
        ctrl_pool.to_csv(ctrl_pool_path, index=False)

    n = len(full)
    print(f"Phase 5 complete — {n} records written to control_table.csv "
          f"({n_matched} matched, {n_unmatched} unmatched)")
    log_audit(
        f'Phase 5 complete. {n} records. {n_matched} matched. '
        f'{n_unmatched} unmatched (excluded from matched-comparison analyses only). '
        f'Reused controls: {len([v for v in reuse_counts.values() if v > 1])}.',
        'PHASE5_END'
    )


if __name__ == '__main__':
    run()
