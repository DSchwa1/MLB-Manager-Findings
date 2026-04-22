"""Phase 8: Robustness Checks.

Re-runs the primary regression four times with modified sample or window:
  1. Excluding truncated_window == 1 cases
  2. Filtering to pre_games >= 15
  3. Using a 30-game post window instead of 40
  4. Using a 50-game post window instead of 40

Reports whether β1 holds direction and significance across all checks.

Output: outputs/robustness_checks.txt
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import pythagorean_wpct, log_audit, OUTPUT_DIR

CTRL_TABLE   = os.path.join(OUTPUT_DIR, 'control_table.csv')
CTRL_POOL    = os.path.join(OUTPUT_DIR, 'control_pool.csv')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics_table.csv')
PROJ_PATH    = os.path.join(OUTPUT_DIR, 'projections_table.csv')
EVENT_TABLE  = os.path.join(OUTPUT_DIR, 'event_table.csv')
GAME_LOG     = os.path.join(OUTPUT_DIR, 'game_log.csv')
OUT_PATH     = os.path.join(OUTPUT_DIR, 'robustness_checks.txt')

PRIMARY_PREDICTORS = (
    'fired', 'projection_residual_pre', 'pyth_gap_at_firing', 'game_number_at_firing'
)


def run_ols_safe(df, label):
    """OLS with NA removal. Returns (result, n_used, warn_msg)."""
    cols   = ['pyth_delta'] + list(PRIMARY_PREDICTORS)
    subset = df[[c for c in cols if c in df.columns]].dropna()
    missing = [c for c in cols if c not in df.columns]

    if missing:
        return None, 0, f'Missing columns: {missing}'
    if len(subset) < len(PRIMARY_PREDICTORS) + 5:
        return None, len(subset), f'Insufficient data: {len(subset)} rows'

    y = subset['pyth_delta']
    X = sm.add_constant(subset[list(PRIMARY_PREDICTORS)])
    result = sm.OLS(y, X).fit()
    return result, len(subset), None


def recompute_pyth_delta_for_window(game_log, ctrl_pool, events, window_games):
    """
    Recompute pyth_delta for a different post-window length.

    For firing events: re-slice game_log post windows to window_games.
    For controls: re-slice ctrl_pool pseudo-post windows.

    Returns (fired_df, ctrl_df) with updated pyth_delta.
    """
    updated_fired = []
    for _, ev in events.iterrows():
        fid   = int(ev['firing_id'])
        gnum  = int(ev['game_number_at_firing'])
        trunc = int(ev.get('truncated_window', 0))

        ev_log = game_log[game_log['firing_id'] == fid]
        pre    = ev_log[ev_log['window'] == 'pre']
        post_all = ev_log[ev_log['window'] == 'post']

        # Re-truncate post to window_games
        post = post_all.head(window_games)

        if len(post) < 25:
            continue  # still enforce minimum

        pre_pyth  = pythagorean_wpct(pre['runs_scored'].sum(), pre['runs_allowed'].sum())
        post_pyth = pythagorean_wpct(post['runs_scored'].sum(), post['runs_allowed'].sum())
        pyth_delta = post_pyth - pre_pyth if not np.isnan(pre_pyth) and not np.isnan(post_pyth) else np.nan

        updated_fired.append({'firing_id': fid, 'pyth_delta_rob': pyth_delta})

    fired_upd = pd.DataFrame(updated_fired)

    # Update ctrl_pool post windows (slice to window_games)
    updated_ctrl = []
    if ctrl_pool is not None and len(ctrl_pool) > 0:
        from phase2_game_logs import _get_raw_log
        done = set()
        for _, row in ctrl_pool.iterrows():
            cid    = int(row['ctrl_id'])
            team   = str(row['ctrl_team'])
            season = int(row['ctrl_season'])
            pgnum  = int(row['pseudo_game_number'])
            key    = (team, season, pgnum)
            if key in done:
                continue
            done.add(key)

            gl = _get_raw_log(season, team)
            if gl is None or len(gl) == 0:
                continue

            pre  = gl[gl['game_number'] <= pgnum]
            post = gl[(gl['game_number'] > pgnum)
                      & (gl['game_number'] <= pgnum + window_games)]
            if len(post) < 25:
                continue

            pre_pyth  = pythagorean_wpct(pre['runs_scored'].sum(), pre['runs_allowed'].sum())
            post_pyth = pythagorean_wpct(post['runs_scored'].sum(), post['runs_allowed'].sum())
            pd_val    = post_pyth - pre_pyth if not np.isnan(pre_pyth) and not np.isnan(post_pyth) else np.nan

            updated_ctrl.append({
                'ctrl_id':      cid,
                'pyth_delta_rob': pd_val,
                **{k: row[k] for k in ('projection_residual_pre',
                                        'pyth_gap_at_pseudo_fire',
                                        'pseudo_game_number')
                   if k in row.index},
            })

    ctrl_upd = pd.DataFrame(updated_ctrl)
    return fired_upd, ctrl_upd


def build_check_df(ctrl_table, ctrl_pool, metrics, projections,
                   exclude_truncated=False,
                   min_pre_games=None,
                   pyth_delta_override_fired=None,
                   pyth_delta_override_ctrl=None):
    """
    Build the regression dataframe for one robustness check.

    pyth_delta_override_fired / _ctrl: DataFrames with firing_id/ctrl_id and
    pyth_delta_rob column (for window-length checks).
    """
    # Start with matched fired events
    fired = ctrl_table[ctrl_table['match_found'] == 1].copy()
    fired['fired'] = 1

    if exclude_truncated:
        fired = fired[fired['truncated_window'] == 0]

    if min_pre_games is not None:
        # pre_games is in metrics
        pre_g = metrics[['firing_id', 'pre_games']].copy()
        fired = fired.merge(pre_g, on='firing_id', how='left')
        fired = fired[pd.to_numeric(fired['pre_games'], errors='coerce') >= min_pre_games]

    # Override pyth_delta if window-length check
    if pyth_delta_override_fired is not None and len(pyth_delta_override_fired) > 0:
        fired = fired.merge(pyth_delta_override_fired, on='firing_id', how='inner')
        fired['pyth_delta'] = fired['pyth_delta_rob']

    # Build control rows
    ctrl_df = pd.DataFrame()
    if ctrl_pool is not None and len(ctrl_pool) > 0 and 'ctrl_id' in fired.columns:
        ctrl_ids = fired['ctrl_id'].dropna().astype(int).unique()
        ctrl_df = ctrl_pool[ctrl_pool['ctrl_id'].isin(ctrl_ids)].copy()
        ctrl_df = ctrl_df.rename(columns={
            'pseudo_game_number':      'game_number_at_firing',
            'projection_residual_pre': 'projection_residual_pre',
            'pyth_gap_at_pseudo_fire': 'pyth_gap_at_firing',
        })
        ctrl_df['fired'] = 0

        if pyth_delta_override_ctrl is not None and len(pyth_delta_override_ctrl) > 0:
            ctrl_df = ctrl_df.merge(
                pyth_delta_override_ctrl[['ctrl_id', 'pyth_delta_rob']],
                on='ctrl_id', how='inner'
            )
            ctrl_df['pyth_delta'] = ctrl_df['pyth_delta_rob']

    combined = pd.concat([fired, ctrl_df], ignore_index=True, sort=False)
    for col in PRIMARY_PREDICTORS + ('pyth_delta',):
        combined[col] = pd.to_numeric(combined.get(col), errors='coerce')

    return combined


def run():
    print("Phase 8: Running robustness checks ...")
    log_audit('='*60, 'PHASE8_START')

    for path, label in [
        (CTRL_TABLE,   'control_table.csv'),
        (METRICS_PATH, 'metrics_table.csv'),
        (PROJ_PATH,    'projections_table.csv'),
        (EVENT_TABLE,  'event_table.csv'),
        (GAME_LOG,     'game_log.csv'),
    ]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} not found.")
            return

    ctrl_table  = pd.read_csv(CTRL_TABLE)
    metrics     = pd.read_csv(METRICS_PATH)
    projections = pd.read_csv(PROJ_PATH)
    events      = pd.read_csv(EVENT_TABLE)
    game_log    = pd.read_csv(GAME_LOG)
    ctrl_pool   = pd.read_csv(CTRL_POOL) if os.path.exists(CTRL_POOL) else pd.DataFrame()

    game_log['runs_scored']  = pd.to_numeric(game_log['runs_scored'],  errors='coerce')
    game_log['runs_allowed'] = pd.to_numeric(game_log['runs_allowed'], errors='coerce')

    results_summary = []

    checks = [
        # (label, kwargs_for_build_check_df, needs_window_recompute, window_size)
        ('Check 1: Exclude truncated_window==1',
         dict(exclude_truncated=True), False, None),
        ('Check 2: Restrict to pre_games >= 15',
         dict(min_pre_games=15), False, None),
        ('Check 3: 30-game post window',
         dict(), True, 30),
        ('Check 4: 50-game post window',
         dict(), True, 50),
    ]

    output_blocks = [
        'MLB Manager Firings — Robustness Checks',
        '='*70,
        '',
        'Primary model:',
        '  pyth_delta = β0 + β1(fired) + β2(projection_residual_pre)',
        '             + β3(pyth_gap_at_firing) + β4(game_number_at_firing) + ε',
        '',
    ]

    for label, kwargs, needs_recompute, window in checks:
        print(f"  Running: {label} ...")

        if needs_recompute and window is not None:
            print(f"    Recomputing pyth_delta for {window}-game post window ...")
            fired_ovr, ctrl_ovr = recompute_pyth_delta_for_window(
                game_log, ctrl_pool, events, window
            )
            kwargs['pyth_delta_override_fired'] = fired_ovr
            kwargs['pyth_delta_override_ctrl']  = ctrl_ovr

        df = build_check_df(ctrl_table, ctrl_pool, metrics, projections, **kwargs)
        n_fired = int((df['fired'] == 1).sum()) if 'fired' in df.columns else 0
        n_ctrl  = int((df['fired'] == 0).sum()) if 'fired' in df.columns else 0

        result, n_used, warn = run_ols_safe(df, label)

        block = [
            '─'*70,
            f'{label}',
            f'  N fired={n_fired}, N ctrl={n_ctrl}, N used in model={n_used}',
            '',
        ]

        if warn:
            block += [f'  WARNING: {warn}', '']
            results_summary.append({'check': label, 'b1': np.nan, 'p1': np.nan,
                                    'dir_holds': 'N/A', 'sig': 'N/A'})
        else:
            b1   = result.params.get('fired', np.nan)
            se1  = result.bse.get('fired', np.nan)
            p1   = result.pvalues.get('fired', np.nan)
            r2   = result.rsquared
            block += [
                result.summary().as_text(), '',
                f'  β1(fired) = {b1:+.4f}  SE={se1:.4f}  p={p1:.4f}  R²={r2:.4f}',
                '',
            ]
            dir_holds = 'YES' if b1 > 0 else 'NO'
            sig       = 'YES (p<0.05)' if p1 < 0.05 else f'NO (p={p1:.3f})'
            results_summary.append({'check': label, 'b1': b1, 'p1': p1,
                                    'dir_holds': dir_holds, 'sig': sig})
            log_audit(
                f'Robustness {label}: β1={b1:.4f}, p={p1:.4f}', 'PHASE8'
            )

        output_blocks.extend(block)

    # Summary table
    output_blocks += ['', '='*70, 'ROBUSTNESS SUMMARY', '='*70, '']
    output_blocks.append(f'  {"Check":<45} {"β1":>8} {"p":>8} {"Dir?":>6} {"Sig?":>14}')
    output_blocks.append('  ' + '-'*82)
    for r in results_summary:
        b1_str = f'{r["b1"]:+.4f}' if not np.isnan(r['b1']) else '  N/A  '
        p_str  = f'{r["p1"]:.4f}'  if not np.isnan(r['p1']) else '  N/A  '
        output_blocks.append(
            f'  {r["check"]:<45} {b1_str:>8} {p_str:>8} {r["dir_holds"]:>6} {r["sig"]:>14}'
        )
    output_blocks.append('')

    # Overall verdict
    all_pos = all(r['dir_holds'] == 'YES' for r in results_summary if r['dir_holds'] != 'N/A')
    output_blocks.append(f'β1 holds direction across all checks: {"YES" if all_pos else "NO"}')

    with open(OUT_PATH, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(output_blocks))

    print(f"Phase 8 complete — robustness results written to robustness_checks.txt")
    log_audit(
        f'Phase 8 complete. Direction holds: {all_pos}. '
        f'Results: {results_summary}', 'PHASE8_END'
    )


if __name__ == '__main__':
    run()
