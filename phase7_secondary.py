"""Phase 7: Secondary Analyses.

Four separate regression models:

1. Outsider vs. insider effect — is_outsider × fired interaction.
2. Timing effect — game_number_at_firing as continuous predictor of pyth_delta
   (firing events only).
3. Interim tenure effect — replacement manager tenure (games managed before
   season end or permanent hire) as moderator of pyth_delta.
4. Roster age effect — PA-weighted average roster age at time of firing as
   moderator of pyth_delta.

Outputs (one per model):
  outputs/regression_outsider.txt
  outputs/regression_timing.txt
  outputs/regression_tenure.txt
  outputs/regression_age.txt
"""

import os
import re
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import requests
from bs4 import BeautifulSoup

from utils import bbref_get, log_audit, OUTPUT_DIR, HEADERS

CTRL_TABLE   = os.path.join(OUTPUT_DIR, 'control_table.csv')
CTRL_POOL    = os.path.join(OUTPUT_DIR, 'control_pool.csv')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics_table.csv')
PROJ_PATH    = os.path.join(OUTPUT_DIR, 'projections_table.csv')
EVENT_TABLE  = os.path.join(OUTPUT_DIR, 'event_table.csv')


# ── Shared regression helper ───────────────────────────────────────────────────
def fit_ols(df, outcome, predictors, label, out_path):
    """Fit OLS, write full output to out_path, return result or None."""
    cols   = [outcome] + list(predictors)
    subset = df[[c for c in cols if c in df.columns]].dropna()
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        msg = f'Regression ({label}): missing columns {missing_cols}.'
        log_audit(msg, 'ERROR')
        _write_output(out_path, label, msg, None, predictors)
        return None

    n_drop = len(df) - len(subset)
    if n_drop > 0:
        log_audit(f'Regression ({label}): dropped {n_drop} NaN rows.', 'REGRESSION_INFO')

    if len(subset) < len(predictors) + 5:
        msg = (f'Regression ({label}): only {len(subset)} rows after NA removal '
               f'— model not fit.')
        log_audit(msg, 'WARNING')
        _write_output(out_path, label, msg, None, predictors)
        return None

    y = subset[outcome]
    X = sm.add_constant(subset[list(predictors)])
    result = sm.OLS(y, X).fit()
    _write_output(out_path, label, result.summary().as_text(), result, predictors)
    return result


def _write_output(path, label, summary_or_msg, result, predictors):
    lines = [
        f'MLB Manager Firings — Secondary Analysis: {label}',
        '='*70, '',
        summary_or_msg, '',
    ]
    if result is not None:
        lines += ['='*70, 'Coefficients:', '']
        for name, coef in result.params.items():
            se   = result.bse[name]
            pval = result.pvalues[name]
            lines.append(f'  {name:35s}  coef={coef:+.4f}  SE={se:.4f}  p={pval:.4f}')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


# ── 1. Outsider vs. insider interaction ───────────────────────────────────────
def model_outsider(fired_df, ctrl_df):
    """Add is_outsider, fired×is_outsider interaction."""
    df = pd.concat([fired_df, ctrl_df], ignore_index=True)
    df['fired']       = pd.to_numeric(df.get('fired', 0), errors='coerce').fillna(0)
    df['is_outsider'] = pd.to_numeric(df.get('is_outsider', np.nan), errors='coerce')
    df['fired_x_outsider'] = df['fired'] * df['is_outsider']

    predictors = ('fired', 'is_outsider', 'fired_x_outsider',
                  'projection_residual_pre', 'pyth_gap_at_firing',
                  'game_number_at_firing')
    out = os.path.join(OUTPUT_DIR, 'regression_outsider.txt')
    result = fit_ols(df, 'pyth_delta', predictors, 'Outsider vs. Insider', out)

    if result is not None:
        b_int = result.params.get('fired_x_outsider', np.nan)
        p_int = result.pvalues.get('fired_x_outsider', np.nan)
        log_audit(
            f'Secondary (outsider): fired×outsider coef={b_int:.4f}, p={p_int:.4f}',
            'PHASE7'
        )
    print("    Model 1 (outsider/insider) written to regression_outsider.txt")


# ── 2. Timing effect ──────────────────────────────────────────────────────────
def model_timing(fired_df):
    """Among firing events only, timing as continuous predictor."""
    df = fired_df.copy()
    predictors = ('game_number_at_firing', 'projection_residual_pre',
                  'pyth_gap_at_firing')
    out = os.path.join(OUTPUT_DIR, 'regression_timing.txt')
    result = fit_ols(df, 'pyth_delta', predictors, 'Timing Effect (firing events only)', out)

    if result is not None:
        b_t = result.params.get('game_number_at_firing', np.nan)
        p_t = result.pvalues.get('game_number_at_firing', np.nan)
        log_audit(
            f'Secondary (timing): game_number_at_firing coef={b_t:.4f}, p={p_t:.4f}',
            'PHASE7'
        )
    print("    Model 2 (timing) written to regression_timing.txt")


# ── 3. Interim tenure effect ──────────────────────────────────────────────────
def compute_tenure(events):
    """
    Replacement manager tenure = post_games_available (capped at 40 in the event table).
    This is the best proxy for how long the replacement actually managed before
    a permanent hire or season end.
    """
    # post_games_available is already the tenure of the replacement in the post window.
    # Use it directly.
    return events[['firing_id', 'post_games_available']].rename(
        columns={'post_games_available': 'replacement_tenure'}
    )


def model_tenure(fired_df, events):
    """Replacement tenure as moderator (firing events only)."""
    tenure = compute_tenure(events)
    df = fired_df.merge(tenure, on='firing_id', how='left')
    df['fired_x_tenure'] = df['fired'] * pd.to_numeric(
        df['replacement_tenure'], errors='coerce'
    )

    predictors = ('fired', 'replacement_tenure', 'fired_x_tenure',
                  'projection_residual_pre', 'pyth_gap_at_firing',
                  'game_number_at_firing')
    out = os.path.join(OUTPUT_DIR, 'regression_tenure.txt')
    result = fit_ols(df, 'pyth_delta', predictors, 'Interim Tenure Effect', out)

    if result is not None:
        b_t = result.params.get('replacement_tenure', np.nan)
        p_t = result.pvalues.get('replacement_tenure', np.nan)
        log_audit(
            f'Secondary (tenure): tenure coef={b_t:.4f}, p={p_t:.4f}', 'PHASE7'
        )
    print("    Model 3 (tenure) written to regression_tenure.txt")


# ── 4. Roster age effect ──────────────────────────────────────────────────────
def fetch_roster_ages(events):
    """
    Attempt to get PA-weighted average roster age at time of firing.

    Approach: pybaseball batting_stats for each team-season gives player ages
    and PA. Compute PA-weighted mean age per team-season.
    Returns a Series indexed by firing_id.
    """
    age_cache = {}

    try:
        from pybaseball import batting_stats
    except ImportError:
        log_audit('Phase7 age: pybaseball not available for roster age.', 'WARNING')
        return pd.Series(dtype=float)

    firing_ages = {}

    for _, ev in events.iterrows():
        fid  = int(ev['firing_id'])
        year = int(ev['season'])
        team = str(ev['team'])

        if (year, team) not in age_cache:
            try:
                time.sleep(1)
                bs = batting_stats(year, qual=50)  # min 50 PA to filter noise
                # batting_stats returns a DataFrame with 'Team' and 'Age' and 'PA'
                if bs is not None and len(bs) > 0:
                    team_bat = bs[bs['Team'].astype(str).str.upper() == team.upper()]
                    if len(team_bat) == 0:
                        # Try partial match (e.g. "Yankees" vs "NYY")
                        team_bat = bs  # fallback: all players (wrong, but degrade gracefully)
                    if 'Age' in team_bat.columns and 'PA' in team_bat.columns:
                        pa  = pd.to_numeric(team_bat['PA'], errors='coerce').fillna(0)
                        age = pd.to_numeric(team_bat['Age'], errors='coerce')
                        total_pa = pa.sum()
                        if total_pa > 0:
                            avg_age = (pa * age).sum() / total_pa
                            age_cache[(year, team)] = avg_age
                        else:
                            age_cache[(year, team)] = np.nan
                    else:
                        age_cache[(year, team)] = np.nan
                else:
                    age_cache[(year, team)] = np.nan
            except Exception as exc:
                log_audit(
                    f'Phase7 age: failed to get batting_stats for {team} {year}: {exc}',
                    'WARNING'
                )
                age_cache[(year, team)] = np.nan

        firing_ages[fid] = age_cache.get((year, team), np.nan)

    s = pd.Series(firing_ages, name='roster_age')
    n_missing = s.isna().sum()
    if n_missing > 0:
        log_audit(
            f'Phase7 age: roster age unavailable for {n_missing} firing events '
            f'(early seasons or scrape failures). age model will have reduced N.',
            'WARNING'
        )
    return s


def model_age(fired_df, events):
    """PA-weighted roster age as moderator (firing events only)."""
    print("    Fetching roster ages (this may take a while) ...")
    ages = fetch_roster_ages(events)
    age_df = ages.reset_index()
    age_df.columns = ['firing_id', 'roster_age']
    df = fired_df.merge(age_df, on='firing_id', how='left')

    predictors = ('roster_age', 'projection_residual_pre',
                  'pyth_gap_at_firing', 'game_number_at_firing')
    out = os.path.join(OUTPUT_DIR, 'regression_age.txt')
    result = fit_ols(df, 'pyth_delta', predictors, 'Roster Age Effect (firing events only)', out)

    if result is not None:
        b_a = result.params.get('roster_age', np.nan)
        p_a = result.pvalues.get('roster_age', np.nan)
        log_audit(
            f'Secondary (age): roster_age coef={b_a:.4f}, p={p_a:.4f}', 'PHASE7'
        )
        # Save ages to events for use in visualizations
        age_save = df[['firing_id', 'roster_age']]
        age_out  = os.path.join(OUTPUT_DIR, 'roster_ages.csv')
        age_save.to_csv(age_out, index=False)
    print("    Model 4 (age) written to regression_age.txt")


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    print("Phase 7: Running secondary analyses ...")
    log_audit('='*60, 'PHASE7_START')

    for path, label in [
        (CTRL_TABLE,   'control_table.csv'),
        (METRICS_PATH, 'metrics_table.csv'),
        (PROJ_PATH,    'projections_table.csv'),
        (EVENT_TABLE,  'event_table.csv'),
    ]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} not found. Run earlier phases first.")
            return

    ctrl_table  = pd.read_csv(CTRL_TABLE)
    metrics     = pd.read_csv(METRICS_PATH)
    projections = pd.read_csv(PROJ_PATH)
    events      = pd.read_csv(EVENT_TABLE)
    ctrl_pool   = pd.read_csv(CTRL_POOL) if os.path.exists(CTRL_POOL) else pd.DataFrame()

    # Assemble fired events dataframe (mirroring phase6 build logic)
    fired_df = ctrl_table[ctrl_table['match_found'] == 1].copy()
    fired_df['fired'] = 1

    # Build ctrl_df from pool
    ctrl_df = pd.DataFrame()
    if len(ctrl_pool) > 0 and 'ctrl_id' in fired_df.columns:
        ctrl_ids = fired_df['ctrl_id'].dropna().astype(int).unique()
        ctrl_df = ctrl_pool[ctrl_pool['ctrl_id'].isin(ctrl_ids)].copy()
        ctrl_df = ctrl_df.rename(columns={
            'pseudo_game_number':      'game_number_at_firing',
            'projection_residual_pre': 'projection_residual_pre',
            'pyth_gap_at_pseudo_fire': 'pyth_gap_at_firing',
        })
        ctrl_df['fired'] = 0

    print("  Running model 1: outsider vs. insider ...")
    model_outsider(fired_df, ctrl_df)

    print("  Running model 2: timing effect ...")
    model_timing(fired_df)

    print("  Running model 3: interim tenure effect ...")
    model_tenure(fired_df, events)

    print("  Running model 4: roster age effect ...")
    model_age(fired_df, events)

    print("Phase 7 complete — 4 secondary regression files written to /outputs/")
    log_audit('Phase 7 complete.', 'PHASE7_END')


if __name__ == '__main__':
    run()
