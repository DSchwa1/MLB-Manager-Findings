"""Phase 6: Primary OLS Regression.

Model:
  pyth_delta = β0 + β1(fired) + β2(projection_residual_pre)
             + β3(pyth_gap_at_firing) + β4(game_number_at_firing) + ε

Where:
  fired = 1 for firing events, 0 for matched controls.
  β1 is the primary finding: managerial change effect net of regression pressure.

Outputs: outputs/regression_primary.txt
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import log_audit, OUTPUT_DIR

CTRL_TABLE   = os.path.join(OUTPUT_DIR, 'control_table.csv')
CTRL_POOL    = os.path.join(OUTPUT_DIR, 'control_pool.csv')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics_table.csv')
PROJ_PATH    = os.path.join(OUTPUT_DIR, 'projections_table.csv')
OUT_PATH     = os.path.join(OUTPUT_DIR, 'regression_primary.txt')


def build_regression_df(ctrl_table, ctrl_pool, metrics, projections):
    """
    Assemble the combined fired + control dataframe for regression.

    Fired rows: firing events with matched controls (match_found==1).
    Control rows: the matched control observations.
    """
    # Firing event rows
    fired_rows = ctrl_table[ctrl_table['match_found'] == 1].copy()
    fired_rows['fired'] = 1

    # Required columns for fired rows
    # pyth_delta, projection_residual_pre, pyth_gap_at_firing,
    # game_number_at_firing already merged in ctrl_table from phase5

    # Control rows — pull from control pool using ctrl_id
    if ctrl_pool is not None and len(ctrl_pool) > 0 and 'ctrl_id' in fired_rows.columns:
        ctrl_ids = fired_rows['ctrl_id'].dropna().astype(int).unique()
        ctrl_rows = ctrl_pool[ctrl_pool['ctrl_id'].isin(ctrl_ids)].copy()
        ctrl_rows = ctrl_rows.rename(columns={
            'pseudo_game_number':      'game_number_at_firing',
            'projection_residual_pre': 'projection_residual_pre',
            'pyth_gap_at_pseudo_fire': 'pyth_gap_at_firing',
        })
        ctrl_rows['fired'] = 0
    else:
        ctrl_rows = pd.DataFrame()

    combined = pd.concat([fired_rows, ctrl_rows], ignore_index=True, sort=False)

    # Ensure numeric types
    for col in ('pyth_delta', 'projection_residual_pre',
                'pyth_gap_at_firing', 'game_number_at_firing', 'fired'):
        combined[col] = pd.to_numeric(combined[col], errors='coerce')

    return combined


def run_ols(df, outcome='pyth_delta',
            predictors=('fired', 'projection_residual_pre',
                        'pyth_gap_at_firing', 'game_number_at_firing'),
            label='Primary'):
    """
    Run OLS with statsmodels. Returns (model result, summary string).
    Drops rows with any NaN in outcome or predictors.
    """
    cols   = [outcome] + list(predictors)
    subset = df[cols].dropna()
    n_drop = len(df) - len(subset)
    if n_drop > 0:
        log_audit(
            f'Regression ({label}): dropped {n_drop} rows with NaN '
            f'in [{", ".join(cols)}].', 'REGRESSION_INFO'
        )

    if len(subset) < len(predictors) + 5:
        msg = (f'Regression ({label}): insufficient data after NA removal '
               f'({len(subset)} rows). Cannot fit model.')
        log_audit(msg, 'ERROR')
        return None, msg

    y = subset[outcome]
    X = sm.add_constant(subset[list(predictors)])
    model = sm.OLS(y, X).fit()
    return model, model.summary().as_text()


def run():
    print("Phase 6: Running primary OLS regression ...")
    log_audit('='*60, 'PHASE6_START')

    for path, label in [
        (CTRL_TABLE,   'control_table.csv'),
        (METRICS_PATH, 'metrics_table.csv'),
        (PROJ_PATH,    'projections_table.csv'),
    ]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} not found. Run earlier phases first.")
            return

    ctrl_table  = pd.read_csv(CTRL_TABLE)
    metrics     = pd.read_csv(METRICS_PATH)
    projections = pd.read_csv(PROJ_PATH)
    ctrl_pool   = pd.read_csv(CTRL_POOL) if os.path.exists(CTRL_POOL) else pd.DataFrame()

    df = build_regression_df(ctrl_table, ctrl_pool, metrics, projections)

    n_fired   = int((df['fired'] == 1).sum()) if 'fired' in df.columns else 0
    n_control = int((df['fired'] == 0).sum()) if 'fired' in df.columns else 0
    print(f"  Regression sample: {n_fired} fired events + {n_control} controls")

    result, summary_text = run_ols(df)

    output_lines = [
        'MLB Manager Firings — Primary OLS Regression',
        '='*70,
        '',
        'Model:',
        '  pyth_delta = β0 + β1(fired) + β2(projection_residual_pre)',
        '             + β3(pyth_gap_at_firing) + β4(game_number_at_firing) + ε',
        '',
        f'  fired=1 for managerial change events (N={n_fired})',
        f'  fired=0 for matched control observations (N={n_control})',
        '',
        'β1 is the primary finding: managerial change effect net of regression pressure.',
        '',
        '='*70,
        '',
        summary_text if summary_text else 'MODEL COULD NOT BE FIT — see data_audit.txt',
        '',
    ]

    if result is not None:
        b1     = result.params.get('fired', np.nan)
        se1    = result.bse.get('fired', np.nan)
        pval1  = result.pvalues.get('fired', np.nan)
        output_lines += [
            '='*70,
            'KEY FINDING (β1):',
            f'  Coefficient on fired : {b1:.4f}',
            f'  Std. Error           : {se1:.4f}',
            f'  p-value              : {pval1:.4f}',
            f'  R²                   : {result.rsquared:.4f}',
            '',
            ('  INTERPRETATION: A positive β1 indicates that managerial changes are '
             'associated with improved Pythagorean W% beyond what regression to the '
             'mean and pre-firing performance would predict.'),
        ]
        log_audit(
            f'Phase 6 primary regression: β1(fired)={b1:.4f}, SE={se1:.4f}, '
            f'p={pval1:.4f}, R²={result.rsquared:.4f}, N_fired={n_fired}, '
            f'N_ctrl={n_control}.', 'PHASE6_END'
        )
    else:
        log_audit('Phase 6: primary regression could not be fit.', 'ERROR')

    full_text = '\n'.join(output_lines)
    with open(OUT_PATH, 'w', encoding='utf-8') as fh:
        fh.write(full_text)

    print(f"Phase 6 complete — regression results written to regression_primary.txt")


if __name__ == '__main__':
    run()
