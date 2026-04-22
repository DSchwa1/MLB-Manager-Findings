"""Phase 10: Summary Output.

Reads all prior outputs and writes summary_findings.txt with:
  - Sample counts and exclusions
  - Projection coverage
  - Control reuse summary
  - Primary regression β1
  - Mean pyth_delta for fired vs. control
  - Robustness check results
  - One-sentence top-line finding

Output: outputs/summary_findings.txt
"""

import os
import re
import numpy as np
import pandas as pd
from utils import log_audit, OUTPUT_DIR

EVENT_TABLE  = os.path.join(OUTPUT_DIR, 'event_table.csv')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics_table.csv')
PROJ_PATH    = os.path.join(OUTPUT_DIR, 'projections_table.csv')
CTRL_TABLE   = os.path.join(OUTPUT_DIR, 'control_table.csv')
CTRL_POOL    = os.path.join(OUTPUT_DIR, 'control_pool.csv')
REG_PRIMARY  = os.path.join(OUTPUT_DIR, 'regression_primary.txt')
ROBUSTNESS   = os.path.join(OUTPUT_DIR, 'robustness_checks.txt')
AUDIT_FILE   = os.path.join(os.path.dirname(OUTPUT_DIR), 'data_audit.txt')
OUT_PATH     = os.path.join(OUTPUT_DIR, 'summary_findings.txt')


def _safe_load(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def _read_text(path):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            return fh.read()
    return ''


def extract_b1_from_regression(reg_text):
    """Parse β1, SE, and p from the primary regression output file."""
    b1 = se1 = p1 = r2 = np.nan
    for line in reg_text.splitlines():
        if 'β1(fired)' in line or 'Coefficient on fired' in line:
            nums = re.findall(r'[-+]?\d*\.?\d+', line)
            if nums:
                b1 = float(nums[0])
        if 'Std. Error' in line and 'fired' not in line.lower():
            nums = re.findall(r'[-+]?\d*\.?\d+', line)
            if nums:
                se1 = float(nums[0])
        if 'p-value' in line:
            nums = re.findall(r'[-+]?\d*\.?\d+', line)
            if nums:
                p1 = float(nums[-1])
        if 'R²' in line or 'R-squared' in line:
            nums = re.findall(r'[-+]?\d*\.?\d+', line)
            if nums:
                r2 = float(nums[-1])

    # Also try statsmodels summary table format:  "fired   0.0123  0.0456  2.70  0.008"
    for line in reg_text.splitlines():
        if line.strip().startswith('fired'):
            parts = line.split()
            if len(parts) >= 5:
                try:
                    b1   = float(parts[1])
                    se1  = float(parts[2])
                    p1   = float(parts[4])
                except (ValueError, IndexError):
                    pass
            break

    return b1, se1, p1, r2


def count_audit_exclusions(audit_text):
    """Count EXCLUSION log entries in data_audit.txt."""
    return audit_text.count('[EXCLUSION]')


def run():
    print("Phase 10: Writing summary findings ...")
    log_audit('='*60, 'PHASE10_START')

    events   = _safe_load(EVENT_TABLE)
    metrics  = _safe_load(METRICS_PATH)
    proj     = _safe_load(PROJ_PATH)
    ctrl_tbl = _safe_load(CTRL_TABLE)
    ctrl_pool = _safe_load(CTRL_POOL)

    reg_text  = _read_text(REG_PRIMARY)
    rob_text  = _read_text(ROBUSTNESS)
    audit_txt = _read_text(AUDIT_FILE)

    # ── Sample counts ──────────────────────────────────────────────────────────
    n_events      = len(events)
    n_excluded    = count_audit_exclusions(audit_txt)
    n_truncated   = int(events['truncated_window'].sum()) if 'truncated_window' in events.columns else 0

    # ── Projection coverage ────────────────────────────────────────────────────
    n_proj_avail   = int(proj['projected_wpct'].notna().sum())  if len(proj) > 0 else 0
    n_proj_total   = len(proj)
    proj_src_counts = (proj['projection_source'].value_counts().to_dict()
                       if len(proj) > 0 else {})

    seasons_coverage = {}
    if len(proj) > 0:
        for season, grp in proj.groupby('season'):
            n_avail = grp['projected_wpct'].notna().sum()
            n_tot   = len(grp)
            seasons_coverage[int(season)] = f'{n_avail}/{n_tot}'

    full_cov_seasons    = [y for y, v in seasons_coverage.items() if v.split('/')[0] == v.split('/')[1]]
    partial_cov_seasons = [y for y, v in seasons_coverage.items() if v.split('/')[0] != v.split('/')[1]]

    # ── Control reuse ─────────────────────────────────────────────────────────
    n_reused_controls = 0
    if len(ctrl_tbl) > 0 and 'ctrl_id' in ctrl_tbl.columns:
        ctrl_id_counts = ctrl_tbl[ctrl_tbl['match_found'] == 1]['ctrl_id'].value_counts()
        n_reused_controls = int((ctrl_id_counts > 1).sum())

    # ── Pyth_delta summaries ───────────────────────────────────────────────────
    mean_fired_pyth_delta = np.nan
    mean_ctrl_pyth_delta  = np.nan

    if len(metrics) > 0 and 'pyth_delta' in metrics.columns:
        mean_fired_pyth_delta = metrics['pyth_delta'].dropna().mean()

    if len(ctrl_pool) > 0 and 'pyth_delta' in ctrl_pool.columns:
        # Only control obs actually matched to a firing event
        if len(ctrl_tbl) > 0 and 'ctrl_id' in ctrl_tbl.columns:
            matched_ctrl_ids = ctrl_tbl[ctrl_tbl['match_found'] == 1]['ctrl_id'].dropna().astype(int).unique()
            mean_ctrl_pyth_delta = (
                ctrl_pool[ctrl_pool['ctrl_id'].isin(matched_ctrl_ids)]['pyth_delta']
                .dropna().mean()
            )

    # ── Primary regression β1 ─────────────────────────────────────────────────
    b1, se1, p1, r2 = extract_b1_from_regression(reg_text)

    # ── Robustness check verdicts ──────────────────────────────────────────────
    rob_checks = []
    for line in rob_text.splitlines():
        if line.strip().startswith('Check'):
            check_label = line.strip()
        if 'β1(fired)' in line:
            nums = re.findall(r'[+-]?\d+\.\d+', line)
            b1_r = float(nums[0]) if nums else np.nan
            p_r  = float(nums[2]) if len(nums) > 2 else np.nan
            rob_checks.append({
                'label':     check_label if 'check_label' in dir() else 'Unknown',
                'b1':        b1_r,
                'p':         p_r,
                'dir_holds': 'YES' if b1_r > 0 else 'NO',
                'sig':       'YES' if p_r < 0.05 else 'NO',
            })

    all_dir_hold = all(r['dir_holds'] == 'YES' for r in rob_checks) if rob_checks else None
    all_sig      = all(r['sig']       == 'YES' for r in rob_checks) if rob_checks else None

    # ── Top-line finding ───────────────────────────────────────────────────────
    if not np.isnan(b1) and not np.isnan(p1):
        direction = "positive" if b1 > 0 else "negative"
        sig_word  = "statistically significant" if p1 < 0.05 else "not statistically significant"
        topline = (
            f"After controlling for regression to the mean and pre-firing performance, "
            f"mid-season managerial changes are associated with a {direction} change in "
            f"Pythagorean W% (β1 = {b1:+.3f}, p = {p1:.3f}), which is {sig_word}."
        )
    else:
        topline = (
            "Primary regression could not be evaluated — insufficient data "
            "or model fitting error (see data_audit.txt)."
        )

    # ── Build output text ──────────────────────────────────────────────────────
    lines = [
        'MLB Manager Firings — Summary Findings',
        '='*70,
        f'Generated from data covering MLB 2000–2025.',
        '',
        '── SAMPLE ──────────────────────────────────────────────────────────────',
        f'Total firing events in final sample    : {n_events}',
        f'Events excluded (see data_audit.txt)   : {n_excluded}',
        f'Truncated-window cases (flag=1)        : {n_truncated}',
        '',
        '── PROJECTION COVERAGE ─────────────────────────────────────────────────',
        f'Total events with projection data      : {n_proj_avail} / {n_proj_total}',
        f'Sources used                           : {proj_src_counts}',
        f'Seasons with full coverage             : {sorted(full_cov_seasons)}',
        f'Seasons with partial/no coverage       : {sorted(partial_cov_seasons)}',
        '',
        '── CONTROL GROUP ───────────────────────────────────────────────────────',
        f'Firing events matched to a control     : {int(ctrl_tbl["match_found"].sum()) if len(ctrl_tbl) > 0 else "N/A"}',
        f'Firing events without a match          : {int((ctrl_tbl["match_found"] == 0).sum()) if len(ctrl_tbl) > 0 else "N/A"}',
        f'Reused control obs (ctrl_id used >1x)  : {n_reused_controls}',
        '',
        '── PERFORMANCE ─────────────────────────────────────────────────────────',
        f'Mean pyth_delta, fired teams           : {mean_fired_pyth_delta:+.4f}' if not np.isnan(mean_fired_pyth_delta) else 'Mean pyth_delta, fired teams           : N/A',
        f'Mean pyth_delta, matched controls      : {mean_ctrl_pyth_delta:+.4f}'  if not np.isnan(mean_ctrl_pyth_delta)  else 'Mean pyth_delta, matched controls      : N/A',
        '',
        '── PRIMARY REGRESSION ───────────────────────────────────────────────────',
        f'β1 (fired effect)                      : {b1:+.4f}' if not np.isnan(b1) else 'β1 (fired effect)                      : N/A',
        f'Standard Error                         : {se1:.4f}'  if not np.isnan(se1) else 'Standard Error                         : N/A',
        f'p-value                                : {p1:.4f}'   if not np.isnan(p1)  else 'p-value                                : N/A',
        f'R²                                     : {r2:.4f}'   if not np.isnan(r2)  else 'R²                                     : N/A',
        '',
        '── ROBUSTNESS CHECKS ───────────────────────────────────────────────────',
    ]

    if rob_checks:
        lines.append(f'  {"Check":<45} {"β1":>8} {"p":>8} {"Dir?":<6} {"Sig?":<6}')
        lines.append('  ' + '-'*74)
        for r in rob_checks:
            b1_s = f'{r["b1"]:+.4f}' if not np.isnan(r['b1']) else '  N/A  '
            p_s  = f'{r["p"]:.4f}'   if not np.isnan(r['p'])  else '  N/A  '
            lines.append(f'  {r["label"]:<45} {b1_s:>8} {p_s:>8} {r["dir_holds"]:<6} {r["sig"]:<6}')
        lines += [
            '',
            f'β1 direction holds across all checks   : {"YES" if all_dir_hold else "NO" if all_dir_hold is not None else "N/A"}',
            f'β1 significant across all checks       : {"YES" if all_sig else "NO" if all_sig is not None else "N/A"}',
        ]
    else:
        lines.append('  Robustness check output not found or could not be parsed.')

    lines += [
        '',
        '── TOP-LINE FINDING ─────────────────────────────────────────────────────',
        topline,
        '',
        '── CAVEATS ──────────────────────────────────────────────────────────────',
        '  1. is_outsider classifications are automated and require manual verification.',
        '     See data_audit.txt for all VERIFY_OUTSIDER entries.',
        '  2. Projection coverage is limited pre-2006 and patchy 2006–2014.',
        '     Residual-based analyses should be interpreted as scoped to years with data.',
        '  3. Control reuse (if any) may inflate effective sample size for matched',
        '     comparisons. Sensitivity to reuse is flagged in data_audit.txt.',
        '  4. BBRef scraping is subject to site structure changes. Verify all',
        '     game-log data for completeness before final publication.',
        '',
        '='*70,
    ]

    full_text = '\n'.join(lines)
    with open(OUT_PATH, 'w', encoding='utf-8') as fh:
        fh.write(full_text)

    print(f"Phase 10 complete — summary written to summary_findings.txt")
    log_audit(f'Phase 10 complete. Top-line: {topline}', 'PHASE10_END')


if __name__ == '__main__':
    run()
