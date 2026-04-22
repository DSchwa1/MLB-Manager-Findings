"""Phase 9: Visualizations.

Produces five publication-ready charts exported as PNG to /outputs/charts/:

  improvement_distribution.png  — histogram of pyth_delta (firing events)
  fired_vs_control.png           — side-by-side mean pyth_delta with CIs
  timing_scatter.png             — game_number_at_firing vs. pyth_delta w/ trend
  outsider_insider.png           — box plot of pyth_delta by is_outsider
  roster_age.png                 — scatter of PA-weighted roster age vs. pyth_delta
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

from utils import log_audit, OUTPUT_DIR, CHARTS_DIR

CTRL_TABLE   = os.path.join(OUTPUT_DIR, 'control_table.csv')
CTRL_POOL    = os.path.join(OUTPUT_DIR, 'control_pool.csv')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics_table.csv')
ROSTER_AGES  = os.path.join(OUTPUT_DIR, 'roster_ages.csv')

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
PALETTE = {'Fired': '#c0392b', 'Control': '#2980b9'}
FIG_DPI = 150


def save_fig(fig, filename):
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {filename}")


# ── 1. Improvement distribution ───────────────────────────────────────────────
def chart_improvement_distribution(metrics):
    vals = metrics['pyth_delta'].dropna()
    if len(vals) == 0:
        log_audit('chart_improvement_distribution: no data', 'WARNING')
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=25, color='#c0392b', alpha=0.75, edgecolor='white', linewidth=0.5)

    mean_val   = vals.mean()
    median_val = vals.median()
    ax.axvline(mean_val,   color='#2c3e50', linestyle='--', linewidth=1.5,
               label=f'Mean = {mean_val:+.3f}')
    ax.axvline(median_val, color='#7f8c8d', linestyle=':',  linewidth=1.5,
               label=f'Median = {median_val:+.3f}')
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Pythagorean W% Δ (Post − Pre)', fontsize=12)
    ax.set_ylabel('Number of Firing Events', fontsize=12)
    ax.set_title('Distribution of Post-Firing Pythagorean W% Change\n(Mid-Season Managerial Firings, 2000–2025)', fontsize=13)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_fig(fig, 'improvement_distribution.png')


# ── 2. Fired vs. control comparison ───────────────────────────────────────────
def chart_fired_vs_control(ctrl_table, ctrl_pool):
    fired = ctrl_table[ctrl_table['match_found'] == 1]['pyth_delta'].dropna()

    ctrl_deltas = pd.Series(dtype=float)
    if ctrl_pool is not None and len(ctrl_pool) > 0:
        if 'ctrl_id' in ctrl_table.columns:
            ctrl_ids = ctrl_table['ctrl_id'].dropna().astype(int).unique()
            ctrl_deltas = ctrl_pool[ctrl_pool['ctrl_id'].isin(ctrl_ids)]['pyth_delta'].dropna()

    if len(fired) == 0 and len(ctrl_deltas) == 0:
        log_audit('chart_fired_vs_control: no data', 'WARNING')
        return

    groups = {}
    if len(fired)       > 0: groups['Fired'] = fired
    if len(ctrl_deltas) > 0: groups['Control'] = ctrl_deltas

    means = {k: v.mean() for k, v in groups.items()}
    sems  = {k: stats.sem(v) for k, v in groups.items()}
    ci95  = {k: 1.96 * sems[k] for k in groups}

    fig, ax = plt.subplots(figsize=(6, 5))
    x_pos   = list(range(len(groups)))
    labels  = list(groups.keys())
    colors  = [PALETTE.get(l, '#95a5a6') for l in labels]

    bars = ax.bar(x_pos, [means[l] for l in labels],
                  yerr=[ci95[l] for l in labels],
                  color=colors, alpha=0.85, edgecolor='white',
                  capsize=6, error_kw={'linewidth': 1.5})

    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f'{l}\n(n={len(groups[l])})' for l in labels], fontsize=11
    )
    ax.set_ylabel('Mean Pythagorean W% Δ (Post − Pre)', fontsize=11)
    ax.set_title('Mean Post-Firing Pythagorean W% Change\nFired Teams vs. Matched Controls (±95% CI)', fontsize=12)

    for bar, lbl in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci95[lbl] + 0.003,
                f'{means[lbl]:+.3f}', ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    save_fig(fig, 'fired_vs_control.png')


# ── 3. Timing scatter ─────────────────────────────────────────────────────────
def chart_timing_scatter(ctrl_table):
    df = ctrl_table[ctrl_table['match_found'] == 1][
        ['game_number_at_firing', 'pyth_delta']
    ].dropna()

    if len(df) < 5:
        log_audit('chart_timing_scatter: insufficient data', 'WARNING')
        return

    x = df['game_number_at_firing'].values
    y = df['pyth_delta'].values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color='#c0392b', alpha=0.55, s=35, edgecolors='none')

    slope, intercept, r, pval, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, intercept + slope * x_line,
            color='#2c3e50', linewidth=1.8,
            label=f'Trend: slope={slope:+.4f}, r={r:.2f}, p={pval:.3f}')
    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Games into Season at Firing', fontsize=12)
    ax.set_ylabel('Pythagorean W% Δ (Post − Pre)', fontsize=12)
    ax.set_title('Firing Timing vs. Post-Firing Performance Change\n(Earlier firings: more or less improvement?)', fontsize=12)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_fig(fig, 'timing_scatter.png')


# ── 4. Outsider vs. insider box plot ──────────────────────────────────────────
def chart_outsider_insider(ctrl_table):
    df = ctrl_table[ctrl_table['match_found'] == 1][
        ['is_outsider', 'pyth_delta']
    ].dropna()

    if len(df) < 5 or 'is_outsider' not in df.columns:
        log_audit('chart_outsider_insider: insufficient data', 'WARNING')
        return

    df['Hire Type'] = df['is_outsider'].map({0: 'Insider', 1: 'Outsider'})
    df = df.dropna(subset=['Hire Type'])

    fig, ax = plt.subplots(figsize=(6, 5))
    palette = {'Insider': '#2980b9', 'Outsider': '#c0392b'}
    sns.boxplot(data=df, x='Hire Type', y='pyth_delta',
                palette=palette, width=0.45, linewidth=1.2,
                flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5},
                ax=ax)
    sns.stripplot(data=df, x='Hire Type', y='pyth_delta',
                  palette=palette, size=3, alpha=0.35, jitter=True, ax=ax)

    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    n_counts = df['Hire Type'].value_counts()
    ticks = ax.get_xticklabels()
    ax.set_xticklabels([
        f'{t.get_text()}\n(n={n_counts.get(t.get_text(), 0)})'
        for t in ticks
    ])
    ax.set_xlabel('Replacement Manager Hire Type', fontsize=12)
    ax.set_ylabel('Pythagorean W% Δ (Post − Pre)', fontsize=12)
    ax.set_title('Post-Firing Performance by Hire Type\n(Insider vs. Outsider Replacement)', fontsize=12)

    # Annotate medians
    for i, ht in enumerate(['Insider', 'Outsider']):
        sub = df[df['Hire Type'] == ht]['pyth_delta']
        if len(sub) > 0:
            ax.text(i, sub.median() + 0.005, f'{sub.median():+.3f}',
                    ha='center', va='bottom', fontsize=9, color='black')

    fig.tight_layout()
    save_fig(fig, 'outsider_insider.png')


# ── 5. Roster age scatter ─────────────────────────────────────────────────────
def chart_roster_age(ctrl_table):
    if not os.path.exists(ROSTER_AGES):
        log_audit('chart_roster_age: roster_ages.csv not found (run phase7 first)', 'WARNING')
        print("    Skipping roster_age.png — roster_ages.csv not available.")
        return

    ages = pd.read_csv(ROSTER_AGES)
    df = ctrl_table[ctrl_table['match_found'] == 1][['firing_id', 'pyth_delta']].merge(
        ages, on='firing_id', how='inner'
    ).dropna()

    if len(df) < 5:
        log_audit('chart_roster_age: insufficient data after merge', 'WARNING')
        return

    x = df['roster_age'].values
    y = df['pyth_delta'].values

    slope, intercept, r, pval, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color='#27ae60', alpha=0.55, s=35, edgecolors='none')

    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, intercept + slope * x_line,
            color='#2c3e50', linewidth=1.8,
            label=f'Trend: slope={slope:+.4f}, r={r:.2f}, p={pval:.3f}')
    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('PA-Weighted Mean Roster Age', fontsize=12)
    ax.set_ylabel('Pythagorean W% Δ (Post − Pre)', fontsize=12)
    ax.set_title('Roster Age vs. Post-Firing Performance Change', fontsize=12)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_fig(fig, 'roster_age.png')


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    print("Phase 9: Generating visualizations ...")
    log_audit('='*60, 'PHASE9_START')

    for path, label in [
        (CTRL_TABLE,   'control_table.csv'),
        (METRICS_PATH, 'metrics_table.csv'),
    ]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} not found. Run earlier phases first.")
            return

    ctrl_table = pd.read_csv(CTRL_TABLE)
    metrics    = pd.read_csv(METRICS_PATH)
    ctrl_pool  = pd.read_csv(CTRL_POOL) if os.path.exists(CTRL_POOL) else pd.DataFrame()

    # Merge game_number_at_firing from metrics into ctrl_table if not present
    if 'game_number_at_firing' not in ctrl_table.columns and 'game_number_at_firing' in metrics.columns:
        ctrl_table = ctrl_table.merge(
            metrics[['firing_id', 'game_number_at_firing']], on='firing_id', how='left'
        )

    print("  Chart 1: improvement_distribution.png ...")
    chart_improvement_distribution(metrics)

    print("  Chart 2: fired_vs_control.png ...")
    chart_fired_vs_control(ctrl_table, ctrl_pool)

    print("  Chart 3: timing_scatter.png ...")
    chart_timing_scatter(ctrl_table)

    print("  Chart 4: outsider_insider.png ...")
    chart_outsider_insider(ctrl_table)

    print("  Chart 5: roster_age.png ...")
    chart_roster_age(ctrl_table)

    print(f"Phase 9 complete — charts written to /outputs/charts/")
    log_audit('Phase 9 complete.', 'PHASE9_END')


if __name__ == '__main__':
    run()
